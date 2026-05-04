import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import certifi
from databricks import sql
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field, model_validator

from prompts import (
    DEFAULT_PROMPT_ID,
    PROMPTS,
    PromptSpec,
    build_user_prompt,
    get_prompt,
)


# Allowed values for self-tagged admin level on each affected location.
# Kept as a Literal-like list (not a Pydantic Literal) so the prompt can
# still negotiate values the schema accepts as a fallback.
LOCATION_ADMIN_LEVELS = (
    "point",           # building / facility / single named structure
    "neighbourhood",   # sub-city: ward, suburb, quarter
    "city",            # town / city / village / hamlet
    "admin2",          # county / district / department / municipality
    "admin1",          # state / province / region / oblast
    "country",         # nation
    "coastal_zone",    # named coast / shoreline ("Andaman coast")
    "basin",           # river basin / watershed / lake
    "feature",         # geological/oceanic feature (fault, subduction zone)
    "unknown",
)

REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
REASONING_EFFORT_ARG_DEFAULT = object()


class EventExtraction(BaseModel):
    is_single_actual_event: bool = Field(
        description=(
            "True only when the document's primary subject is one specific disaster"
            " event, OR a clearly delineated section devotes substantive coverage"
            " (multi-sentence narrative, named places, named impacts, named dates)"
            " to one specific event. False is allowed for broader, policy-oriented,"
            " or multi-event documents — the remaining fields should still hold the"
            " best candidate event mention if one is present."
        )
    )
    is_verifiable_event: bool = Field(
        description=(
            "Permissive candidate flag for downstream collection. True only when the"
            " extraction has a hazard, at least one event date, at least one location"
            " that satisfies the Phase 2 location rules (sub-national hazard scope,"
            " country justification, technological/Chernobyl rule), and confidence is"
            " high or medium."
        )
    )
    event_hazard: Optional[str] = Field(
        default=None,
        description="Hazard type explicitly stated in the text, such as Flood or Earthquake.",
    )
    event_dates: List[str] = Field(
        default_factory=list,
        description=(
            "Event timing only (NOT publication / workshop / report dates)."
            " Prefer YYYY-MM-DD; allow YYYY-MM, YYYY, or a short source phrase such"
            " as 'monsoon season 2017' when that is the most precise form the text"
            " supports."
        ),
    )
    affected_locations: List[str] = Field(
        default_factory=list,
        description=(
            "Specific locations explicitly said to be affected by THIS event. Prefer"
            " granular places (city, district, neighbourhood, facility). Apply the"
            " strict Phase 2 location rules: bare country only when justified;"
            " technological / nuclear / industrial events restricted to facility +"
            " exclusion zone + directly-impacted settlements (no downwind/fall-out"
            " countries); no country+sub-region duplication. If the most granular"
            " location is a sub-building / vehicle / room / pier / hangar that is"
            " too specific to find on a map, ALSO include its parent city or district"
            " as a SEPARATE item, so downstream geocoding has a polygon fallback."
        ),
    )
    location_admin_levels: List[str] = Field(
        default_factory=list,
        description=(
            "PARALLEL to affected_locations (same length, same order). For each entry,"
            " self-tag the most specific admin level the location refers to, picked from:"
            " 'point' (building/facility), 'neighbourhood', 'city', 'admin2'"
            " (county/district), 'admin1' (state/province/region), 'country',"
            " 'coastal_zone' (e.g. 'Andaman coast'), 'basin' (river/lake basin),"
            " 'feature' (geological/oceanic feature), or 'unknown' as a last resort."
            " You MUST emit one tag per location, in the same order."
        ),
    )
    key_impacts: List[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 concise impact statements explicitly stated in the text. Include"
            " qualitative impacts when numbers are missing (e.g. 'homes damaged',"
            " 'communities displaced', 'agricultural losses reported'). Do not invent"
            " numbers."
        ),
    )
    event_confidence: Optional[str] = Field(
        default=None,
        description=(
            "Extraction confidence: 'high' (hazard + valid location + date + at least one"
            " impact, all explicit), 'medium' (hazard + valid location explicit; date or"
            " impacts partial), 'low' (event real but evidence thin, only passing mention,"
            " or any Phase 2 rule forced fields to be left empty), or null when no"
            " candidate event exists."
        ),
    )
    evidence_snippets: List[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 short snippets or paraphrases (<= 25 words each) from the source"
            " that justify the extraction. Paraphrase if needed; no long quotes."
        ),
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description=(
            "Missing or weak elements, e.g. 'exact date missing', 'impact numbers"
            " missing', 'only country-level location but country rule failed', 'only"
            " passing mention', 'downwind countries dropped per Chernobyl rule',"
            " 'hazard type broad'."
        ),
    )

    @model_validator(mode="after")
    def _align_admin_levels(self) -> "EventExtraction":
        """Pad/truncate location_admin_levels so it lines up with affected_locations.

        Pydantic + the OpenAI structured-output decoder normally enforce list
        lengths, but the LLM can still emit a mismatched parallel list. We
        normalise here rather than crash: missing tags become 'unknown',
        extra tags are dropped, and any non-allowed value is coerced to
        'unknown'. This keeps downstream code simple (always-present, always-
        same-length), at the cost of silently rescuing schema drift — log it
        if you want to track LLM compliance.
        """
        n = len(self.affected_locations)
        levels = list(self.location_admin_levels)
        if len(levels) < n:
            levels.extend(["unknown"] * (n - len(levels)))
        elif len(levels) > n:
            levels = levels[:n]
        levels = [
            (lvl if lvl in LOCATION_ADMIN_LEVELS else "unknown") for lvl in levels
        ]
        # Pydantic v2: assign through __dict__ to avoid revalidation loop.
        object.__setattr__(self, "location_admin_levels", levels)
        return self


@dataclass
class DocumentRow:
    asset_key: str
    source_id: Optional[int]
    attachment_id: Optional[int]
    asset_url: Optional[str]
    asset_type: Optional[str]
    status: Optional[str]
    extraction_method: Optional[str]
    char_count: Optional[int]
    text: str
    last_processed_at: Optional[str]
    title: Optional[str]
    publication_year: Optional[str]
    language: Optional[str]
    countries: Optional[str]
    themes: Optional[str]
    hazards: Optional[str]
    organizations: Optional[str]
    content_url: Optional[str]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def connect_databricks():
    ca_file = os.getenv("DATABRICKS_CA_FILE", certifi.where())
    return sql.connect(
        server_hostname=require_env("DATABRICKS_SERVER_HOSTNAME"),
        http_path=require_env("DATABRICKS_HTTP_PATH"),
        access_token=require_env("DATABRICKS_TOKEN"),
        _tls_trusted_ca_file=ca_file,
    )

def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=require_env("AZURE_OPENAI_API_KEY"),
        azure_deployment=require_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=require_env("API_VERSION"),
    )

def smoke_test_databricks(limit: int) -> None:
    asset_table = os.getenv(
        "DATABRICKS_SOURCE_ASSET_TABLE",
        "undrr_d_datahub_databricks_ws.gold.asset_texts",
    )

    queries = {
        "context": "SELECT current_catalog(), current_schema()",
        "sample": f"""
            SELECT
                asset_key,
                asset_type,
                char_count,
                CAST(last_processed_at AS STRING) AS last_processed_at
            FROM {asset_table}
            WHERE text IS NOT NULL
              AND LENGTH(TRIM(text)) > 0
            ORDER BY last_processed_at DESC
            LIMIT {int(limit)}
        """,
    }

    with connect_databricks() as connection:
        with connection.cursor() as cursor:
            logging.info("Checking current catalog/schema")
            cursor.execute(queries["context"])
            context_rows = cursor.fetchall()
            logging.info("Current context: %s", context_rows)

            logging.info("Running Databricks smoke test with table %s", asset_table)
            cursor.execute(queries["sample"])
            rows = cursor.fetchall()

    logging.info("Smoke test succeeded. Retrieved %s rows.", len(rows))
    for i, row in enumerate(rows[:5], start=1):
        logging.info("Preview %s: %s", i, row)

def inspect_databricks_objects() -> None:
    with connect_databricks() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_catalog(), current_schema()")
            print("CONTEXT:", cursor.fetchall())

            cursor.execute("SHOW SCHEMAS IN undrr_d_datahub_databricks_ws")
            schemas = cursor.fetchall()
            print("\nSCHEMAS:")
            for row in schemas:
                print(row)

            # Try likely schemas
            for schema_name in ["gold", "silver", "bronze", "default"]:
                try:
                    cursor.execute(f"SHOW TABLES IN undrr_d_datahub_databricks_ws.{schema_name}")
                    rows = cursor.fetchall()
                    print(f"\nTABLES IN {schema_name}:")
                    for row in rows[:50]:
                        print(row)
                except Exception as e:
                    print(f"\nCould not inspect schema {schema_name}: {e}")

def fetch_documents(limit: int) -> List[DocumentRow]:
    asset_table = os.getenv(
        "DATABRICKS_SOURCE_ASSET_TABLE",
        "undrr_d_datahub_databricks_ws.gold.asset_texts",
    )
    chunk_table = os.getenv(
        "DATABRICKS_SOURCE_CHUNK_TABLE",
        "undrr_d_datahub_databricks_ws.gold.asset_text_chunks",
    )
    query = f"""
    WITH chunk_meta AS (
        SELECT
            asset_key,
            MAX(title) AS title,
            MAX(publication_year) AS publication_year,
            MAX(language) AS language,
            MAX(countries) AS countries,
            MAX(themes) AS themes,
            MAX(hazards) AS hazards,
            MAX(organizations) AS organizations,
            MAX(content_url) AS content_url
        FROM {chunk_table}
        GROUP BY asset_key
    )
    SELECT
        a.asset_key,
        a.source_id,
        a.attachment_id,
        a.asset_url,
        a.asset_type,
        a.status,
        a.extraction_method,
        a.char_count,
        a.text,
        CAST(a.last_processed_at AS STRING) AS last_processed_at,
        m.title,
        m.publication_year,
        m.language,
        m.countries,
        m.themes,
        m.hazards,
        m.organizations,
        m.content_url
    FROM {asset_table} a
    LEFT JOIN chunk_meta m
        ON a.asset_key = m.asset_key
    WHERE a.text IS NOT NULL
      AND LENGTH(TRIM(a.text)) > 0
    ORDER BY a.last_processed_at DESC
    LIMIT {int(limit)}
    """

    documents: List[DocumentRow] = []
    with connect_databricks() as connection:
        with connection.cursor() as cursor:
            logging.info("Running Databricks query for up to %s documents", limit)
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

    for row in rows:
        row_dict = dict(zip(columns, row))
        documents.append(DocumentRow(**row_dict))

    logging.info("Fetched %s documents from Databricks", len(documents))
    return documents


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_prompt_payload(doc: DocumentRow, max_chars: int) -> dict:
    return {
        "document_metadata": {
            "asset_key": doc.asset_key,
            "source_id": doc.source_id,
            "attachment_id": doc.attachment_id,
            "asset_url": doc.asset_url,
            "content_url": doc.content_url,
            "asset_type": doc.asset_type,
            "title": doc.title,
            "publication_year": doc.publication_year,
            "language": doc.language,
            "countries": doc.countries,
            "themes": doc.themes,
            "hazards": doc.hazards,
            "organizations": doc.organizations,
            "last_processed_at": doc.last_processed_at,
        },
        "document_text": trim_text(doc.text, max_chars=max_chars),
    }


def extract_document_with_openai(
    client: AzureOpenAI,
    doc: DocumentRow,
    max_chars: int,
    model: str,
    reasoning_effort: Optional[str],
    prompt: PromptSpec,
) -> EventExtraction:
    payload = build_prompt_payload(doc, max_chars=max_chars)
    completion_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": build_user_prompt(prompt, payload)},
        ],
        "response_format": EventExtraction,
    }
    if reasoning_effort:
        completion_kwargs["reasoning_effort"] = reasoning_effort

    completion = client.beta.chat.completions.parse(**completion_kwargs)
    return completion.choices[0].message.parsed


def save_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "asset_key",
        "title",
        "publication_year",
        "language",
        "countries",
        "themes",
        "hazards",
        "organizations",
        "content_url",
        "is_single_actual_event",
        "is_verifiable_event",
        "event_hazard",
        "event_dates",
        "affected_locations",
        "location_admin_levels",
        "key_impacts",
        "event_confidence",
        "evidence_snippets",
        "missing_information",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def process_documents(
    documents: List[DocumentRow],
    max_chars: int,
    output_dir: Path,
    model: str,
    reasoning_effort: Optional[str],
    prompt: PromptSpec,
    client: AzureOpenAI,
) -> None:
    output_rows = []

    logging.info(
        "Running prompt %02d (%s) on %s documents",
        prompt.id,
        prompt.name,
        len(documents),
    )

    for idx, doc in enumerate(documents, start=1):
        logging.info(
            "Processing document %s/%s | asset_key=%s | title=%s",
            idx,
            len(documents),
            doc.asset_key,
            doc.title,
        )
        try:
            extraction = extract_document_with_openai(
                client=client,
                doc=doc,
                max_chars=max_chars,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=prompt,
            )
        except Exception as exc:
            logging.exception("OpenAI extraction failed for asset_key=%s: %s", doc.asset_key, exc)
            output_rows.append(
                {
                    "asset_key": doc.asset_key,
                    "title": doc.title,
                    "publication_year": doc.publication_year,
                    "language": doc.language,
                    "countries": doc.countries,
                    "themes": doc.themes,
                    "hazards": doc.hazards,
                    "organizations": doc.organizations,
                    "content_url": doc.content_url,
                    "is_single_actual_event": None,
                    "is_verifiable_event": None,
                    "event_hazard": None,
                    "event_dates": "",
                    "affected_locations": "",
                    "location_admin_levels": "",
                    "key_impacts": f"ERROR: {exc}",
                    "event_confidence": None,
                    "evidence_snippets": "",
                    "missing_information": "",
                }
            )
            continue

        output_rows.append(
            {
                "asset_key": doc.asset_key,
                "title": doc.title,
                "publication_year": doc.publication_year,
                "language": doc.language,
                "countries": doc.countries,
                "themes": doc.themes,
                "hazards": doc.hazards,
                "organizations": doc.organizations,
                "content_url": doc.content_url,
                "is_single_actual_event": extraction.is_single_actual_event,
                "is_verifiable_event": extraction.is_verifiable_event,
                "event_hazard": extraction.event_hazard,
                "event_dates": " | ".join(extraction.event_dates),
                "affected_locations": " | ".join(extraction.affected_locations),
                "location_admin_levels": " | ".join(extraction.location_admin_levels),
                "key_impacts": " | ".join(extraction.key_impacts),
                "event_confidence": extraction.event_confidence,
                "evidence_snippets": " | ".join(extraction.evidence_snippets),
                "missing_information": " | ".join(extraction.missing_information),
            }
        )

    save_jsonl(output_rows, output_dir / "event_extractions.jsonl")
    save_csv(output_rows, output_dir / "event_extractions.csv")

    logging.info("Finished. Wrote:")
    logging.info("  - %s", output_dir / "event_extractions.jsonl")
    logging.info("  - %s", output_dir / "event_extractions.csv")


def slugify_model(model: str) -> str:
    slug = model.strip().lower().replace(".", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"(?<=gpt)-(?=\d)", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "model"


def reasoning_slug(reasoning_effort: Optional[str]) -> str:
    return reasoning_effort or "default"


def all_mode_output_dir(
    output_root: Path,
    limit: int,
    model: str,
    reasoning_effort: Optional[str],
    prompt: PromptSpec,
) -> Path:
    return output_root / (
        f"test_{limit}_{slugify_model(model)}_"
        f"{reasoning_slug(reasoning_effort)}_prompt{prompt.id:02d}"
    )


def list_prompts() -> None:
    for prompt in PROMPTS:
        print(f"{prompt.id:02d}: {prompt.name}")


def parse_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None

    if normalized not in REASONING_EFFORT_VALUES:
        allowed = ", ".join(REASONING_EFFORT_VALUES)
        raise argparse.ArgumentTypeError(
            f"invalid reasoning effort {value!r}; expected one of: {allowed}"
        )

    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple UNDRR event-focused extractor")
    parser.add_argument("--limit", type=int, default=1000, help="How many Databricks documents to process.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum number of characters from each document to send to OpenAI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/event_extractions"),
        help="Folder where JSONL and CSV outputs will be saved.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder for prompt-specific output directories when --all is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=parse_reasoning_effort,
        default=REASONING_EFFORT_ARG_DEFAULT,
        metavar="{none,minimal,low,medium,high,xhigh}",
        help=(
            "Optional reasoning effort for reasoning-capable OpenAI models. "
            "Defaults to AZURE_OPENAI_REASONING_EFFORT when set."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=DEFAULT_PROMPT_ID,
        help="Numbered prompt to use for a single run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every prompt with the same documents, model, and reasoning effort.",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available numbered prompts and exit.",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Only test Databricks connectivity and document fetch, without calling OpenAI.",
    )
    args = parser.parse_args()
    if args.reasoning_effort is REASONING_EFFORT_ARG_DEFAULT:
        try:
            args.reasoning_effort = parse_reasoning_effort(
                os.getenv("AZURE_OPENAI_REASONING_EFFORT")
            )
        except argparse.ArgumentTypeError as exc:
            parser.error(f"invalid AZURE_OPENAI_REASONING_EFFORT: {exc}")

    if not args.all and not args.list_prompts and not args.db_only:
        try:
            get_prompt(args.prompt)
        except ValueError as exc:
            parser.error(str(exc))

    return args


def main() -> None:
    load_dotenv()
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    args = parse_args()

    if args.list_prompts:
        list_prompts()
        return

    if args.db_only:
        smoke_test_databricks(limit=args.limit)
        return

    client = get_azure_openai_client()
    documents = fetch_documents(limit=args.limit)

    if args.all:
        for prompt in PROMPTS:
            output_dir = all_mode_output_dir(
                output_root=args.output_root,
                limit=args.limit,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                prompt=prompt,
            )
            process_documents(
                documents=documents,
                max_chars=args.max_chars,
                output_dir=output_dir,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                prompt=prompt,
                client=client,
            )
        return

    prompt = get_prompt(args.prompt)
    process_documents(
        documents=documents,
        max_chars=args.max_chars,
        output_dir=args.output_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt=prompt,
        client=client,
    )


if __name__ == "__main__":
    main()
