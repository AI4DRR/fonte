import argparse
import csv
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import certifi
from databricks import sql
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field, model_validator

from groundsource.prompts import (
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

OUTPUT_HEADERS = [
    "asset_key",
    "title",
    "publication_year",
    "language",
    "countries",
    "themes",
    "hazards",
    "organizations",
    "content_url",
    "event_index",
    "event_count",
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


class EventExtraction(BaseModel):
    is_single_actual_event: bool = Field(
        description=(
            "True only when the document's primary subject is one specific disaster"
            " event, OR a clearly delineated section devotes substantive coverage"
            " (multi-sentence narrative, named places, named impacts, named dates)"
            " to this specific event. False is allowed for broader, policy-oriented,"
            " or multi-event documents."
        )
    )
    is_verifiable_event: bool = Field(
        description=(
            "True only when this event has a hazard, at least one event date, at least"
            " one location that satisfies the Phase 2 location rules (sub-national"
            " hazard scope, country justification, technological/Chernobyl rule), and"
            " confidence is high or medium. Returned events should normally all be"
            " verifiable."
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
            " impacts partial), or 'low' (event real but evidence thin, only passing"
            " mention, or any Phase 2 rule forced fields to be left empty). Multi-event"
            " extraction should return only high/medium events."
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


class DocumentEventExtractions(BaseModel):
    events: List[EventExtraction] = Field(
        default_factory=list,
        description=(
            "All distinct, source-supported, verifiable disaster/hazard events found in"
            " the document. Include only high- or medium-confidence events with explicit"
            " hazard, event timing, and valid directly affected locations. Return an"
            " empty list when no event satisfies those reliability requirements."
        ),
    )

    @model_validator(mode="after")
    def _keep_verified_unique_events(self) -> "DocumentEventExtractions":
        """Keep the output focused on reliable event records.

        The prompts ask for verifiable high/medium events only. This post-check makes
        the contract robust if a model still emits a low-confidence or under-specified
        candidate.
        """
        kept: list[EventExtraction] = []
        seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        for event in self.events:
            if not event.is_verifiable_event:
                continue
            if event.event_confidence not in {"high", "medium"}:
                continue
            if not event.event_hazard or not event.event_dates or not event.affected_locations:
                continue
            key = (
                event.event_hazard.strip().lower(),
                tuple(s.strip().lower() for s in event.event_dates if s.strip()),
                tuple(s.strip().lower() for s in event.affected_locations if s.strip()),
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append(event)

        object.__setattr__(self, "events", kept)
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
            ORDER BY last_processed_at DESC, asset_key ASC
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
    ORDER BY a.last_processed_at DESC, a.asset_key ASC
    LIMIT {int(limit)}
    """

    started_at = time.perf_counter()
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

    elapsed = time.perf_counter() - started_at
    logging.info("Fetched %s documents from Databricks in %.1fs", len(documents), elapsed)
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
) -> DocumentEventExtractions:
    payload = build_prompt_payload(doc, max_chars=max_chars)
    completion_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": build_user_prompt(prompt, payload)},
        ],
        "response_format": DocumentEventExtractions,
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _base_output_row(doc: DocumentRow) -> dict:
    return {
        "asset_key": doc.asset_key,
        "title": doc.title,
        "publication_year": doc.publication_year,
        "language": doc.language,
        "countries": doc.countries,
        "themes": doc.themes,
        "hazards": doc.hazards,
        "organizations": doc.organizations,
        "content_url": doc.content_url,
    }


def extraction_to_output_rows(doc: DocumentRow, extraction: DocumentEventExtractions) -> list[dict]:
    event_count = len(extraction.events)
    if event_count == 0:
        return [{
            **_base_output_row(doc),
            "event_index": "",
            "event_count": 0,
            "is_single_actual_event": False,
            "is_verifiable_event": False,
            "event_hazard": None,
            "event_dates": "",
            "affected_locations": "",
            "location_admin_levels": "",
            "key_impacts": "",
            "event_confidence": None,
            "evidence_snippets": "",
            "missing_information": "no verified reliable event extracted",
        }]

    rows: list[dict] = []
    for idx, event in enumerate(extraction.events, start=1):
        rows.append({
            **_base_output_row(doc),
            "event_index": idx,
            "event_count": event_count,
            "is_single_actual_event": event.is_single_actual_event,
            "is_verifiable_event": event.is_verifiable_event,
            "event_hazard": event.event_hazard,
            "event_dates": " | ".join(event.event_dates),
            "affected_locations": " | ".join(event.affected_locations),
            "location_admin_levels": " | ".join(event.location_admin_levels),
            "key_impacts": " | ".join(event.key_impacts),
            "event_confidence": event.event_confidence,
            "evidence_snippets": " | ".join(event.evidence_snippets),
            "missing_information": " | ".join(event.missing_information),
        })
    return rows


def error_output_rows(doc: DocumentRow, exc: Exception) -> list[dict]:
    return [{
        **_base_output_row(doc),
        "event_index": "",
        "event_count": 0,
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
    }]


def is_error_row(row: dict) -> bool:
    return str(row.get("key_impacts") or "").startswith("ERROR:")


def load_completed_rows(output_dir: Path) -> dict[str, list[dict]]:
    progress_path = output_dir / "event_extractions_progress.jsonl"
    final_path = output_dir / "event_extractions.jsonl"
    source_path = progress_path if progress_path.exists() else final_path
    if not source_path.exists():
        return {}

    rows: dict[str, list[dict]] = {}
    with source_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed resume row %s:%s: %s", source_path, line_no, exc)
                continue
            asset_key = row.get("asset_key")
            if asset_key:
                rows.setdefault(asset_key, []).append(row)
    logging.info("Loaded %s completed rows from %s", len(rows), source_path)
    return rows


def append_progress_rows(rows: list[dict], path: Path, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_documents(
    documents: List[DocumentRow],
    max_chars: int,
    output_dir: Path,
    model: str,
    reasoning_effort: Optional[str],
    prompt: PromptSpec,
    client: Optional[AzureOpenAI],
    workers: int = 1,
    resume: bool = False,
) -> None:
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "event_extractions_progress.jsonl"
    progress_lock = threading.Lock()
    thread_local = threading.local()

    completed_by_asset = load_completed_rows(output_dir) if resume else {}
    if not resume:
        progress_path.write_text("", encoding="utf-8")

    logging.info(
        "Running prompt %02d (%s) on %s documents with workers=%s resume=%s",
        prompt.id,
        prompt.name,
        len(documents),
        workers,
        resume,
    )

    pending_documents = [
        doc for doc in documents
        if doc.asset_key not in completed_by_asset
    ]
    skipped_count = len(documents) - len(pending_documents)
    if skipped_count:
        logging.info("Skipping %s already completed documents for prompt %02d", skipped_count, prompt.id)

    def get_thread_client() -> AzureOpenAI:
        if workers == 1 and client is not None:
            return client
        local_client = getattr(thread_local, "client", None)
        if local_client is None:
            local_client = get_azure_openai_client()
            thread_local.client = local_client
        return local_client

    def process_one(doc: DocumentRow) -> tuple[list[dict], float, bool]:
        doc_started_at = time.perf_counter()
        try:
            extraction = extract_document_with_openai(
                client=get_thread_client(),
                doc=doc,
                max_chars=max_chars,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=prompt,
            )
            rows = extraction_to_output_rows(doc, extraction)
            failed = False
        except Exception as exc:
            logging.exception("OpenAI extraction failed for asset_key=%s: %s", doc.asset_key, exc)
            rows = error_output_rows(doc, exc)
            failed = True
        return rows, time.perf_counter() - doc_started_at, failed

    new_completed = 0
    new_failures = 0
    total_pending = len(pending_documents)

    def record_completed(rows: list[dict], elapsed: float, failed: bool) -> None:
        nonlocal new_completed, new_failures
        asset_key = rows[0]["asset_key"] if rows else ""
        completed_by_asset[asset_key] = rows
        append_progress_rows(rows, progress_path, progress_lock)
        new_completed += 1
        if failed:
            new_failures += 1
        logging.info(
            "Completed prompt %02d document %s/%s in %.1fs | asset_key=%s | %s",
            prompt.id,
            new_completed,
            total_pending,
            elapsed,
            asset_key,
            "error" if failed else "ok",
        )

    if total_pending and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_doc = {
                executor.submit(process_one, doc): doc
                for doc in pending_documents
            }
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    rows, elapsed, failed = future.result()
                except Exception as exc:
                    logging.exception("Worker failed for asset_key=%s: %s", doc.asset_key, exc)
                    rows = error_output_rows(doc, exc)
                    elapsed = 0.0
                    failed = True
                record_completed(rows, elapsed, failed)
    else:
        for doc in pending_documents:
            rows, elapsed, failed = process_one(doc)
            record_completed(rows, elapsed, failed)

    output_rows = []
    for doc in documents:
        output_rows.extend(completed_by_asset.get(doc.asset_key, []))

    save_jsonl(output_rows, output_dir / "event_extractions.jsonl")
    save_csv(output_rows, output_dir / "event_extractions.csv")

    elapsed = time.perf_counter() - started_at
    docs_per_min = (new_completed / elapsed * 60.0) if elapsed > 0 else 0.0
    failure_count = sum(1 for row in output_rows if is_error_row(row))
    logging.info(
        "Finished prompt %02d (%s) in %.1fs | processed=%s | skipped=%s | failures=%s | %.2f docs/min",
        prompt.id,
        prompt.name,
        elapsed,
        new_completed,
        skipped_count,
        failure_count,
        docs_per_min,
    )
    logging.info("Wrote:")
    logging.info("  - %s", output_dir / "event_extractions_progress.jsonl")
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
        "--workers",
        type=int,
        default=1,
        help="Number of local OpenAI extraction workers. Use 1 for serial behavior.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing event_extractions_progress.jsonl or event_extractions.jsonl.",
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

    if args.workers < 1:
        parser.error("--workers must be >= 1")

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

    documents = fetch_documents(limit=args.limit)
    client = get_azure_openai_client() if args.workers == 1 else None

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
                workers=args.workers,
                resume=args.resume,
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
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
