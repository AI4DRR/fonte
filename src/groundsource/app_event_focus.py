import argparse
import csv
import json
import logging
import math
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
    "event_confidence_logprob",
    "event_confidence_prob",
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


def format_duration(seconds: float) -> str:
    """Render an elapsed duration as e.g. '1h 02m 03.4s' or '7.8s'."""
    total = max(seconds, 0.0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def connect_databricks():
    ca_file = os.getenv("DATABRICKS_CA_FILE", certifi.where())
    if not os.path.isfile(ca_file):
        raise ValueError(
            f"DATABRICKS_CA_FILE={ca_file!r} is not a file. If you're running "
            "via Docker Compose, this usually means the host-side "
            "./databricks_chain.pem bind-mount source didn't exist when the "
            "container started, so Docker created an empty directory there "
            "instead. Either place the real CA chain at that path, or unset "
            "DATABRICKS_CA_FILE (and its volume mount) to fall back to "
            "certifi's default trust store."
        )
    return sql.connect(
        server_hostname=require_env("DATABRICKS_SERVER_HOSTNAME"),
        http_path=require_env("DATABRICKS_HTTP_PATH"),
        access_token=require_env("DATABRICKS_TOKEN"),
        _tls_trusted_ca_file=ca_file,
    )

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def get_azure_openai_client() -> AzureOpenAI:
    # The SDK automatically retries 429 / 5xx with exponential backoff and
    # honors the server's Retry-After header. The default (2) is too low for
    # sustained concurrent runs, so bump it and give requests a generous
    # timeout. Both are env-overridable.
    return AzureOpenAI(
        azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=require_env("AZURE_OPENAI_API_KEY"),
        azure_deployment=require_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=require_env("API_VERSION"),
        max_retries=_env_int("OPENAI_MAX_RETRIES", 8),
        timeout=_env_float("OPENAI_TIMEOUT", 60.0),
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


# Locate each event_confidence value in the raw JSON. Group 1 is the value
# only (not the surrounding quotes), so its char span maps to the value tokens.
_EVENT_CONFIDENCE_VALUE_RE = re.compile(r'"event_confidence"\s*:\s*"(high|medium|low)"')


def _event_identity_key(hazard, dates, locations) -> tuple:
    """Mirror DocumentEventExtractions._keep_verified_unique_events so we can
    match a post-validation event back to its object in the raw model JSON."""
    return (
        (hazard or "").strip().lower(),
        tuple(s.strip().lower() for s in (dates or []) if s and s.strip()),
        tuple(s.strip().lower() for s in (locations or []) if s and s.strip()),
    )


def event_confidence_logprobs(completion, extraction: DocumentEventExtractions) -> list[Optional[dict]]:
    """Token-level probability of each surviving event's event_confidence value.

    Returns a list aligned to ``extraction.events``; each entry is
    ``{"value": str, "logprob": float, "prob": float}`` or ``None`` when the
    value could not be located (e.g. logprobs unavailable on a reasoning model,
    or the raw JSON could not be matched).

    The model's validator drops low-confidence / duplicate / under-specified
    events, so surviving events do NOT line up positionally with the raw token
    stream. We therefore reconstruct the response text from the token stream
    (recording each token's char span), pair every raw event with the logprob
    of its own confidence value, then match each surviving event to a raw event
    by identity key.
    """
    n = len(extraction.events)
    try:
        choice = completion.choices[0]
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if not content:
            return [None] * n

        # Reconstruct text with per-token (start, end, logprob) char spans.
        spans: list[tuple[int, int, float]] = []
        parts: list[str] = []
        cursor = 0
        for tok in content:
            s = tok.token
            spans.append((cursor, cursor + len(s), tok.logprob))
            parts.append(s)
            cursor += len(s)
        full = "".join(parts)

        # Logprob for each confidence value occurrence, in raw JSON order.
        value_lps: list[dict] = []
        for m in _EVENT_CONFIDENCE_VALUE_RE.finditer(full):
            v_start, v_end = m.span(1)
            total = 0.0
            found = False
            for (s, e, token_lp) in spans:
                if e <= v_start or s >= v_end:
                    continue
                total += token_lp
                found = True
            if found:
                value_lps.append({"value": m.group(1), "logprob": total, "prob": math.exp(total)})

        raw_events = (json.loads(full) or {}).get("events", []) or []

        # Pair each raw event that carries a matchable confidence value with the
        # next value logprob (same left-to-right order as the regex scan).
        raw_pairs: list[tuple[tuple, dict]] = []  # (identity_key, logprob_dict)
        vi = 0
        for ev in raw_events:
            conf = ev.get("event_confidence")
            if conf not in ("high", "medium", "low"):
                continue  # null/other → no regex match was produced for it
            if vi >= len(value_lps):
                break
            key = _event_identity_key(
                ev.get("event_hazard"), ev.get("event_dates"), ev.get("affected_locations")
            )
            raw_pairs.append((key, value_lps[vi]))
            vi += 1

        # Attach to each surviving event by identity, consuming matches once.
        results: list[Optional[dict]] = []
        used = [False] * len(raw_pairs)
        for event in extraction.events:
            key = _event_identity_key(
                event.event_hazard, event.event_dates, event.affected_locations
            )
            match = None
            for i, (raw_key, lp_dict) in enumerate(raw_pairs):
                if not used[i] and raw_key == key:
                    used[i] = True
                    match = lp_dict
                    break
            results.append(match)
        return results
    except Exception:  # never let confidence scoring break extraction
        logging.warning("event_confidence logprob mapping failed", exc_info=True)
        return [None] * n


def extract_document_with_openai(
    client: AzureOpenAI,
    doc: DocumentRow,
    max_chars: int,
    model: str,
    reasoning_effort: Optional[str],
    prompt: PromptSpec,
) -> tuple[DocumentEventExtractions, list[Optional[dict]]]:
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
        # Reasoning models do not return token logprobs; request them only for
        # standard models so confidence scoring degrades gracefully to None.
        completion_kwargs["reasoning_effort"] = reasoning_effort
    else:
        completion_kwargs["logprobs"] = True

    completion = client.beta.chat.completions.parse(**completion_kwargs)
    extraction = completion.choices[0].message.parsed
    conf_logprobs = event_confidence_logprobs(completion, extraction)
    return extraction, conf_logprobs


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


def extraction_to_output_rows(
    doc: DocumentRow,
    extraction: DocumentEventExtractions,
    conf_logprobs: Optional[list[Optional[dict]]] = None,
) -> list[dict]:
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
            "event_confidence_logprob": None,
            "event_confidence_prob": None,
            "evidence_snippets": "",
            "missing_information": "no verified reliable event extracted",
        }]

    if not conf_logprobs or len(conf_logprobs) != event_count:
        conf_logprobs = [None] * event_count

    rows: list[dict] = []
    for idx, (event, conf_lp) in enumerate(zip(extraction.events, conf_logprobs), start=1):
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
            "event_confidence_logprob": (conf_lp or {}).get("logprob"),
            "event_confidence_prob": (conf_lp or {}).get("prob"),
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
        "event_confidence_logprob": None,
        "event_confidence_prob": None,
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

    # An asset counts as "done" only if it has rows and none of them are error
    # rows. Assets whose previous attempt errored (e.g. a 429) are re-queued so
    # resume actually retries them instead of treating them as complete.
    def asset_succeeded(asset_key: str) -> bool:
        rows = completed_by_asset.get(asset_key)
        return bool(rows) and not any(is_error_row(row) for row in rows)

    pending_documents = [
        doc for doc in documents
        if not asset_succeeded(doc.asset_key)
    ]
    retry_count = sum(
        1 for doc in pending_documents if doc.asset_key in completed_by_asset
    )
    skipped_count = len(documents) - len(pending_documents)
    if skipped_count:
        logging.info("Skipping %s already completed documents for prompt %02d", skipped_count, prompt.id)
    if retry_count:
        logging.info("Retrying %s previously failed documents for prompt %02d", retry_count, prompt.id)

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
            extraction, conf_logprobs = extract_document_with_openai(
                client=get_thread_client(),
                doc=doc,
                max_chars=max_chars,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=prompt,
            )
            rows = extraction_to_output_rows(doc, extraction, conf_logprobs)
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

    # Compact the append-only progress log to the final deduped state so the
    # next resume sees one clean record per asset (retried successes no longer
    # carry their stale error rows from earlier attempts).
    save_jsonl(output_rows, progress_path)

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
        default=None,
        help=(
            "Folder where JSONL and CSV outputs will be saved. "
            "Defaults to $EXTRACT_OUTPUT_DIR when not given."
        ),
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

    run_started_at = time.perf_counter()
    status = "ok"
    try:
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

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(require_env("EXTRACT_OUTPUT_DIR"))

        prompt = get_prompt(args.prompt)
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
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "error"
        raise
    finally:
        total_elapsed = time.perf_counter() - run_started_at
        logging.info(
            "Total run time: %s (%.1fs) | status=%s",
            format_duration(total_elapsed),
            total_elapsed,
            status,
        )


if __name__ == "__main__":
    main()
