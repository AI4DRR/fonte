"""
eval/prefill_gold_validation.py — build the gold-run validation queue.

Flattens a prompt-5 extraction JSONL (default: the `gold` production run) to
one row per verifiable event, scores a label-free risk heuristic, and queues a
stratified, risk-ranked sample (~100 events) for a single correct/partial/wrong
verdict per event. Everything is automated except the verdict.

Adds:
  stratum      hazard bucket used for coverage (the event_hazard)
  auto_risk    0..1 heuristic risk that the extraction is wrong
  review_rank  order to work through (risk-first, balanced across strata)
  in_queue     Y for the capped sample to actually judge, N otherwise
  verdict      left blank for the human (correct | partial | wrong)
  notes        left blank for the human

Risk heuristics (no labels, from the extraction itself):
  + medium/low event_confidence            (model itself is less sure)
  + multi-event document                   (disaggregation is harder)
  + vague date (year-only, "since", range)  (temporal precision risk)
  + country/region admin level             (location specificity risk)
  + thin evidence / flagged missing_information

Writes eval/gold_validation.csv (backed up first if it already exists).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "outputs" / "gold" / "event_extractions.jsonl"
DEFAULT_OUT = REPO_ROOT / "eval" / "gold_validation.csv"

REFERENCE_FIELDS = (
    "asset_key",
    "event_index",
    "event_count",
    "title",
    "content_url",
    "language",
    "event_hazard",
    "event_dates",
    "affected_locations",
    "location_admin_levels",
    "key_impacts",
    "event_confidence",
    "evidence_snippets",
    "missing_information",
)
COMPUTED_FIELDS = ("stratum", "auto_risk", "review_rank", "in_queue")
JUDGMENT_FIELDS = ("verdict", "notes")
OUTPUT_FIELDS = (*REFERENCE_FIELDS, *COMPUTED_FIELDS, *JUDGMENT_FIELDS)

_VAGUE_DATE_HINTS = ("since", "early", "late", "mid", "around", "before", "after", "?")


def _is_verifiable(obj: dict) -> bool:
    if not str(obj.get("event_index", "")).strip():
        return False
    v = obj.get("is_verifiable_event")
    return v is True or str(v).strip().lower() == "true"


def _count_parts(value: str) -> int:
    return len([p for p in str(value or "").split("|") if p.strip()])


def _date_is_vague(value: str) -> bool:
    v = str(value or "").strip().lower()
    if not v:
        return True
    if any(h in v for h in _VAGUE_DATE_HINTS):
        return True
    # A precise date is YYYY-MM-DD; anything shorter (year, year-month) is vague.
    first = v.split("|")[0].strip()
    return len(first) < 10


def risk(obj: dict) -> float:
    r = 0.0
    conf = str(obj.get("event_confidence", "")).strip().lower()
    if conf == "medium":
        r += 0.40
    elif conf == "low":
        r += 0.60
    elif conf == "high":
        r += 0.10
    else:
        r += 0.30  # missing/unknown confidence

    try:
        if int(obj.get("event_count", 1)) > 1:
            r += 0.15
    except (TypeError, ValueError):
        pass

    if _date_is_vague(obj.get("event_dates", "")):
        r += 0.20

    admin = str(obj.get("location_admin_levels", "")).strip().lower()
    if any(a in admin for a in ("country", "national", "region", "state")):
        r += 0.20

    if _count_parts(obj.get("evidence_snippets", "")) <= 1:
        r += 0.10
    if _count_parts(obj.get("missing_information", "")) >= 1:
        r += 0.10

    return round(min(r, 1.0), 3)


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_verifiable(obj):
                events.append(obj)
    return events


def queue_round_robin(events: list[dict], target: int) -> None:
    """Mark ~target events in_queue=Y, risk-first but balanced across strata.

    Picks the highest-risk remaining event from each stratum in turn until the
    target is reached, so no single hazard dominates the sample.
    """
    by_stratum: dict[str, list[dict]] = {}
    for e in events:
        by_stratum.setdefault(e["stratum"], []).append(e)
    for group in by_stratum.values():
        group.sort(key=lambda x: x["auto_risk"], reverse=True)

    order = sorted(by_stratum.keys())
    cursors = {s: 0 for s in order}
    picked: list[dict] = []
    target = min(target, len(events))
    while len(picked) < target:
        progressed = False
        for s in order:
            if len(picked) >= target:
                break
            grp = by_stratum[s]
            i = cursors[s]
            if i < len(grp):
                picked.append(grp[i])
                cursors[s] = i + 1
                progressed = True
        if not progressed:
            break

    for e in events:
        e["in_queue"] = "N"
        e["review_rank"] = ""
    for rank, e in enumerate(picked, start=1):
        e["in_queue"] = "Y"
        e["review_rank"] = rank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="Prompt-5 extraction JSONL (default: the gold run).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Validation CSV to write.")
    ap.add_argument("--target", type=int, default=100,
                    help="Approximate number of events to queue for judging.")
    args = ap.parse_args()

    if not args.source.exists():
        ap.error(
            f"Source extraction not found: {args.source}\n"
            "Run the gold extraction first, e.g.:\n"
            "  groundsource-extract --prompt 5 --limit 2000 --workers 12 "
            "--reasoning-effort medium --output-dir outputs/gold --resume"
        )

    events = load_events(args.source)
    if not events:
        ap.error(f"No verifiable events found in {args.source}.")

    for e in events:
        e["stratum"] = (e.get("event_hazard") or "(unknown)").strip() or "(unknown)"
        e["auto_risk"] = risk(e)
        e["verdict"] = ""
        e["notes"] = ""

    queue_round_robin(events, args.target)

    # Output rows ordered: queued first by review_rank, then the rest by risk.
    queued = [e for e in events if e["in_queue"] == "Y"]
    rest = [e for e in events if e["in_queue"] != "Y"]
    queued.sort(key=lambda x: x["review_rank"])
    rest.sort(key=lambda x: x["auto_risk"], reverse=True)
    ordered = queued + rest

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        backup = args.out.with_suffix(f".csv.bak.{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(args.out, backup)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(OUTPUT_FIELDS), extrasaction="ignore")
        w.writeheader()
        for e in ordered:
            w.writerow({c: e.get(c, "") for c in OUTPUT_FIELDS})

    n_strata = len({e["stratum"] for e in queued})
    print(f"{len(events)} verifiable events; queued {len(queued)} across {n_strata} hazard strata.")
    counts: dict[str, int] = {}
    for e in queued:
        counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
    for s, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {s}: {n}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
