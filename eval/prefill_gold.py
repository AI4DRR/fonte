"""
eval/prefill_gold.py — AI/auto pre-fill for Stage 3 (gold labelling).

Turns the 75-doc hand-labelling pass into a *verification* pass: for each
gold row we compute a consensus answer across the 8 runs in
`unified.parquet` and write it into the blank gold columns. The human then
only has to confirm the easy rows and actually read the documents the runs
disagree on.

Adds two helper columns (ignored by score_against_gold.py):
  auto_agreement   0..1, how strongly the 8 runs agreed on is_verifiable_event
  review_priority  HIGH / MED / LOW  — what to read first

Consensus rules
  is_real_event   majority vote of is_verifiable_event (T/F)
  true_hazard     most common event_hazard among runs that called it verifiable
  true_country    document-level `countries` field (already model-pooled)
  true_locations  locations named by >= half of the verifiable runs (pipe-joined)
  true_date_start/end  min start / max end across runs' event_dates

Never clobbers a row a human already filled (keyed on is_real_event being set).
Backs up the existing gold.csv before writing.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "eval" / "gold.csv"
UNIFIED = REPO / "eval" / "unified.parquet"

RUN_IDS = [
    f"{eff}_prompt{n:02d}" for eff in ("high", "medium") for n in range(1, 5)
]
GOLD_COLS = [
    "asset_key", "sample_bucket", "hazard_family", "title", "content_url",
    "language", "runs_hazard_summary", "is_real_event", "true_hazard",
    "true_country", "true_locations", "true_date_start", "true_date_end", "notes",
]
HELPER_COLS = ["auto_agreement", "review_priority"]


def _bool(v) -> bool | None:
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _locs(v) -> list[str]:
    return [p.strip() for p in re.split(r"[|;]", str(v or "")) if p.strip()]


def _dates(v) -> list[str]:
    out = []
    for p in str(v or "").split("|"):
        p = p.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p):
            out.append(p)
    return out


def consensus(row: pd.Series) -> dict:
    verif = [_bool(row.get(f"{r}__is_verifiable_event")) for r in RUN_IDS]
    n_true = sum(1 for v in verif if v is True)
    n_known = sum(1 for v in verif if v is not None)
    is_real = "T" if n_true * 2 >= max(n_known, 1) and n_true > 0 else "F"
    agreement = (max(n_true, n_known - n_true) / n_known) if n_known else 0.0

    verif_runs = [r for r, v in zip(RUN_IDS, verif) if v is True]

    haz = Counter(
        str(row.get(f"{r}__event_hazard")).strip()
        for r in verif_runs
        if str(row.get(f"{r}__event_hazard")).strip() not in ("", "None")
    )
    true_hazard = haz.most_common(1)[0][0] if haz else ""

    loc_counter: Counter = Counter()
    for r in verif_runs:
        for l in {x.lower() for x in _locs(row.get(f"{r}__affected_locations"))}:
            loc_counter[l] += 1
    half = max(len(verif_runs), 1) / 2
    true_locations = " | ".join(sorted(l for l, c in loc_counter.items() if c >= half))

    starts, ends = [], []
    for r in verif_runs:
        ds = _dates(row.get(f"{r}__event_dates"))
        if ds:
            starts.append(min(ds))
            ends.append(max(ds))
    true_start = min(starts) if starts else ""
    true_end = max(ends) if ends else ""

    country = "" if str(row.get("countries")) in ("None", "nan", "") else str(row.get("countries"))

    if is_real == "F" and agreement >= 0.875:
        priority = "LOW"   # near-unanimous non-event: quick skim
    elif agreement >= 0.875 and true_hazard and true_locations:
        priority = "LOW"   # runs agree on a clean event
    elif agreement >= 0.625:
        priority = "MED"
    else:
        priority = "HIGH"  # runs genuinely split — must read the doc

    return {
        "is_real_event": is_real,
        "true_hazard": true_hazard if is_real == "T" else "",
        "true_country": country if is_real == "T" else "",
        "true_locations": true_locations if is_real == "T" else "",
        "true_date_start": true_start if is_real == "T" else "",
        "true_date_end": true_end if is_real == "T" else "",
        "auto_agreement": round(agreement, 3),
        "review_priority": priority,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite human-filled rows too")
    args = ap.parse_args()

    u = pd.read_parquet(UNIFIED).set_index("asset_key")
    with GOLD.open() as f:
        gold = list(csv.DictReader(f))

    shutil.copy(GOLD, GOLD.with_suffix(f".csv.bak.{datetime.now():%Y%m%d-%H%M%S}"))

    filled = skipped = 0
    for row in gold:
        if row.get("is_real_event", "").strip() and not args.force:
            skipped += 1
            row.setdefault("auto_agreement", "")
            row.setdefault("review_priority", "DONE")
            continue
        key = row["asset_key"]
        if key not in u.index:
            continue
        row.update(consensus(u.loc[key]))
        filled += 1

    with GOLD.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GOLD_COLS + HELPER_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(gold)

    prio = Counter(r.get("review_priority", "") for r in gold)
    print(f"pre-filled {filled} rows, kept {skipped} human-labelled")
    print("review_priority:", dict(prio))
    print(f"-> HIGH rows need real reading; LOW rows just need a spot-check.")


if __name__ == "__main__":
    main()
