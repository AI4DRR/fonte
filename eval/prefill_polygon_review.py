"""
eval/prefill_polygon_review.py — auto-prioritise the polygon validation pass.

The generated sample (565 rows for test_2000_prompt05) is too large to hand-judge.
This caps it to a defensible stratified review queue and auto-flags the rows most
likely to be wrong, so the human judges the risky ones first and can stop early
with a statistically meaningful sample.

Adds:
  auto_risk    0..1 heuristic risk that the resolution is wrong
  review_rank  order to work through (highest risk first, within stratum)
  in_queue     Y for the capped sample to actually judge, N otherwise

Heuristics (label-free, from resolver metadata):
  + used a country-level fallback        (coarse / overshoot risk)
  + admin_level == 'country' but the text names a sub-national place
  + low resolution_confidence
  + huge area_km2 for a place that reads specific
Does not fill location_correct/geometry_correct/date_correct — those stay blank
for the human. Backs up the file first.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

DEFAULT = (
    "outputs/test_2000_gpt5_medium_prompt05_polygon_validation/"
    "polygon_validation_sample.csv"
)


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def risk(row: dict) -> float:
    r = 0.0
    if str(row.get("resolution_used_fallback", "")).strip().lower() in {"true", "1", "yes"}:
        r += 0.4
    admin = str(row.get("resolution_admin_level", "")).strip().lower()
    text = str(row.get("location_text", "")).strip()
    if admin == "country" and len(text.split()) > 1:
        r += 0.25
    conf = _f(row.get("resolution_confidence"), 1.0)
    if conf < 0.4:
        r += 0.25
    elif conf < 0.7:
        r += 0.1
    if _f(row.get("area_km2")) > 500_000 and admin in {"", "region", "state"}:
        r += 0.1
    return min(r, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT)
    ap.add_argument("--per-stratum", type=int, default=20,
                    help="rows to queue per stratum (high/low/unresolved)")
    args = ap.parse_args()

    path = Path(__file__).resolve().parent.parent / args.file
    rows = list(csv.DictReader(path.open()))
    shutil.copy(path, path.with_suffix(f".csv.bak.{datetime.now():%Y%m%d-%H%M%S}"))

    for r in rows:
        r["auto_risk"] = round(risk(r), 3)

    # Rank within stratum by risk desc; queue the top `per-stratum` of each.
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r.get("stratum", "?"), []).append(r)
    for stratum, group in by_stratum.items():
        group.sort(key=lambda x: x["auto_risk"], reverse=True)
        for i, r in enumerate(group, 1):
            r["review_rank"] = i
            r["in_queue"] = "Y" if i <= args.per_stratum else "N"

    fields = list(rows[0].keys())
    for c in ("auto_risk", "review_rank", "in_queue"):
        if c not in fields:
            fields.append(c)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    queued = sum(1 for r in rows if r.get("in_queue") == "Y")
    print(f"{len(rows)} rows scored; queued {queued} for human judgement "
          f"({args.per_stratum}/stratum).")
    for s, g in by_stratum.items():
        q = sum(1 for r in g if r["in_queue"] == "Y")
        print(f"  {s}: {q} queued of {len(g)}")


if __name__ == "__main__":
    main()
