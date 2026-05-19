"""
eval/sample_for_review.py — Stage 2.2 of the funnel.

Reads `eval/unified.parquet` (built by `unified_table.py`) and selects 75
documents for hand review. Splits:

  * 50 docs: highest `disagreement_score` — most actionable, since this
    is where the 8 runs visibly disagree and human input changes the
    ranking.
  * 15 docs: stratified by hazard family across the long tail (flood,
    earthquake, cyclone, drought, wildfire, landslide, volcano, other)
    so per-segment scoring in Stage 4 has at least a few examples per
    family.
  * 10 docs: random low-disagreement — sanity check that when the runs
    agree they're not just consistently wrong.

The output `eval/review_sample.csv` is wide and meant to be opened in a
spreadsheet: one row per doc, with the 8 runs' fields laid out
side-by-side for inline reading. It is the input to the gold-labelling
pass in Stage 3.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIFIED = REPO_ROOT / "eval" / "unified.parquet"
DEFAULT_SAMPLE = REPO_ROOT / "eval" / "review_sample.csv"

# Hazard family taxonomy. Order matters for the regex search — we take
# the first match. The "other" family is implicit (anything that matches
# none of the patterns).
HAZARD_FAMILIES: tuple[tuple[str, re.Pattern], ...] = (
    ("flood", re.compile(r"flood|inundation|deluge", re.I)),
    ("earthquake", re.compile(r"earthquake|seismic|quake|tsunami", re.I)),
    ("cyclone", re.compile(r"cyclone|hurricane|typhoon|storm", re.I)),
    ("drought", re.compile(r"drought|famine|water scarcity", re.I)),
    ("wildfire", re.compile(r"wildfire|bushfire|forest fire|fire", re.I)),
    ("landslide", re.compile(r"landslide|mudslide|rockfall", re.I)),
    ("volcano", re.compile(r"volcan|eruption|lahar", re.I)),
    ("heat", re.compile(r"heat ?wave|extreme heat", re.I)),
)


def _pooled_hazards(row: pd.Series, run_ids: list[str]) -> str:
    """Concatenate non-empty hazard strings across runs for family detection."""
    parts: list[str] = []
    for rid in run_ids:
        h = row.get(f"{rid}__event_hazard")
        if h and str(h).strip().lower() not in {"", "none"}:
            parts.append(str(h).strip())
    return " | ".join(parts)


def _hazard_family(pooled: str) -> str:
    for fam, pat in HAZARD_FAMILIES:
        if pat.search(pooled):
            return fam
    return "other"


def _run_ids_from_columns(cols: list[str]) -> list[str]:
    rids = sorted({c.split("__", 1)[0] for c in cols if "__" in c})
    return rids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=DEFAULT_UNIFIED)
    ap.add_argument("--out", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--n-disagree", type=int, default=50)
    ap.add_argument("--n-stratified", type=int, default=15)
    ap.add_argument("--n-low", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.unified)
    run_ids = _run_ids_from_columns([c for c in df.columns if "__" in c])
    if not run_ids:
        raise SystemExit("No run columns found in unified parquet")

    df = df.copy()
    df["pooled_hazards"] = df.apply(lambda r: _pooled_hazards(r, run_ids), axis=1)
    df["hazard_family"] = df["pooled_hazards"].map(_hazard_family)

    # We only want to review docs where at least one run thinks something
    # event-like happened — totally-empty docs (all 8 runs verifiable=False
    # AND zero locations) carry no information for the comparison. We keep
    # them eligible for the random low-disagreement bucket so the sanity
    # check covers the "everyone agreed it's not an event" case too.
    has_any_event = pd.Series([False] * len(df), index=df.index)
    for rid in run_ids:
        col = f"{rid}__is_verifiable_event"
        if col in df.columns:
            has_any_event |= df[col].astype(str).str.strip().str.lower().eq("true")

    rng = pd.Series(range(len(df)), index=df.index).sample(frac=1.0, random_state=args.seed)

    # Bucket 1: top disagreement. Restrict to docs that have at least one
    # verifiable event somewhere, otherwise "high disagreement" is mostly
    # noise about empty rows.
    disagree_pool = df[has_any_event].sort_values(
        "disagreement_score", ascending=False
    )
    bucket_disagree = disagree_pool.head(args.n_disagree)

    # Bucket 2: stratified by hazard family across docs not already picked.
    remaining = df.drop(index=bucket_disagree.index)
    stratified_rows: list[pd.DataFrame] = []
    fam_target = max(1, args.n_stratified // (len(HAZARD_FAMILIES) + 1))  # +1 for "other"
    for fam, _ in HAZARD_FAMILIES:
        pool = remaining[(remaining["hazard_family"] == fam) & has_any_event.reindex(remaining.index, fill_value=False)]
        if pool.empty:
            continue
        take = pool.sample(min(fam_target, len(pool)), random_state=args.seed)
        stratified_rows.append(take)
        remaining = remaining.drop(index=take.index)
    bucket_stratified = pd.concat(stratified_rows) if stratified_rows else remaining.head(0)
    # If we under-shot the target (rare families thin), top up from any
    # remaining verifiable docs.
    if len(bucket_stratified) < args.n_stratified:
        topup_pool = remaining[has_any_event.reindex(remaining.index, fill_value=False)]
        if not topup_pool.empty:
            topup = topup_pool.sample(
                min(args.n_stratified - len(bucket_stratified), len(topup_pool)),
                random_state=args.seed,
            )
            bucket_stratified = pd.concat([bucket_stratified, topup])
            remaining = remaining.drop(index=topup.index)

    # Bucket 3: low-disagreement random sample. Eligible from any remaining
    # doc (including all-False ones), but biased toward score <= 1 so it's
    # actually "low" disagreement.
    low_pool = remaining[remaining["disagreement_score"] <= 1.0]
    if len(low_pool) < args.n_low:
        low_pool = remaining  # fall back if not enough
    bucket_low = low_pool.sample(min(args.n_low, len(low_pool)), random_state=args.seed)

    bucket_disagree = bucket_disagree.assign(sample_bucket="disagreement")
    bucket_stratified = bucket_stratified.assign(sample_bucket="stratified_by_hazard")
    bucket_low = bucket_low.assign(sample_bucket="low_disagreement_random")

    sample = pd.concat([bucket_disagree, bucket_stratified, bucket_low]).drop_duplicates(
        subset=["asset_key"]
    )

    # Order columns for spreadsheet readability: identifiers first, then
    # disagreement / family / bucket, then per-run blocks grouped by run.
    leading = [
        "asset_key", "sample_bucket", "disagreement_score", "hazard_family",
        "title", "language", "countries", "content_url",
    ]
    per_run_cols: list[str] = []
    for rid in run_ids:
        for f in ("is_verifiable_event", "event_hazard", "affected_locations",
                  "event_dates", "event_confidence"):
            col = f"{rid}__{f}"
            if col in sample.columns:
                per_run_cols.append(col)
    cols = [c for c in leading if c in sample.columns] + per_run_cols
    sample = sample[cols].sort_values(["sample_bucket", "disagreement_score"],
                                      ascending=[True, False])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(sample)} rows)")
    print(sample["sample_bucket"].value_counts().to_string())
    print()
    print("Hazard family distribution in sample:")
    print(sample["hazard_family"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
