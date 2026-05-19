"""
eval/gold_template.py — Stage 3 of the funnel.

Generates `eval/gold.csv`: an empty labelling template, one row per
sampled document from `eval/review_sample.csv`. The user fills in the
label columns; everything else is reference context. This is the
half-day labelling pass.

Label columns (the user fills these):
    is_real_event       T / F / unclear
    true_hazard         free text (or controlled vocab from UNDRR taxonomy)
    true_country        country name as written in the document
    true_locations      semicolon-separated set, sub-national specificity ok
    true_date_start     YYYY-MM-DD, blank if undetermined
    true_date_end       YYYY-MM-DD, blank if undetermined
    notes               free text — surprises, ambiguity, edge cases

Reference columns (read-only, do not edit):
    asset_key, sample_bucket, title, content_url, language, hazard_family,
    runs_hazard_summary — one-line digest of what each run guessed
                          (helps anchor the labeller without forcing them
                          to flip back to the review CSV)

We refuse to overwrite an existing `eval/gold.csv` — that would clobber
work in progress. Pass `--force` to override (e.g., when re-sampling).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = REPO_ROOT / "eval" / "review_sample.csv"
DEFAULT_GOLD = REPO_ROOT / "eval" / "gold.csv"

LABEL_COLUMNS = (
    "is_real_event",
    "true_hazard",
    "true_country",
    "true_locations",
    "true_date_start",
    "true_date_end",
    "notes",
)


def _hazard_summary(row: pd.Series) -> str:
    """Compact one-liner of what each run guessed for event_hazard."""
    bits: list[str] = []
    for col in row.index:
        if not col.endswith("__event_hazard"):
            continue
        rid = col.split("__", 1)[0]
        val = str(row[col] or "").strip()
        if val and val.lower() != "none":
            bits.append(f"{rid}={val}")
    return "; ".join(bits) if bits else "(all empty)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing gold.csv (DANGER: clobbers labels)")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} already exists. Pass --force to overwrite "
            f"(will clobber any labels in progress)."
        )

    sample = pd.read_csv(args.sample)
    summary = sample.apply(_hazard_summary, axis=1)

    ref_cols = [
        c for c in ("asset_key", "sample_bucket", "hazard_family", "title",
                    "content_url", "language")
        if c in sample.columns
    ]
    gold = sample[ref_cols].copy()
    gold["runs_hazard_summary"] = summary
    for c in LABEL_COLUMNS:
        gold[c] = ""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(gold)} rows ready for labelling)")
    print()
    print("Open in a spreadsheet and fill in the LABEL columns:")
    for c in LABEL_COLUMNS:
        print(f"  - {c}")
    print()
    print("Reference columns (do not edit):", ", ".join(ref_cols + ["runs_hazard_summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
