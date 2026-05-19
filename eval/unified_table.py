"""
eval/unified_table.py — Stage 2.1 of the funnel.

Joins all 8 extraction runs (test_400_gpt5_{high,medium}_prompt{01..04})
by asset_key into a wide parquet keyed on document, with one column-block
per run holding the structured outputs we'll compare across prompts.

Also computes a `disagreement_score` per document so Stage 2.2 can
sample the cases where reviewer attention has highest ROI — the
docs where the 8 runs emit visibly different events.

Schema of the output parquet (`eval/unified.parquet`):

    asset_key, title, content_url, language, doc_countries, doc_hazards,
    <run_id>__is_verifiable, <run_id>__event_hazard,
    <run_id>__affected_locations, <run_id>__event_dates,
    <run_id>__event_confidence, <run_id>__key_impacts
    ... (one block per run_id, 8 of them) ...
    disagreement_score

`disagreement_score` formula is documented in `_disagreement_for_doc`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_UNIFIED = REPO_ROOT / "eval" / "unified.parquet"

RUN_DIR_RE = re.compile(r"^test_400_gpt5_(?P<effort>high|medium)_prompt(?P<prompt>0[1-4])$")

# Per-run fields lifted into the wide table. Keep this small — the
# Stage 4 scorer reads only what's here, and the wide table needs to
# stay easy to scan in a spreadsheet.
PER_RUN_FIELDS = (
    "is_verifiable_event",
    "event_hazard",
    "affected_locations",
    "event_dates",
    "event_confidence",
    "key_impacts",
)

# Doc-level fields (same across runs in principle — we keep them once).
DOC_FIELDS = (
    "title",
    "content_url",
    "language",
    "countries",
    "hazards",
)


def _truthy(s) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() == "true"


def _split_pipes(s) -> list[str]:
    if not s or str(s).strip().lower() in {"", "none"}:
        return []
    return [p.strip() for p in str(s).split("|") if p.strip()]


def _norm_hazard(s) -> str:
    return str(s or "").strip().lower()


def _load_run(run_dir: Path) -> tuple[str, pd.DataFrame]:
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        raise ValueError(f"unexpected run dir name: {run_dir.name}")
    run_id = f"{m.group('effort')}_prompt{m.group('prompt')}"
    rows: list[dict] = []
    with (run_dir / "event_extractions.jsonl").open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            rows.append({
                "asset_key": r["asset_key"],
                **{k: r.get(k) for k in DOC_FIELDS},
                **{f"{run_id}__{k}": r.get(k) for k in PER_RUN_FIELDS},
            })
    return run_id, pd.DataFrame(rows)


def _disagreement_for_doc(per_run: dict[str, dict]) -> float:
    """
    Score how much the 8 runs disagree on a single document.

    Components, summed:
      * 1.0 × number of distinct (boolean) `is_verifiable_event` values
        across runs that actually have output (0 or 1 in {0,1,2}).
      * 1.0 × number of distinct non-empty `event_hazard` values
        (lowercased). Hazard disagreement is the strongest signal that
        prompts are interpreting the document differently.
      * 0.5 × Jaccard distance of pooled `affected_locations` sets.
        Computed as 1 - |intersection| / |union| where the intersection
        is across runs that emitted any locations. Runs that emit no
        locations don't drag the score either way.

    The formula is deliberately rough — it's a ranking signal, not a
    calibrated metric. The aim is to surface the most argumentative
    documents first; the precise weights matter less than the order.
    """
    score = 0.0

    verif_values = {bool(_truthy(v.get("is_verifiable_event"))) for v in per_run.values()}
    score += 1.0 * len(verif_values)

    hazards = {_norm_hazard(v.get("event_hazard")) for v in per_run.values()}
    hazards.discard("")
    hazards.discard("none")
    score += 1.0 * len(hazards)

    # Jaccard distance over pooled location sets.
    loc_sets = [set(_split_pipes(v.get("affected_locations"))) for v in per_run.values()]
    loc_sets = [s for s in loc_sets if s]
    if len(loc_sets) >= 2:
        union = set().union(*loc_sets)
        inter = set(loc_sets[0])
        for s in loc_sets[1:]:
            inter &= s
        jacc = (len(inter) / len(union)) if union else 1.0
        score += 0.5 * (1.0 - jacc)

    return float(score)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_UNIFIED)
    args = ap.parse_args()

    run_dirs = sorted(
        d for d in args.outputs_dir.iterdir()
        if d.is_dir() and RUN_DIR_RE.match(d.name)
    )
    if not run_dirs:
        raise SystemExit(f"No matching run dirs under {args.outputs_dir}")

    # Load each run; merge progressively on asset_key. Doc-level fields are
    # carried from the first run only (they should match, but we don't
    # bother validating — if they don't, the spreadsheet review will catch).
    merged: pd.DataFrame | None = None
    run_ids: list[str] = []
    for d in run_dirs:
        run_id, df = _load_run(d)
        run_ids.append(run_id)
        if merged is None:
            merged = df
        else:
            df_run_only = df.drop(columns=list(DOC_FIELDS), errors="ignore")
            merged = merged.merge(df_run_only, on="asset_key", how="outer")

    assert merged is not None

    # Compute disagreement per row.
    def _row_disagreement(row: pd.Series) -> float:
        per_run = {
            rid: {f: row.get(f"{rid}__{f}") for f in PER_RUN_FIELDS}
            for rid in run_ids
        }
        return _disagreement_for_doc(per_run)

    merged["disagreement_score"] = merged.apply(_row_disagreement, axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({len(merged)} docs × {len(run_ids)} runs)")
    print(f"disagreement_score: min={merged['disagreement_score'].min():.2f} "
          f"median={merged['disagreement_score'].median():.2f} "
          f"max={merged['disagreement_score'].max():.2f}")
    print(f"docs with score >= 2: {(merged['disagreement_score'] >= 2).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
