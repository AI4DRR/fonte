# Prompt × reasoning-effort selection — runbook

Goal: pick the best prompt + reasoning-effort combination for the
`groundsource-extract` extraction step, using the 8 runs already produced
under `outputs/test_400_gpt5_{high,medium}_prompt{01..04}/`.

The strategy is a five-stage funnel. Stages 1–2 are label-free and run
end-to-end in a few minutes. Stage 3 is the half-day hand-labelling pass.
Stages 4–5 score against those labels and emit a recommendation.

## Stages

### 1. Triage metrics — `triage_metrics.py`

```
python3 eval/triage_metrics.py
```

Writes `eval/triage.csv`: one row per run with `verifiable_event_rate`,
`polygon_resolution_rate`, `country_hallucination_rate` (cross-checks the
LLM's `countries` field against Nominatim's resolved country),
`country_overshoot_rate`, plus a handful of emission-rate columns. Use
this to drop the bottom 2–3 runs from further review.

### 2.1. Unified comparison table — `unified_table.py`

```
python3 eval/unified_table.py
```

Joins all 8 runs by `asset_key` into `eval/unified.parquet`, with a
`disagreement_score` per document measuring how much the runs disagree
on `is_verifiable_event`, `event_hazard`, and pooled `affected_locations`.

### 2.2. Review sample — `sample_for_review.py`

```
python3 eval/sample_for_review.py
```

Writes `eval/review_sample.csv` (75 docs):
50 highest-disagreement + 15 stratified by hazard family + 10 random
low-disagreement. The CSV is wide and meant to be read in a spreadsheet,
with the 8 runs' outputs side-by-side.

### 3. Gold labelling — `gold_template.py`

```
python3 eval/gold_template.py
```

Writes `eval/gold.csv` (refuses to clobber an existing file; pass
`--force` if you really mean to overwrite). Fill in the label columns
(`is_real_event`, `true_hazard`, `true_country`, `true_locations`,
`true_date_start`, `true_date_end`, `notes`) by reading each document
via `content_url`. Use `runs_hazard_summary` as a cheat sheet — but don't
let it bias you.

**Verification before committing all 75:** re-label 5 docs blind and
check stability. If your second-pass labels disagree with the first on
more than 1 doc, write down a rubric before continuing.

### 4. Scoring against gold — `score_against_gold.py`

```
python3 eval/score_against_gold.py
```

Joins gold ↔ unified, writes:

- `eval/scores.csv` — one row per run with overall `quality_score`
  (weighted mean of field F1s) and per-field P/R/F1.
- `eval/scores_by_segment.csv` — same metrics broken down by
  `hazard_family`, `language`, and `region`, with `n_docs_in_segment`.

Field weights are at the top of `score_against_gold.py` (`FIELD_WEIGHTS`)
— tune to project priorities.

### 5. Recommendation — `recommend.py`

```
python3 eval/recommend.py
```

Reads the score CSVs and writes `eval/RECOMMENDATION.md` — a one-page
summary with:

- Best overall run (raw quality).
- Cost-adjusted pick: medium is preferred if it's within 0.05 F1 of
  the best high run (`--cost-margin` to override).
- Per-segment routing suggestions, only when a segment with ≥5 gold
  docs has a winner that beats the global pick by >0.10 F1
  (`--segment-margin`, `--segment-min-docs` to override).

## Cleaning up between iterations

If you re-sample or re-label:

```
rm eval/{unified.parquet,review_sample.csv,gold.csv,scores.csv,scores_by_segment.csv,RECOMMENDATION.md}
```

Then re-run from Stage 1.

## Files

| Stage | Script | Inputs | Outputs |
|-------|--------|--------|---------|
| 1 | `triage_metrics.py` | `outputs/test_400_gpt5_*` | `triage.csv` |
| 2.1 | `unified_table.py` | `outputs/test_400_gpt5_*` | `unified.parquet` |
| 2.2 | `sample_for_review.py` | `unified.parquet` | `review_sample.csv` |
| 3 | `gold_template.py` | `review_sample.csv` | `gold.csv` (template) |
| 4 | `score_against_gold.py` | `gold.csv`, `unified.parquet` | `scores.csv`, `scores_by_segment.csv` |
| 5 | `recommend.py` | `scores.csv`, `scores_by_segment.csv` | `RECOMMENDATION.md` |
