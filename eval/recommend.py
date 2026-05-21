"""
eval/recommend.py — Stage 5 of the funnel.

Reads `eval/scores.csv` and `eval/scores_by_segment.csv` (Stage 4) and
emits `eval/RECOMMENDATION.md`: a one-page summary with

  * Best overall prompt+effort (raw quality)
  * Cost-adjusted choice: if a medium run is within `--cost-margin` F1
    points of the best high run, prefer medium (user preference is "yes,
    prefer medium if close" — default margin is 0.05).
  * Per-segment routing suggestions: a per-segment winner is only
    emitted when (a) it beats the global winner by more than
    `--segment-margin` F1 points and (b) the segment has at least
    `--segment-min-docs` documents in the gold sample.

The decision rules are configurable on the command line so the
recommendation logic doesn't get baked into a hard-coded threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = REPO_ROOT / "eval" / "scores.csv"
DEFAULT_SCORES_SEG = REPO_ROOT / "eval" / "scores_by_segment.csv"
DEFAULT_OUT = REPO_ROOT / "eval" / "RECOMMENDATION.md"

# Prompt name lookup (kept in sync with groundsource.prompts PROMPTS tuple). One
# line per prompt, matching the id used in run_id like 'prompt01'.
PROMPT_NAMES = {
    "01": ("baseline_strict_location",
           "High-recall gate with strict sub-national location discipline."),
    "02": ("case_study_high_recall",
           "Tuned for long multi-section documents; digs for buried events."),
    "03": ("verification_first",
           "Conservative — emits only explicitly source-supported facts."),
    "04": ("compact_schema_focused",
           "Factory-style minimal preamble with strict structural discipline."),
}


def _split_run(run_id: str) -> tuple[str, str]:
    """high_prompt03 -> ('high', '03')."""
    effort, prompt = run_id.split("_prompt")
    return effort, prompt


def _prompt_label(run_id: str) -> str:
    effort, p = _split_run(run_id)
    name, _ = PROMPT_NAMES.get(p, (f"prompt{p}", ""))
    return f"{effort}/{name} ({run_id})"


def _f(v) -> str:
    """Format an F1 score as percent."""
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return str(v)


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a small markdown table from a DataFrame."""
    cols = [c for c in columns if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--seg-scores", type=Path, default=DEFAULT_SCORES_SEG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cost-margin", type=float, default=0.05,
                    help="Prefer medium if its quality is within this many "
                         "F1 points of the best high (default 0.05 = 5%%).")
    ap.add_argument("--segment-margin", type=float, default=0.10,
                    help="Per-segment winner must beat global winner by "
                         "more than this on the segment's F1 (default 0.10).")
    ap.add_argument("--segment-min-docs", type=int, default=5,
                    help="Per-segment winner requires at least this many "
                         "gold docs in the segment (default 5).")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    seg = pd.read_csv(args.seg_scores)

    if scores.empty:
        raise SystemExit(f"{args.scores} is empty")

    scores = scores.sort_values("quality_score", ascending=False).reset_index(drop=True)
    scores["effort"], scores["prompt"] = zip(*scores["run_id"].map(_split_run))

    overall_best = scores.iloc[0]
    best_high = scores[scores["effort"] == "high"].head(1)
    best_medium = scores[scores["effort"] == "medium"].head(1)

    # Cost-adjusted: prefer medium if the best medium run is within
    # --cost-margin of the best high run.
    cost_choice = overall_best["run_id"]
    cost_rationale = "No medium run is close enough to prefer on cost grounds."
    if not best_high.empty and not best_medium.empty:
        hq = float(best_high["quality_score"].iloc[0])
        mq = float(best_medium["quality_score"].iloc[0])
        if (hq - mq) <= args.cost_margin:
            cost_choice = best_medium["run_id"].iloc[0]
            cost_rationale = (
                f"Best medium ({best_medium['run_id'].iloc[0]}, F1={mq:.3f}) is "
                f"within {args.cost_margin:.2f} of best high "
                f"({best_high['run_id'].iloc[0]}, F1={hq:.3f}). "
                f"Pick medium for lower cost / latency."
            )
        else:
            cost_rationale = (
                f"Best medium ({best_medium['run_id'].iloc[0]}, F1={mq:.3f}) trails "
                f"best high ({best_high['run_id'].iloc[0]}, F1={hq:.3f}) by "
                f"{hq - mq:.3f}, above the {args.cost_margin:.2f} margin. "
                f"Pick high."
            )

    # Per-segment winners: for each (segment_kind, segment_value) with
    # enough docs, find the run that beats the *cost-adjusted overall
    # winner* by more than --segment-margin on that segment's F1.
    overall_for_seg = cost_choice
    overall_seg_lookup = seg[seg["run_id"] == overall_for_seg].set_index(
        ["segment_kind", "segment_value"])["quality_score"].to_dict()
    seg_winners: list[dict] = []
    if not seg.empty:
        big = seg[seg["n_docs_in_segment"] >= args.segment_min_docs]
        for (kind, val), sub in big.groupby(["segment_kind", "segment_value"]):
            sub = sub.sort_values("quality_score", ascending=False)
            top = sub.iloc[0]
            global_q = overall_seg_lookup.get((kind, val))
            if global_q is None:
                continue
            if top["run_id"] == overall_for_seg:
                continue
            if float(top["quality_score"]) - float(global_q) > args.segment_margin:
                seg_winners.append({
                    "segment_kind": kind,
                    "segment_value": val,
                    "n_docs": int(top["n_docs_in_segment"]),
                    "winner_run": top["run_id"],
                    "winner_q": float(top["quality_score"]),
                    "global_q": float(global_q),
                    "delta": float(top["quality_score"]) - float(global_q),
                })

    # ---- render the report ------------------------------------------------
    lines: list[str] = []
    lines.append("# Prompt × effort selection — recommendation")
    lines.append("")
    lines.append(f"**Overall best:** `{overall_best['run_id']}` "
                 f"(F1 = {overall_best['quality_score']:.3f}) — {_prompt_label(overall_best['run_id'])}")
    lines.append("")
    lines.append(f"**Cost-adjusted pick:** `{cost_choice}` — _{cost_rationale}_")
    lines.append("")
    if seg_winners:
        lines.append("**Per-segment routing (optional):** the following segments would benefit "
                     f"from routing to a non-default prompt (Δ > {args.segment_margin:.2f}, "
                     f"≥{args.segment_min_docs} docs):")
        lines.append("")
        seg_df = pd.DataFrame(seg_winners)
        lines.append(_table(seg_df, ["segment_kind", "segment_value", "n_docs",
                                     "winner_run", "winner_q", "global_q", "delta"]))
        lines.append("")
    else:
        lines.append("**Per-segment routing:** no segment beats the global winner by "
                     f"more than {args.segment_margin:.2f} on ≥{args.segment_min_docs} docs. "
                     "Use the cost-adjusted pick globally.")
        lines.append("")

    lines.append("## All runs, ranked")
    lines.append("")
    cols = ["run_id", "quality_score", "is_verifiable_event__f1",
            "event_hazard_family__f1", "affected_locations__f1",
            "countries_doc__f1", "event_dates_overlap__f1"]
    lines.append(_table(scores, cols))
    lines.append("")

    lines.append("## Prompt cheat sheet")
    lines.append("")
    for pid, (name, desc) in PROMPT_NAMES.items():
        lines.append(f"- **prompt{pid} — {name}**: {desc}")
    lines.append("")
    lines.append(
        "_Generated by `eval/recommend.py`. Decision thresholds: "
        f"cost-margin={args.cost_margin:.2f}, segment-margin={args.segment_margin:.2f}, "
        f"segment-min-docs={args.segment_min_docs}._"
    )

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
    print()
    print("\n".join(lines[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
