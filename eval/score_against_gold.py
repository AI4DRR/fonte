"""
eval/score_against_gold.py — Stage 4 of the funnel.

Joins `eval/gold.csv` (hand-labelled, Stage 3) with `eval/unified.parquet`
(per-run outputs, Stage 2) and scores each of the 8 runs against the
labels. Produces two CSVs:

  eval/scores.csv             — one row per run, overall metrics.
  eval/scores_by_segment.csv  — one row per (run, segment_kind, segment_value),
                                where segment_kind is one of:
                                  hazard_family   (gold-derived)
                                  language        (jsonl-derived)
                                  region          (country → continent)

For each run we compute per-field precision/recall/F1 on the gold rows:

  is_verifiable_event   binary vs is_real_event (T/F; "unclear" skipped)
  event_hazard_family   family-normalised match vs hazard family of true_hazard
  affected_locations    set F1 vs true_locations
  countries_doc         document-level `countries` exact match vs true_country
  event_dates_overlap   any-overlap rate vs (true_date_start, true_date_end)

The per-run aggregate `quality_score` is a weighted mean of field F1s;
weights are at the top of the file and can be tuned per project priorities.

The per-segment table also carries `n_docs_in_segment` so callers can
filter to segments with ≥5 docs before claiming a per-segment winner
(Stage 5 enforces this threshold).
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = REPO_ROOT / "eval" / "gold.csv"
DEFAULT_UNIFIED = REPO_ROOT / "eval" / "unified.parquet"
DEFAULT_SCORES = REPO_ROOT / "eval" / "scores.csv"
DEFAULT_SCORES_SEG = REPO_ROOT / "eval" / "scores_by_segment.csv"

# Weights for the aggregate quality_score. Sum needn't equal 1 — the
# script normalises. Tune per project priority. Defaults emphasise the
# things that drive downstream pipeline correctness: did we surface the
# event, did we get the hazard family, did we name the right places.
FIELD_WEIGHTS = {
    "is_verifiable_event": 1.0,
    "event_hazard_family": 1.5,
    "affected_locations": 1.5,
    "countries_doc": 1.0,
    "event_dates_overlap": 0.5,
}

# Same hazard family taxonomy as the sampler — kept in sync by copy
# rather than import to keep these scripts independent.
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


def _hazard_family(s: str) -> str:
    if not s:
        return "other"
    for fam, pat in HAZARD_FAMILIES:
        if pat.search(s):
            return fam
    return "other"


# Rough country → continent mapping for the "region" segment. Only the
# countries we actually expect to encounter in the corpus need entries;
# unknown ones fall to "unknown" and just don't get a region-level
# breakdown. Pragmatic; not authoritative.
_CONTINENT = {
    # Africa
    "kenya": "africa", "ethiopia": "africa", "somalia": "africa", "uganda": "africa",
    "tanzania": "africa", "south africa": "africa", "mozambique": "africa",
    "nigeria": "africa", "ghana": "africa", "egypt": "africa", "morocco": "africa",
    "algeria": "africa", "tunisia": "africa", "sudan": "africa", "zambia": "africa",
    "zimbabwe": "africa", "malawi": "africa", "rwanda": "africa", "burundi": "africa",
    "democratic republic of the congo": "africa", "drc": "africa",
    "côte d'ivoire": "africa", "ivory coast": "africa", "senegal": "africa",
    "mali": "africa", "niger": "africa", "burkina faso": "africa", "chad": "africa",
    "cameroon": "africa", "madagascar": "africa", "angola": "africa", "namibia": "africa",
    # Asia
    "japan": "asia", "china": "asia", "india": "asia", "indonesia": "asia",
    "pakistan": "asia", "bangladesh": "asia", "philippines": "asia", "vietnam": "asia",
    "thailand": "asia", "myanmar": "asia", "nepal": "asia", "sri lanka": "asia",
    "south korea": "asia", "north korea": "asia", "iran": "asia", "iraq": "asia",
    "afghanistan": "asia", "syria": "asia", "lebanon": "asia", "israel": "asia",
    "saudi arabia": "asia", "yemen": "asia", "jordan": "asia", "kazakhstan": "asia",
    "uzbekistan": "asia", "turkmenistan": "asia", "tajikistan": "asia", "kyrgyzstan": "asia",
    "mongolia": "asia", "malaysia": "asia", "singapore": "asia", "cambodia": "asia",
    "laos": "asia", "bhutan": "asia", "maldives": "asia",
    # Europe
    "france": "europe", "germany": "europe", "italy": "europe", "spain": "europe",
    "united kingdom": "europe", "uk": "europe", "ireland": "europe", "portugal": "europe",
    "netherlands": "europe", "belgium": "europe", "switzerland": "europe", "austria": "europe",
    "poland": "europe", "czech republic": "europe", "slovakia": "europe", "hungary": "europe",
    "romania": "europe", "bulgaria": "europe", "greece": "europe", "albania": "europe",
    "serbia": "europe", "croatia": "europe", "slovenia": "europe", "bosnia": "europe",
    "ukraine": "europe", "russia": "europe", "belarus": "europe", "moldova": "europe",
    "sweden": "europe", "norway": "europe", "finland": "europe", "denmark": "europe",
    "iceland": "europe", "turkey": "europe", "türkiye": "europe",
    # Americas
    "united states": "americas", "united states of america": "americas", "usa": "americas",
    "canada": "americas", "mexico": "americas", "guatemala": "americas",
    "honduras": "americas", "el salvador": "americas", "nicaragua": "americas",
    "costa rica": "americas", "panama": "americas", "cuba": "americas",
    "dominican republic": "americas", "haiti": "americas", "jamaica": "americas",
    "venezuela": "americas", "colombia": "americas", "ecuador": "americas",
    "peru": "americas", "bolivia": "americas", "chile": "americas",
    "argentina": "americas", "uruguay": "americas", "paraguay": "americas",
    "brazil": "americas",
    # Oceania
    "australia": "oceania", "new zealand": "oceania", "fiji": "oceania",
    "papua new guinea": "oceania", "vanuatu": "oceania", "solomon islands": "oceania",
    "tonga": "oceania", "samoa": "oceania",
}


def _region(country: str) -> str:
    if not country:
        return "unknown"
    return _CONTINENT.get(country.strip().lower(), "unknown")


# ---------------------------------------------------------------------------
# Field-level scoring
# ---------------------------------------------------------------------------

def _norm_bool(s) -> bool | None:
    if pd.isna(s):
        return None
    v = str(s).strip().lower()
    if v in {"true", "t", "yes", "y", "1"}:
        return True
    if v in {"false", "f", "no", "n", "0"}:
        return False
    return None  # "unclear", "" etc.


def _norm_set(s) -> set[str]:
    """Split a free-text locations field by ; or |, lowercase, strip."""
    if pd.isna(s) or not s:
        return set()
    parts = re.split(r"[;|]", str(s))
    return {p.strip().lower() for p in parts if p.strip()}


def _parse_date(s) -> date | None:
    if pd.isna(s) or not s:
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _pr_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def _score_binary(pred: list[bool | None], true: list[bool | None]) -> dict:
    """Counts only rows where both are non-None."""
    tp = fp = fn = tn = 0
    for p, t in zip(pred, true):
        if p is None or t is None:
            continue
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif (not p) and t:
            fn += 1
        else:
            tn += 1
    pr, re_, f = _pr_f1(tp, fp, fn)
    n = tp + fp + fn + tn
    return {"precision": pr, "recall": re_, "f1": f, "n_scored": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _score_set(pred_sets: list[set], true_sets: list[set]) -> dict:
    """Micro-averaged set P/R/F1 across rows."""
    tp = fp = fn = 0
    rows_scored = 0
    for p, t in zip(pred_sets, true_sets):
        if not t and not p:
            continue
        rows_scored += 1
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    pr, re_, f = _pr_f1(tp, fp, fn)
    return {"precision": pr, "recall": re_, "f1": f, "n_scored": rows_scored,
            "tp": tp, "fp": fp, "fn": fn}


def _score_string_match(pred_vals: list[str], true_vals: list[str]) -> dict:
    """Exact-match accuracy treated as both precision and recall."""
    matched = 0
    n = 0
    for p, t in zip(pred_vals, true_vals):
        if not t:
            continue
        n += 1
        if (p or "").strip().lower() == t.strip().lower():
            matched += 1
    acc = (matched / n) if n else 0.0
    return {"precision": acc, "recall": acc, "f1": acc, "n_scored": n,
            "tp": matched, "fp": n - matched, "fn": n - matched}


def _score_date_overlap(
    pred_starts: list[date | None], pred_ends: list[date | None],
    true_starts: list[date | None], true_ends: list[date | None],
) -> dict:
    """Range-overlap rate: predicted ∩ true non-empty counts as a hit."""
    n = 0
    hits = 0
    for ps, pe, ts, te in zip(pred_starts, pred_ends, true_starts, true_ends):
        if ts is None or te is None:
            continue
        n += 1
        if ps is None or pe is None:
            continue
        # Overlap iff ps <= te and ts <= pe
        if ps <= te and ts <= pe:
            hits += 1
    rate = (hits / n) if n else 0.0
    return {"precision": rate, "recall": rate, "f1": rate, "n_scored": n,
            "tp": hits, "fp": n - hits, "fn": n - hits}


# ---------------------------------------------------------------------------
# Extracting per-run vectors from unified.parquet
# ---------------------------------------------------------------------------

def _run_ids_from_columns(cols) -> list[str]:
    return sorted({c.split("__", 1)[0] for c in cols if "__" in c})


def _extract_event_dates(s) -> tuple[date | None, date | None]:
    """JSONL event_dates is pipe-separated YYYY-MM-DD. Take min/max."""
    if pd.isna(s) or not s:
        return None, None
    parts = [p.strip() for p in str(s).split("|") if p.strip()]
    parsed = [d for d in (_parse_date(p) for p in parts) if d is not None]
    if not parsed:
        return None, None
    return min(parsed), max(parsed)


def _score_run(run_id: str, joined: pd.DataFrame) -> dict:
    """All field-level scores for one run, on the labelled subset."""
    # Build per-doc prediction vectors.
    pred_verif: list[bool | None] = []
    pred_haz_fam: list[str] = []
    pred_locs: list[set] = []
    pred_country: list[str] = []
    pred_start: list[date | None] = []
    pred_end: list[date | None] = []
    true_verif: list[bool | None] = []
    true_haz_fam: list[str] = []
    true_locs: list[set] = []
    true_country: list[str] = []
    true_start: list[date | None] = []
    true_end: list[date | None] = []

    for _, row in joined.iterrows():
        pred_verif.append(_norm_bool(row.get(f"{run_id}__is_verifiable_event")))
        pred_haz_fam.append(_hazard_family(str(row.get(f"{run_id}__event_hazard") or "")))
        pred_locs.append({s.lower() for s in re.split(r"\|", str(row.get(f"{run_id}__affected_locations") or "")) if s.strip()})
        # `countries` is doc-level — same across runs in principle, but we
        # still pull it per-run to catch divergence (it has been observed).
        pred_country.append(str(row.get("countries") or "").strip().lower())
        ps, pe = _extract_event_dates(row.get(f"{run_id}__event_dates"))
        pred_start.append(ps)
        pred_end.append(pe)

        true_verif.append(_norm_bool(row.get("is_real_event")))
        true_haz_fam.append(_hazard_family(str(row.get("true_hazard") or "")))
        true_locs.append(_norm_set(row.get("true_locations")))
        true_country.append(str(row.get("true_country") or "").strip().lower())
        true_start.append(_parse_date(row.get("true_date_start")))
        true_end.append(_parse_date(row.get("true_date_end")))

    # Skip hazard_family / location / date scoring on rows where
    # is_real_event=False — predictions there are vacuously "right" if
    # the run also said False, and don't carry signal beyond the binary
    # is_verifiable score. Filter aligned vectors.
    keep = [(t is True) for t in true_verif]
    def _f(xs): return [x for x, k in zip(xs, keep) if k]

    field_scores = {
        "is_verifiable_event": _score_binary(pred_verif, true_verif),
        "event_hazard_family": _score_string_match(_f(pred_haz_fam), _f(true_haz_fam)),
        "affected_locations": _score_set(_f(pred_locs), _f(true_locs)),
        "countries_doc": _score_string_match(_f(pred_country), _f(true_country)),
        "event_dates_overlap": _score_date_overlap(
            _f(pred_start), _f(pred_end), _f(true_start), _f(true_end)
        ),
    }

    total_w = sum(FIELD_WEIGHTS.values())
    quality = sum(FIELD_WEIGHTS[f] * field_scores[f]["f1"] for f in FIELD_WEIGHTS) / total_w

    return {"run_id": run_id, "quality_score": quality, **{
        f"{f}__{m}": v[m] for f, v in field_scores.items() for m in ("precision", "recall", "f1", "n_scored")
    }}


def _split_segments(joined: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {segment_kind: {segment_value: rows}}."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    # hazard_family from gold's true_hazard
    fam = joined["true_hazard"].fillna("").map(_hazard_family)
    out["hazard_family"] = {v: joined[fam == v] for v in fam.unique()}
    # language from doc-level
    if "language" in joined.columns:
        out["language"] = {
            v: joined[joined["language"] == v]
            for v in joined["language"].dropna().unique()
        }
    # region from gold's true_country
    if "true_country" in joined.columns:
        reg = joined["true_country"].fillna("").map(_region)
        out["region"] = {v: joined[reg == v] for v in reg.unique()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--unified", type=Path, default=DEFAULT_UNIFIED)
    ap.add_argument("--scores-out", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--seg-out", type=Path, default=DEFAULT_SCORES_SEG)
    args = ap.parse_args()

    gold = pd.read_csv(args.gold, keep_default_na=False)
    unified = pd.read_parquet(args.unified)

    # Refuse if gold is empty (template state).
    labelled = gold[gold["is_real_event"].astype(str).str.strip() != ""]
    if labelled.empty:
        raise SystemExit(
            f"{args.gold} has no labels yet (is_real_event column is empty). "
            f"Fill it in before running this step."
        )

    joined = labelled.merge(unified, on="asset_key", how="left",
                            suffixes=("", "_unified"))
    missing = joined["disagreement_score"].isna().sum() if "disagreement_score" in joined.columns else 0
    if missing:
        print(f"WARNING: {missing} labelled rows didn't match any asset_key in unified.parquet")

    run_ids = _run_ids_from_columns([c for c in unified.columns if "__" in c])
    if not run_ids:
        raise SystemExit("unified parquet has no per-run columns")

    # Overall scores
    overall = pd.DataFrame([_score_run(rid, joined) for rid in run_ids])
    overall = overall.sort_values("quality_score", ascending=False).reset_index(drop=True)
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.scores_out, index=False)
    print(f"Wrote {args.scores_out}")
    print(overall[["run_id", "quality_score",
                   "is_verifiable_event__f1", "event_hazard_family__f1",
                   "affected_locations__f1", "countries_doc__f1",
                   "event_dates_overlap__f1"]].to_string(index=False))
    print()

    # Per-segment scores
    seg_rows: list[dict] = []
    segments = _split_segments(joined)
    for kind, values in segments.items():
        for val, sub in values.items():
            if sub.empty:
                continue
            for rid in run_ids:
                row = _score_run(rid, sub)
                row["segment_kind"] = kind
                row["segment_value"] = val
                row["n_docs_in_segment"] = len(sub)
                seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    # Order columns: identifiers first
    front = ["segment_kind", "segment_value", "n_docs_in_segment", "run_id", "quality_score"]
    rest = [c for c in seg_df.columns if c not in front]
    seg_df = seg_df[front + rest].sort_values(
        ["segment_kind", "segment_value", "quality_score"],
        ascending=[True, True, False],
    )
    seg_df.to_csv(args.seg_out, index=False)
    print(f"Wrote {args.seg_out} ({len(seg_df)} rows: {len(run_ids)} runs × segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
