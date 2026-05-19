"""
eval/triage_metrics.py — Stage 1 of the prompt × effort selection funnel.

Reads all 8 extraction runs at outputs/test_400_gpt5_{effort}_prompt{NN}/
and produces a single 8-row CSV of label-free quality signals. The goal
is to eliminate clear losers before any human review, not to crown a
winner — that comes after the gold-labelled scoring in Stage 4.

The signals are:

* verifiable_event_rate, single_event_rate, null_hazard_rate,
  mean_locations_per_doc, mean_dates_per_doc, mean_confidence_high_share
  — basic emission rates from event_extractions.jsonl.
* polygon_resolution_rate — fraction of (asset, location) pairs that
  produced a geometry. Re-aggregated from the parquet rather than
  parsed from logs so it's robust to log absence.
* country_hallucination_rate — fraction of resolved rows where the
  Nominatim display_name's country tail disagrees with the LLM's
  countries_raw field. Direct evidence that this is a real, varying
  signal across prompts: Tegucigalpa→Albania, Bucharest→Algeria,
  Beira→Afghanistan all came out of certain prompts but not others.
* country_overshoot_rate — fraction of resolved rows where
  geometry_overshoot=True (resolver had to climb to a whole-country
  polygon for a sub-national query, a structural overshoot signal).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_TRIAGE_CSV = REPO_ROOT / "eval" / "triage.csv"

# Eight run directories follow this naming convention. The runner is the
# extraction app driven with --all + --reasoning-effort {high,medium}.
RUN_DIR_RE = re.compile(r"^test_400_gpt5_(?P<effort>high|medium)_prompt(?P<prompt>0[1-4])$")


def _truthy(s) -> bool:
    """JSONL booleans come through as strings ('True', 'False', 'None')."""
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() == "true"


def _split_pipes(s) -> list[str]:
    if not s or str(s).strip().lower() in {"", "none"}:
        return []
    return [p.strip() for p in str(s).split("|") if p.strip()]


def _country_tail(display_name: str) -> str:
    """
    Last comma-separated token of a Nominatim display_name is the country.
    e.g. 'Miyagi Prefecture, Japan' -> 'japan'. Returns lowercased; empty
    if the field is blank.
    """
    if not display_name:
        return ""
    tail = display_name.rsplit(",", 1)[-1].strip().lower()
    return tail


def _raw_country_tokens(countries_raw: str) -> list[str]:
    if not countries_raw or countries_raw.lower() == "none":
        return []
    return [c.strip().lower() for c in countries_raw.split(",") if c.strip()]


# Aliases: Nominatim's display name uses official English forms while LLM
# countries_raw uses common forms. Without this map, true matches look like
# hallucinations and inflate the rate. Keep this conservative — only add
# pairs we've seen disagree in practice.
_COUNTRY_ALIASES = {
    "united states": {"united states", "united states of america", "usa", "us"},
    "united states of america": {"united states", "united states of america", "usa", "us"},
    "usa": {"united states", "united states of america", "usa", "us"},
    "united kingdom": {"united kingdom", "uk", "great britain"},
    "uk": {"united kingdom", "uk", "great britain"},
    "russia": {"russia", "russian federation"},
    "russian federation": {"russia", "russian federation"},
    "south korea": {"south korea", "korea", "republic of korea"},
    "north korea": {"north korea", "democratic people's republic of korea", "dprk"},
    "iran": {"iran", "islamic republic of iran"},
    "venezuela": {"venezuela", "bolivarian republic of venezuela"},
    "bolivia": {"bolivia", "plurinational state of bolivia"},
    "tanzania": {"tanzania", "united republic of tanzania"},
    "ivory coast": {"ivory coast", "côte d'ivoire", "cote d'ivoire"},
    "côte d'ivoire": {"ivory coast", "côte d'ivoire", "cote d'ivoire"},
    "cape verde": {"cape verde", "cabo verde"},
    "myanmar": {"myanmar", "burma"},
    "turkey": {"turkey", "türkiye", "turkiye"},
    "türkiye": {"turkey", "türkiye", "turkiye"},
    "czech republic": {"czech republic", "czechia"},
    "syria": {"syria", "syrian arab republic"},
    "moldova": {"moldova", "republic of moldova"},
    "laos": {"laos", "lao people's democratic republic"},
    "vietnam": {"vietnam", "viet nam"},
    "viet nam": {"vietnam", "viet nam"},
    "macedonia": {"macedonia", "north macedonia"},
    "north macedonia": {"macedonia", "north macedonia"},
    "palestine": {"palestine", "palestinian territory", "state of palestine"},
    "congo": {"congo", "republic of the congo", "congo-brazzaville"},
    "drc": {"drc", "democratic republic of the congo", "congo-kinshasa"},
    "democratic republic of the congo": {"drc", "democratic republic of the congo", "congo-kinshasa"},
}


def _country_matches(raw_tokens: list[str], display_tail: str) -> bool:
    """
    True if any of the LLM's claimed countries plausibly matches the
    geocoder's country tail. Returns True when raw_tokens is empty
    (nothing to contradict) so we don't penalise undated/uncountried
    documents.
    """
    if not raw_tokens:
        return True
    if not display_tail:
        return True  # display blank shouldn't count as a hallucination
    for tok in raw_tokens:
        if tok == display_tail:
            return True
        aliases = _COUNTRY_ALIASES.get(tok) or {tok}
        if display_tail in aliases:
            return True
        aliases_tail = _COUNTRY_ALIASES.get(display_tail) or {display_tail}
        if tok in aliases_tail:
            return True
    return False


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def metrics_for_run(run_dir: Path) -> dict:
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        raise ValueError(f"unexpected run dir name: {run_dir.name}")
    effort = m.group("effort")
    prompt = m.group("prompt")

    jsonl = run_dir / "event_extractions.jsonl"
    parquet = run_dir / "event_geometries.parquet"

    rows = _read_jsonl(jsonl)
    n = len(rows)
    if n == 0:
        return {
            "run_id": f"{effort}_prompt{prompt}",
            "effort": effort,
            "prompt": prompt,
            "n_docs_processed": 0,
        }

    verifiable = sum(_truthy(r.get("is_verifiable_event")) for r in rows)
    single = sum(_truthy(r.get("is_single_actual_event")) for r in rows)
    null_hazard = sum(
        1 for r in rows
        if _truthy(r.get("is_verifiable_event"))
        and str(r.get("event_hazard") or "").strip().lower() in {"", "none"}
    )
    locs_per_doc = [
        len(_split_pipes(r.get("affected_locations")))
        for r in rows if _truthy(r.get("is_verifiable_event"))
    ]
    dates_per_doc = [
        len(_split_pipes(r.get("event_dates")))
        for r in rows if _truthy(r.get("is_verifiable_event"))
    ]
    high_conf = sum(
        1 for r in rows
        if _truthy(r.get("is_verifiable_event"))
        and str(r.get("event_confidence") or "").strip().lower() == "high"
    )

    out = {
        "run_id": f"{effort}_prompt{prompt}",
        "effort": effort,
        "prompt": prompt,
        "n_docs_processed": n,
        "verifiable_event_rate": verifiable / n,
        "single_event_rate": single / n,
        "null_hazard_rate_when_verifiable": (null_hazard / verifiable) if verifiable else 0.0,
        "mean_locations_per_verifiable_doc": (sum(locs_per_doc) / len(locs_per_doc)) if locs_per_doc else 0.0,
        "mean_dates_per_verifiable_doc": (sum(dates_per_doc) / len(dates_per_doc)) if dates_per_doc else 0.0,
        "high_confidence_share_when_verifiable": (high_conf / verifiable) if verifiable else 0.0,
    }

    # ---- resolver-derived signals ---------------------------------------
    if not parquet.exists():
        out.update({
            "polygon_resolution_rate": float("nan"),
            "country_hallucination_rate": float("nan"),
            "country_overshoot_rate": float("nan"),
            "n_resolved_rows": 0,
            "n_location_pairs": 0,
        })
        return out

    df = pd.read_parquet(parquet)
    n_resolved = len(df)

    # Total location pairs attempted = resolved + unresolved (which aren't
    # in the parquet). We approximate the denominator from the jsonl: sum
    # of split pipe lengths across verifiable rows.
    n_pairs = sum(
        len(_split_pipes(r.get("affected_locations")))
        for r in rows if _truthy(r.get("is_verifiable_event"))
    )

    hallucinations = 0
    for _, r in df.iterrows():
        raw = _raw_country_tokens(r.get("countries_raw") or "")
        tail = _country_tail(r.get("resolution_display_name") or "")
        if not _country_matches(raw, tail):
            hallucinations += 1

    overshoot = int(df["geometry_overshoot"].sum()) if "geometry_overshoot" in df.columns else 0

    out.update({
        "n_location_pairs": n_pairs,
        "n_resolved_rows": n_resolved,
        "polygon_resolution_rate": (n_resolved / n_pairs) if n_pairs else 0.0,
        "country_hallucination_rate": (hallucinations / n_resolved) if n_resolved else 0.0,
        "country_overshoot_rate": (overshoot / n_resolved) if n_resolved else 0.0,
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_TRIAGE_CSV)
    args = ap.parse_args()

    run_dirs = sorted(
        d for d in args.outputs_dir.iterdir()
        if d.is_dir() and RUN_DIR_RE.match(d.name)
    )
    if not run_dirs:
        raise SystemExit(f"No matching run dirs under {args.outputs_dir}")

    rows = [metrics_for_run(d) for d in run_dirs]
    df = pd.DataFrame(rows).sort_values(["effort", "prompt"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} runs)")
    # Print the table to stdout for quick eyeballing.
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
