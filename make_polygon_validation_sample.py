"""
make_polygon_validation_sample.py
---------------------------------

Build a stratified manual-review CSV from the polygon-resolver output.

Strata
------
* high_confidence : confidence >= --high-conf-threshold AND no fallback used AND
                    geometry came directly from an OSM polygon AND admin_level
                    is a recognised admin tier.
* low_confidence  : resolved rows that fail any of the above (low conf, or
                    needed a fallback, or the geometry is a bbox/point buffer,
                    or admin_level == 'other').
* unresolved      : source (asset, location, dates) rows that produced no
                    geometry at all (anti-join from the source JSONL).

Each row in the review CSV has:
* enough source context to find and read the document (asset_key, content_url,
  title, hazard, start/end dates, original location_text)
* what the resolver decided (admin level, geometry source, confidence,
  display_name from Nominatim, fallback step used)
* a clickable OSM link centred on the geometry centroid, with zoom adapted to
  the polygon's area (so country polygons open zoomed-out and city polygons
  open zoomed-in)
* empty reviewer columns: location_correct, geometry_correct,
  correct_admin_level, notes

This slots next to visualize_event_extractions.py and reuses its
--input/--output-dir convention.

Usage
-----

    python make_polygon_validation_sample.py \
      --extractions outputs/test_400/event_extractions.jsonl \
      --geometries  outputs/test_400/event_geometries.parquet \
      --output-dir  outputs/test_400_polygon_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from run_polygon_resolution import explode_extractions, event_uuid


LOG = logging.getLogger("polygon_validation")

ADMIN_TIERS = {"country", "admin1", "admin2", "city", "neighbourhood"}


# ---------------------------------------------------------------------------
# Map URL — OSM, zoom adapted to polygon area
# ---------------------------------------------------------------------------

def _zoom_from_area_km2(area_km2: float) -> int:
    if area_km2 is None or pd.isna(area_km2):
        return 10
    if area_km2 > 100_000:
        return 4
    if area_km2 > 10_000:
        return 6
    if area_km2 > 1_000:
        return 8
    if area_km2 > 100:
        return 10
    if area_km2 > 10:
        return 12
    return 14


def osm_url(geom, area_km2: float) -> str:
    if geom is None or geom.is_empty:
        return ""
    c = geom.centroid
    lat, lon = c.y, c.x
    zoom = _zoom_from_area_km2(area_km2)
    return (
        f"https://www.openstreetmap.org/"
        f"?mlat={lat:.5f}&mlon={lon:.5f}#map={zoom}/{lat:.5f}/{lon:.5f}"
    )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def label_strata(g: gpd.GeoDataFrame, conf_threshold: float) -> pd.Series:
    """Return 'high_confidence' or 'low_confidence' per resolved row."""
    is_high = (
        (g["resolution_confidence"] >= conf_threshold)
        & (g["resolution_used_fallback"] == 0)
        & (g["resolution_geometry_source"] == "polygon")
        & (g["resolution_admin_level"].isin(ADMIN_TIERS))
    )
    return pd.Series(["high_confidence" if h else "low_confidence" for h in is_high],
                     index=g.index)


def find_unresolved(jsonl_path: Path, resolved_uuids: set[str]) -> pd.DataFrame:
    """Return one row per (asset, location) pair from the JSONL whose uuid
    is NOT in `resolved_uuids`."""
    out = []
    for row in explode_extractions(jsonl_path):
        uid = event_uuid(row["asset_key"], row["location_text"],
                         row["start_date"], row["end_date"])
        if uid in resolved_uuids:
            continue
        out.append({
            "uuid": uid,
            "asset_key": row["asset_key"],
            "title": row["title"],
            "content_url": row["content_url"],
            "hazard": row["hazard"],
            "location_text": row["location_text"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "event_dates_raw": row["event_dates_raw"],
        })
    return pd.DataFrame(out)


def build_event_dates_raw_lookup(jsonl_path: Path) -> dict[str, str]:
    """uuid -> raw `event_dates` string from the JSONL. Used to attach the
    raw extraction to resolved rows in the parquet (which doesn't carry it)."""
    out: dict[str, str] = {}
    for row in explode_extractions(jsonl_path):
        uid = event_uuid(row["asset_key"], row["location_text"],
                         row["start_date"], row["end_date"])
        out[uid] = row["event_dates_raw"]
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def take(df: pd.DataFrame, n: int, seed: int, label: str) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0]
    if len(df) <= n:
        LOG.warning("Stratum %r has only %d rows; taking all.", label, len(df))
        return df
    return df.sample(n=n, random_state=seed)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

REVIEWER_COLUMNS = [
    "location_correct",       # Y / N / Partial — judges the EXTRACTOR (was this a real affected place?)
    "geometry_correct",       # Y / N / Partial — judges the RESOLVER (does the polygon match the text?)
    "correct_admin_level",    # free text — what level the location really is
    "date_correct",           # Y / N / Partial — judges the EXTRACTOR's date extraction
    "correct_dates",          # free text — what the dates should have been (YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD)
    "notes",                  # free text
]

PRESENTATION_COLUMNS = [
    # source context
    "stratum", "review_id",
    "hazard",
    "start_date", "end_date", "event_dates_raw",
    "location_text",
    "title", "content_url",
    # what the resolver did
    "resolution_admin_level", "resolution_geometry_source",
    "resolution_confidence", "resolution_used_fallback",
    "resolution_display_name", "resolution_query",
    "area_km2", "map_url",
    # reviewer fields (empty)
    *REVIEWER_COLUMNS,
    # joinable
    "uuid", "asset_key",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--extractions", type=Path, required=True,
                   help="Source JSONL produced by app_event_focus.py")
    p.add_argument("--geometries", type=Path, required=True,
                   help="Parquet produced by run_polygon_resolution.py")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the review CSV and summary JSON")
    p.add_argument("--high-conf-n", type=int, default=15)
    p.add_argument("--low-conf-n", type=int, default=15)
    p.add_argument("--unresolved-n", type=int, default=10)
    p.add_argument("--high-conf-threshold", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.geometries.exists():
        LOG.error("Geometries parquet not found: %s", args.geometries)
        return 1
    if not args.extractions.exists():
        LOG.error("Extractions JSONL not found: %s", args.extractions)
        return 1

    LOG.info("Loading geometries from %s", args.geometries)
    gdf = gpd.read_parquet(args.geometries)
    LOG.info("Loaded %d resolved rows", len(gdf))

    if "resolution_geometry_source" not in gdf.columns:
        LOG.error(
            "Column 'resolution_geometry_source' is missing from the parquet — "
            "you're using an older resolver output. Re-run run_polygon_resolution.py "
            "(it'll be cheap; the Nominatim cache is hot) to regenerate the parquet."
        )
        return 1

    # 1. Label resolved rows.
    gdf = gdf.copy()
    gdf["stratum"] = label_strata(gdf, args.high_conf_threshold)
    gdf["map_url"] = [osm_url(g, a) for g, a in zip(gdf.geometry, gdf["area_km2"])]

    # Attach the raw `event_dates` string from the JSONL so the reviewer can
    # see what the LLM actually extracted (the explode in the resolver
    # collapses 3+ dates to [min, max] silently).
    raw_dates = build_event_dates_raw_lookup(args.extractions)
    gdf["event_dates_raw"] = gdf["uuid"].astype(str).map(raw_dates).fillna("")

    # 2. Find unresolved rows by anti-joining the source JSONL on uuid.
    LOG.info("Anti-joining extractions to find unresolved rows...")
    unresolved = find_unresolved(args.extractions, set(gdf["uuid"].astype(str)))
    LOG.info("Unresolved rows: %d", len(unresolved))
    unresolved["stratum"] = "unresolved"
    for col in (
        "resolution_admin_level", "resolution_geometry_source",
        "resolution_confidence", "resolution_used_fallback",
        "resolution_display_name", "resolution_query",
        "area_km2", "map_url",
    ):
        unresolved[col] = pd.NA

    # 3. Sample each stratum.
    high_pool = gdf[gdf["stratum"] == "high_confidence"]
    low_pool = gdf[gdf["stratum"] == "low_confidence"]
    LOG.info("Pool sizes: high=%d  low=%d  unresolved=%d",
             len(high_pool), len(low_pool), len(unresolved))

    samples = pd.concat([
        take(high_pool, args.high_conf_n, args.seed, "high_confidence"),
        take(low_pool, args.low_conf_n, args.seed, "low_confidence"),
        take(unresolved, args.unresolved_n, args.seed, "unresolved"),
    ], ignore_index=True)

    # 4. Reviewer columns (empty).
    for col in REVIEWER_COLUMNS:
        samples[col] = ""

    # 5. Sequential review_id, ordered by stratum then random.
    samples = samples.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    samples["review_id"] = range(1, len(samples) + 1)

    # Drop the geometry column from the CSV — reviewers use the map_url instead.
    if "geometry" in samples.columns:
        samples = samples.drop(columns=["geometry"])

    # Project to the presentation column order; create any missing columns.
    for col in PRESENTATION_COLUMNS:
        if col not in samples.columns:
            samples[col] = pd.NA
    samples = samples[PRESENTATION_COLUMNS]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "polygon_validation_sample.csv"
    samples.to_csv(csv_path, index=False)

    # Summary file with the parameters used and the per-stratum counts —
    # useful for reproducibility and for downstream analysis later.
    summary = {
        "extractions": str(args.extractions),
        "geometries": str(args.geometries),
        "params": {
            "high_conf_n": args.high_conf_n,
            "low_conf_n": args.low_conf_n,
            "unresolved_n": args.unresolved_n,
            "high_conf_threshold": args.high_conf_threshold,
            "seed": args.seed,
        },
        "pool_sizes": {
            "high_confidence": int(len(high_pool)),
            "low_confidence": int(len(low_pool)),
            "unresolved": int(len(unresolved)),
        },
        "sampled_counts": (
            samples["stratum"].value_counts().to_dict()
        ),
    }
    summary_path = args.output_dir / "polygon_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    LOG.info("Wrote %d review rows -> %s", len(samples), csv_path)
    LOG.info("Wrote summary -> %s", summary_path)
    LOG.info("Sampled per stratum: %s", summary["sampled_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
