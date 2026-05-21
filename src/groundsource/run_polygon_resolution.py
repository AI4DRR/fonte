"""
groundsource.run_polygon_resolution
-------------------------

Demo runner: read the pipeline's JSONL output (as produced by groundsource-extract),
resolve every `affected_locations` value to a geometry, and write a parquet
matching Google Groundsource's schema:

    uuid, area_km2, geometry, start_date, end_date

Plus sidecar columns for traceability (asset_key, hazard, location_text,
resolution_method, resolution_admin_level, resolution_confidence, content_url).

Usage
-----

    # one-time:
    pip install geopandas shapely pyarrow rapidfuzz requests

    # then:
    groundsource-polygons \
        --input  outputs/test_400/event_extractions.jsonl \
        --output outputs/test_400/event_geometries.parquet \
        --user-agent "undrr-groundsource/0.1 (your-email@undrr.org)"

The cache file (default: .nominatim_cache.json next to the script) makes
reruns essentially free, so iterate freely.

To swap geocoders for production (self-hosted Nominatim, GeoNames+GADM, etc.)
implement a class with a `resolve(text, country_hint=None) -> ResolutionResult`
method and pass it in place of NominatimResolver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from groundsource.polygon_resolver import NominatimResolver, ResolutionResult
from groundsource.geometry_postprocess import (
    HazardAwarePostprocessor,
    is_sub_national_hazard,
)


LOG = logging.getLogger("polygon_resolution")

# Equal-area projection used to compute area_km2. EPSG:6933 is the
# NSIDC EASE-Grid 2.0 Global — accurate to ~0.5% globally for areas.
EQUAL_AREA_EPSG = 6933


# ---------------------------------------------------------------------------
# Input parsing: matches groundsource-extract JSONL output.
# ---------------------------------------------------------------------------

def _split_pipes(s: Optional[str]) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _country_hint_from_countries(countries_raw: str) -> Optional[str]:
    """
    Convert the JSONL's `countries` field to an ISO-3166 alpha-2 hint for
    Nominatim. Only applies when EXACTLY one country is named: multi-country
    regional reports get no hint so we don't bias toward one of them.
    """
    if not countries_raw:
        return None
    parts = [p.strip() for p in countries_raw.split(",") if p.strip()]
    if len(parts) != 1:
        return None
    try:
        import pycountry  # type: ignore
    except ImportError:
        return None
    try:
        return pycountry.countries.lookup(parts[0]).alpha_2.lower()
    except LookupError:
        return None


def explode_extractions(jsonl_path: Path) -> Iterator[dict]:
    """
    Yield one dict per (asset, location) pair. Pipes split multi-locations and
    multi-dates. If two dates are present we treat them as a [start, end] range;
    otherwise start = end = the single date. Three-or-more-date extractions get
    collapsed to [min, max] — we surface the raw string in `event_dates_raw`
    so downstream validation can see what the LLM actually emitted.

    The new ``location_admin_levels`` field (added 2026-04-29) is a parallel
    pipe-separated list — one self-tag per location. Older JSONL files written
    before that change won't have it; we default each tag to "unknown" so
    downstream code never has to special-case the missing-field path.
    """
    with jsonl_path.open() as fh:
        for ln in fh:
            row = json.loads(ln)
            if not row.get("is_verifiable_event"):
                continue

            locs = _split_pipes(row.get("affected_locations"))
            if not locs:
                continue

            admin_levels = _split_pipes(row.get("location_admin_levels"))
            if len(admin_levels) < len(locs):
                admin_levels = admin_levels + ["unknown"] * (len(locs) - len(admin_levels))
            elif len(admin_levels) > len(locs):
                admin_levels = admin_levels[: len(locs)]

            event_dates_raw = (row.get("event_dates") or "").strip()
            dates = sorted(_split_pipes(event_dates_raw))
            if not dates:
                LOG.debug("skipping %s: no dates", row.get("asset_key"))
                continue
            start_date, end_date = dates[0], dates[-1]

            # Derive country_hint from the upstream `countries` field. We only
            # apply a hint when the document names exactly ONE country —
            # multi-country regional reports (e.g. "Philippines, United States
            # of America" for a Mindanao record) shouldn't bias geocoding to
            # one of them. ISO-3166 alpha-2 conversion via pycountry; without
            # it we fall back to no hint.
            countries_raw = (row.get("countries") or "").strip()
            country_hint = _country_hint_from_countries(countries_raw)

            for loc, lvl in zip(locs, admin_levels):
                yield {
                    "asset_key": row.get("asset_key"),
                    "title": row.get("title"),
                    "content_url": row.get("content_url"),
                    "hazard": row.get("event_hazard"),
                    "location_text": loc,
                    "extractor_admin_level": lvl,
                    "country_hint": country_hint,
                    "countries_raw": countries_raw,
                    "start_date": start_date,
                    "end_date": end_date,
                    "event_dates_raw": event_dates_raw,
                }


# ---------------------------------------------------------------------------
# UUID — deterministic, 32-char hex (matches Google Groundsource style)
# ---------------------------------------------------------------------------

def event_uuid(asset_key: str, location_text: str, start_date: str, end_date: str) -> str:
    raw = f"{asset_key}\x1f{location_text}\x1f{start_date}\x1f{end_date}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _coerce_polygon(geom):
    """Cast Polygon to MultiPolygon for schema consistency, leave others as-is."""
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom


def compute_area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Areas in EPSG:6933 (equal-area), converted to km^2."""
    return gdf.to_crs(EQUAL_AREA_EPSG).area / 1_000_000.0


# ---------------------------------------------------------------------------
# Confidence — a simple bounded score
# ---------------------------------------------------------------------------

def confidence_score(
    r: ResolutionResult,
    *,
    geometry_overshoot: bool = False,
) -> float:
    """
    Quick heuristic 0..1 score combining Nominatim importance, polygon-vs-bbox,
    and how many fallback steps were needed. Replace with a calibrated metric
    once you have enough manual-validation labels.

    The country-fallback step is treated as low-quality regardless of importance
    (a country polygon for a sub-country query is structurally an overshoot).
    Validation feedback on test_400 confirmed: "Thailand's Andaman coast"
    resolving to all of Thailand was rated geometry_correct=N, so any consumer
    applying a confidence cutoff should drop these.

    ``geometry_overshoot`` (added 2026-04-29) is the runner-computed flag for
    "the LLM emitted a bare country for a hazard whose footprint is inherently
    sub-national". It's a different signal from ``is_country_fallback`` —
    that one fires when the resolver's own ladder climbed up to a country;
    this one fires when the country was the original input. Both deserve the
    same low confidence cap.
    """
    if not r.resolved:
        return 0.0
    if getattr(r, "is_country_fallback", False) or geometry_overshoot:
        return 0.15
    base = float(r.importance or 0.4)
    if r.is_polygon:
        base += 0.25
    base -= 0.15 * r.used_fallback
    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Path to event_extractions.jsonl")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the geo-parquet")
    p.add_argument("--user-agent", required=True,
                   help="Descriptive UA for Nominatim, e.g. "
                        "'undrr-groundsource/0.1 (camilla@undrr.org)'")
    p.add_argument("--cache", type=Path, default=Path(".nominatim_cache.json"),
                   help="Local cache for Nominatim responses")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on rows for a quick test")
    p.add_argument("--coastline", type=Path, default=None,
                   help="Path to a Natural Earth coastline shapefile/zip "
                        "(e.g. data/ne_50m_coastline.zip). When supplied, "
                        "tsunami / storm-surge geometries are clipped to a "
                        "coastal strip. If omitted, those hazards pass through "
                        "unchanged and a warning is logged.")
    p.add_argument("--coastal-buffer-km", type=float, default=20.0,
                   help="Half-width of the coastal strip in km (default 20).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.input.exists():
        LOG.error("Input not found: %s", args.input)
        return 1

    LOG.info("Reading %s", args.input)
    rows = list(explode_extractions(args.input))
    if args.limit:
        rows = rows[: args.limit]
    LOG.info("Will resolve %d (asset, location) pairs", len(rows))
    if not rows:
        LOG.warning("Nothing to resolve.")
        return 0

    resolver = NominatimResolver(
        user_agent=args.user_agent,
        cache_path=args.cache,
    )

    if args.coastline is None:
        LOG.warning(
            "No --coastline supplied; tsunami/storm-surge geometries will not "
            "be clipped. To enable coastal clipping, download Natural Earth "
            "ne_50m_coastline.zip (helper: geometry_postprocess.download_coastline) "
            "and pass --coastline path/to/ne_50m_coastline.zip."
        )
    postprocessor = HazardAwarePostprocessor(
        coastline_path=args.coastline,
        coastal_buffer_km=args.coastal_buffer_km,
    )

    records: list[dict] = []
    geoms: list = []
    try:
        for i, row in enumerate(rows, 1):
            res = resolver.resolve(row["location_text"], country_hint=row["country_hint"])
            if not res.resolved:
                LOG.info("[%d/%d] UNRESOLVED %r (%s)", i, len(rows),
                         row["location_text"], res.error)
                continue

            # ---- country-overshoot guard --------------------------------
            # Two ways the resolved row can be "a whole country for a hazard
            # that doesn't affect a whole country":
            #   1) extractor self-tagged this location as 'country', AND the
            #      hazard is in our sub-national set. (Trusts the LLM's tag.)
            #   2) Nominatim returned admin_level == 'country' for a hazard in
            #      the sub-national set. (Catches cases where the LLM tag is
            #      missing/wrong but the geocoder admits it climbed too high.)
            extractor_lvl = (row.get("extractor_admin_level") or "").lower()
            is_sub_nat = is_sub_national_hazard(row.get("hazard"))
            geometry_overshoot = is_sub_nat and (
                extractor_lvl == "country"
                or res.admin_level == "country"
            )

            # ---- hazard-aware geometry post-processing ------------------
            pp = postprocessor.process(
                hazard=row.get("hazard"),
                geometry=res.geometry,
                location_text=row.get("location_text"),
            )
            final_geom = pp.geometry

            uid = event_uuid(
                row["asset_key"], row["location_text"],
                row["start_date"], row["end_date"],
            )
            records.append({
                "uuid": uid,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                # sidecar
                "asset_key": row["asset_key"],
                "title": row["title"],
                "content_url": row["content_url"],
                "hazard": row["hazard"],
                "location_text": row["location_text"],
                "extractor_admin_level": row.get("extractor_admin_level"),
                "country_hint": row["country_hint"],
                "countries_raw": row["countries_raw"],
                "resolution_method": "nominatim",
                "resolution_admin_level": res.admin_level,
                "resolution_geometry_source": res.geometry_source,
                "resolution_query": res.query,
                "resolution_used_fallback": res.used_fallback,
                "resolution_display_name": res.display_name,
                "resolution_country_fallback": getattr(res, "is_country_fallback", False),
                "geometry_overshoot": geometry_overshoot,
                "geometry_postprocess": pp.note,
                "geometry_pre_postprocess_area_km2": pp.original_area_km2,
                "resolution_confidence": confidence_score(
                    res, geometry_overshoot=geometry_overshoot
                ),
            })
            geoms.append(_coerce_polygon(final_geom))
            if i % 25 == 0:
                resolver.flush_cache()
                LOG.info("[%d/%d] resolved=%d", i, len(rows), len(records))
    finally:
        resolver.flush_cache()

    if not records:
        LOG.warning("No rows resolved.")
        return 0

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
    gdf["area_km2"] = compute_area_km2(gdf)

    # Reorder so the Google-schema columns lead.
    leading = ["uuid", "area_km2", "geometry", "start_date", "end_date"]
    rest = [c for c in gdf.columns if c not in leading]
    gdf = gdf[leading + rest]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.output, index=False)
    LOG.info("Wrote %d rows -> %s", len(gdf), args.output)
    LOG.info("  resolution rate: %d / %d (%.1f%%)",
             len(gdf), len(rows), 100.0 * len(gdf) / len(rows))
    LOG.info("  by admin_level: %s",
             gdf["resolution_admin_level"].value_counts().to_dict())
    if "geometry_overshoot" in gdf.columns:
        n_overshoot = int(gdf["geometry_overshoot"].sum())
        LOG.info("  geometry_overshoot=True: %d (capped to confidence 0.15)",
                 n_overshoot)
    if "geometry_postprocess" in gdf.columns:
        LOG.info("  postprocess actions: %s",
                 gdf["geometry_postprocess"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
