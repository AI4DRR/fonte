"""
groundsource.geometry_postprocess
-----------------------

Hazard-aware geometry post-processing for the polygon resolver.

Motivation
==========

For some hazards, the polygon Nominatim returns is structurally too coarse:

* Tsunamis hit a coastal strip, not a whole country. "India" -> 3.4M km^2 is
  wrong; the affected area is a ~10-50 km strip along the eastern seaboard.
* Storm surge has the same property.
* Nuclear / industrial accidents have a defined exclusion zone, not a country.

This module exposes two things:

1. ``CoastalBuffer`` — lazily builds (and caches on disk) a global "land
   within N km of the coast" geometry from a Natural Earth coastline shapefile.
   Provides ``clip(geom)`` to intersect any polygon with the coastal strip.

2. ``HazardAwarePostprocessor`` — small dispatcher that, given a hazard type
   and a resolved geometry, decides whether to clip / replace / leave alone.
   Currently wired for tsunami and storm surge (coastal clip). Hooks left in
   place for nuclear-exclusion-zone and cyclone-track lookups.

Usage
=====

    from groundsource.geometry_postprocess import HazardAwarePostprocessor

    pp = HazardAwarePostprocessor(coastline_path="data/ne_50m_coastline.zip")
    new_geom, note = pp.process(hazard="Tsunami", geometry=country_polygon)
    # note is one of: 'unchanged', 'coastal_clip', 'exclusion_zone', 'no_intersection'

The coastline shapefile is NOT downloaded automatically — see ``download_coastline``
helper at the bottom for a one-liner, or fetch it manually from
https://www.naturalearthdata.com/downloads/50m-physical-vectors/ .

Design notes
============

* Buffering happens in EPSG:6933 (NSIDC EASE-Grid 2.0 Global, equal-area)
  so the buffer distance is a real km figure, not degrees of lat/lon.
  This is the same projection used by ``run_polygon_resolution.compute_area_km2``.

* The global coastal strip is computed once (slow — minutes) and cached as a
  WKB blob next to the input shapefile. Subsequent runs load it in <1s.

* The clip is ``input.intersection(coastal_strip)``. If the input polygon does
  not touch the coast at all, the result is empty; in that case the
  postprocessor returns the original geometry plus a ``no_intersection`` note,
  so the caller can decide whether to drop or keep the row.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


LOG = logging.getLogger(__name__)


# Same equal-area CRS used elsewhere in the pipeline.
EQUAL_AREA_EPSG = 6933

# Hazard types — case-insensitive substring match — that should be clipped to
# the coastal strip. Kept conservative; tropical cyclones intentionally NOT
# included because they routinely cause inland wind/flood damage.
COASTAL_HAZARDS = (
    "tsunami",
    "storm surge",
)

# Hazard types that have known, narrow exclusion zones. Lookup keyed by a
# normalised location string -> (geometry, source_note). For v1 we ship
# Chernobyl + Fukushima; extend as needed. Geometries are circles around the
# facility centroid in EPSG:6933, then reprojected back to 4326 lazily.
NUCLEAR_EXCLUSION_ZONES_KM = {
    # name (lowercase, stripped) -> (lat, lon, radius_km)
    "chernobyl": (51.389, 30.099, 30.0),
    "chornobyl": (51.389, 30.099, 30.0),
    "pripyat": (51.405, 30.057, 30.0),
    "fukushima": (37.421, 141.033, 20.0),
    "fukushima daiichi": (37.421, 141.033, 20.0),
    "three mile island": (40.153, -76.725, 10.0),
}


# ---------------------------------------------------------------------------
# CoastalBuffer
# ---------------------------------------------------------------------------

class CoastalBuffer:
    """
    Lazily-built global "within N km of the coast" geometry.

    Parameters
    ----------
    coastline_path
        Path to a Natural Earth coastline file (shapefile inside a zip is fine,
        geopandas reads it directly). E.g. ``ne_50m_coastline.zip``.
    buffer_km
        Half-width of the coastal strip, in kilometres.
    cache_path
        Where to pickle the built MultiPolygon. Defaults to a sibling file of
        ``coastline_path`` named ``<stem>_coastal_<km>km.pkl``.
    """

    def __init__(
        self,
        coastline_path: Path,
        buffer_km: float = 20.0,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.coastline_path = Path(coastline_path)
        self.buffer_km = float(buffer_km)
        if cache_path is None:
            stem = self.coastline_path.stem
            cache_path = self.coastline_path.with_name(
                f"{stem}_coastal_{int(self.buffer_km)}km.pkl"
            )
        self.cache_path = Path(cache_path)
        self._geom: Optional[BaseGeometry] = None

    def _load_or_build(self) -> BaseGeometry:
        if self._geom is not None:
            return self._geom
        if self.cache_path.exists():
            LOG.info("Loading cached coastal buffer from %s", self.cache_path)
            with self.cache_path.open("rb") as fh:
                self._geom = pickle.load(fh)
            return self._geom

        if not self.coastline_path.exists():
            raise FileNotFoundError(
                f"Coastline source not found at {self.coastline_path}. "
                "Download Natural Earth 50m coastline (see download_coastline) "
                "or pass an alternate path."
            )

        LOG.info("Building coastal buffer (%.0f km) from %s — slow on first run",
                 self.buffer_km, self.coastline_path)
        # Imported here so callers that never use coastal clipping don't need
        # geopandas to load this module.
        import geopandas as gpd

        gdf = gpd.read_file(self.coastline_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf_eq = gdf.to_crs(EQUAL_AREA_EPSG)
        # buffer in metres, since EPSG:6933 is metric
        buf_m = self.buffer_km * 1000.0
        buffered = gdf_eq.geometry.buffer(buf_m)
        merged_eq = unary_union(buffered.values)
        merged_4326 = (
            gpd.GeoSeries([merged_eq], crs=EQUAL_AREA_EPSG)
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        self._geom = merged_4326

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as fh:
            pickle.dump(self._geom, fh, protocol=pickle.HIGHEST_PROTOCOL)
        LOG.info("Cached coastal buffer to %s", self.cache_path)
        return self._geom

    def clip(self, geom: BaseGeometry) -> Optional[BaseGeometry]:
        """
        Return geom intersected with the coastal strip.

        Returns None if the intersection is empty (geom is fully inland).
        Returns the intersection geometry otherwise — Polygon or MultiPolygon.
        """
        if geom is None or geom.is_empty:
            return None
        coast = self._load_or_build()
        clipped = geom.intersection(coast)
        if clipped.is_empty:
            return None
        # Normalise to (Multi)Polygon — intersections sometimes drop to lines
        # at coastline edges. Filter those out.
        if isinstance(clipped, (Polygon, MultiPolygon)):
            return clipped
        # GeometryCollection: keep only the polygonal parts
        polys = [g for g in getattr(clipped, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        return unary_union(polys)


# ---------------------------------------------------------------------------
# Nuclear exclusion zones
# ---------------------------------------------------------------------------

def _exclusion_zone_geometry(lat: float, lon: float, radius_km: float) -> Polygon:
    """Circle of given radius around (lat, lon) returned in EPSG:4326."""
    import geopandas as gpd
    from shapely.geometry import Point

    centre = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(EQUAL_AREA_EPSG)
    buf = centre.buffer(radius_km * 1000.0)
    return buf.to_crs("EPSG:4326").iloc[0]


def lookup_exclusion_zone(location_text: str) -> Optional[BaseGeometry]:
    """Return a curated exclusion-zone polygon for known nuclear/industrial
    sites, or None if the location is not in our table."""
    if not location_text:
        return None
    key = location_text.strip().lower()
    record = NUCLEAR_EXCLUSION_ZONES_KM.get(key)
    if record is None:
        # Try a loose substring match for things like "Chernobyl Nuclear Power Plant"
        for k, v in NUCLEAR_EXCLUSION_ZONES_KM.items():
            if k in key:
                record = v
                break
    if record is None:
        return None
    lat, lon, radius_km = record
    return _exclusion_zone_geometry(lat, lon, radius_km)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

@dataclass
class PostprocessResult:
    geometry: BaseGeometry
    note: str  # 'unchanged' | 'coastal_clip' | 'exclusion_zone' | 'no_intersection'
    original_area_km2: Optional[float] = None
    new_area_km2: Optional[float] = None


class HazardAwarePostprocessor:
    """
    Decides per (hazard, location) what — if anything — to do to the geometry.

    Lazy: the coastal buffer is only built the first time a coastal hazard is
    seen, so a corpus with no tsunamis pays nothing.

    Parameters
    ----------
    coastline_path
        Path to a Natural Earth coastline file. Required only if you want
        coastal clipping; pass None to disable that branch (rows pass through).
    coastal_buffer_km
        Half-width of the coastal strip, in km.
    """

    def __init__(
        self,
        coastline_path: Optional[Path] = None,
        coastal_buffer_km: float = 20.0,
    ) -> None:
        self._coastal: Optional[CoastalBuffer] = (
            CoastalBuffer(coastline_path, buffer_km=coastal_buffer_km)
            if coastline_path is not None
            else None
        )

    @staticmethod
    def _is_coastal_hazard(hazard: Optional[str]) -> bool:
        if not hazard:
            return False
        h = hazard.lower()
        return any(tag in h for tag in COASTAL_HAZARDS)

    @staticmethod
    def _is_nuclear_hazard(hazard: Optional[str]) -> bool:
        if not hazard:
            return False
        h = hazard.lower()
        return any(tag in h for tag in ("nuclear", "radiological", "radiation"))

    @staticmethod
    def _area_km2(geom: BaseGeometry) -> float:
        import geopandas as gpd
        return float(
            gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(EQUAL_AREA_EPSG).area.iloc[0]
            / 1_000_000.0
        )

    def process(
        self,
        hazard: Optional[str],
        geometry: BaseGeometry,
        location_text: Optional[str] = None,
    ) -> PostprocessResult:
        """
        Returns a PostprocessResult. The caller is responsible for replacing
        the geometry on the row if note != 'unchanged'.
        """
        if geometry is None or geometry.is_empty:
            return PostprocessResult(geometry=geometry, note="unchanged")

        # 1) Nuclear / radiological — known exclusion zone wins outright.
        if self._is_nuclear_hazard(hazard) and location_text:
            zone = lookup_exclusion_zone(location_text)
            if zone is not None:
                return PostprocessResult(
                    geometry=zone,
                    note="exclusion_zone",
                    original_area_km2=self._area_km2(geometry),
                    new_area_km2=self._area_km2(zone),
                )

        # 2) Coastal hazards — clip with coastal buffer.
        if self._is_coastal_hazard(hazard) and self._coastal is not None:
            clipped = self._coastal.clip(geometry)
            if clipped is None:
                # Polygon doesn't touch the coast — leave geometry alone but
                # flag for the caller. (E.g., a landlocked admin1 mistakenly
                # tagged as tsunami-affected.)
                return PostprocessResult(
                    geometry=geometry,
                    note="no_intersection",
                    original_area_km2=self._area_km2(geometry),
                )
            return PostprocessResult(
                geometry=clipped,
                note="coastal_clip",
                original_area_km2=self._area_km2(geometry),
                new_area_km2=self._area_km2(clipped),
            )

        return PostprocessResult(geometry=geometry, note="unchanged")


# ---------------------------------------------------------------------------
# Convenience: download coastline (one-liner)
# ---------------------------------------------------------------------------

NE_COASTLINE_URL = (
    "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"
)


def download_coastline(dest: Path) -> Path:
    """Download Natural Earth 50m coastline to ``dest`` (a .zip path)."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        LOG.info("Coastline already present at %s", dest)
        return dest
    LOG.info("Downloading %s -> %s", NE_COASTLINE_URL, dest)
    urllib.request.urlretrieve(NE_COASTLINE_URL, dest)
    return dest


# ---------------------------------------------------------------------------
# Sub-national hazard set used by the country-overshoot guard in the runner.
# Mirrors (intentionally) the prompt list in groundsource.prompts.
# Centralised here so any consumer (runner, validation tool, dashboard) can
# import the canonical set instead of duplicating it.
# ---------------------------------------------------------------------------

SUB_NATIONAL_HAZARDS = frozenset({
    "tsunami",
    "earthquake",
    "volcanic eruption",
    "volcanic activity",
    "technological hazard",
    "nuclear accident",
    "industrial accident",
    "chemical accident",
    "oil spill",
    "landslide",
    "mudslide",
    "avalanche",
    "tornado",
    "wildfire",
    "flash flood",
    "dam burst",
    "storm surge",
    "transport accident",
})


def is_sub_national_hazard(hazard: Optional[str]) -> bool:
    if not hazard:
        return False
    h = hazard.strip().lower()
    if h in SUB_NATIONAL_HAZARDS:
        return True
    # tolerate substring (e.g. "Volcanic eruption (Mount Etna)")
    return any(tag in h for tag in SUB_NATIONAL_HAZARDS)
