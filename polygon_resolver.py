"""
polygon_resolver.py
-------------------

Resolve free-text location strings (as produced by app_event_focus.py's
`affected_locations` field) into geometries that match Google Groundsource's
output schema:

    uuid (str, 32-char hex)
    area_km2 (float)
    geometry (WKB Polygon / MultiPolygon, EPSG:4326)
    start_date (str, YYYY-MM-DD)
    end_date (str, YYYY-MM-DD)

Design notes
============

* For the prototype we use the public Nominatim (OpenStreetMap) endpoint as
  the geocoder. Nominatim returns a polygon when one exists in OSM, otherwise
  a point + bounding box. By default the resolver only accepts true OSM
  polygons — bbox / point-buffer rectangles are rejected per ladder step
  (so the ladder keeps descending) to avoid emitting axis-aligned generic
  shapes for tiny named places that happen to be stored as OSM nodes. Pass
  ``allow_non_polygon=True`` to NominatimResolver if you want those shapes
  back as a last resort.

* Nominatim's public endpoint requires a descriptive User-Agent and is rate
  limited to roughly 1 request per second. The `_throttle` helper enforces
  this. For UNDRR's full corpus you should swap `NominatimResolver` for
  either a self-hosted Nominatim instance or a local GeoNames + GADM lookup
  that implements the same `resolve(text, country_hint=None)` contract.

* Results are cached on disk (JSON file keyed by query) so reruns are cheap.

* Fallback ladder for messy strings:
    1) original text
    2) possessive rewrite ("Thailand's Andaman coast" -> "Andaman coast", country=TH)
    3) trailing facility noun rewrite ("Beirut seaport" -> ["Port of Beirut", "Beirut"])
    4) trailing comma fragment ("Hin Look Dieu, Phuket" -> "Phuket")
    5) leading qualifier dropped ("surrounding areas of Port-au-Prince" -> "Port-au-Prince")
    6) last resort: if a possessive was matched, the owner alone as a
       country-level fallback ("Thailand's Andaman coast" -> "Thailand")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from shapely.geometry import Polygon, MultiPolygon, shape, box, mapping
from shapely.geometry.base import BaseGeometry


LOG = logging.getLogger(__name__)


# pycountry is used to translate "Thailand" -> "TH" for the possessive
# rewrite ("Thailand's Andaman coast" -> query "Andaman coast", hint TH).
# It's a soft dep — without it, the possessive rewrite still fires but
# without the country narrowing.
try:
    import pycountry  # type: ignore
    _HAS_PYCOUNTRY = True
except ImportError:
    _HAS_PYCOUNTRY = False
    LOG.warning("pycountry not installed — possessive rewrites will run "
                "without country narrowing. `pip install pycountry` to enable.")


def _country_code(name: str) -> Optional[str]:
    """Map a country name to ISO-3166 alpha-2 (lowercase). None if unknown."""
    if not _HAS_PYCOUNTRY or not name:
        return None
    try:
        c = pycountry.countries.lookup(name.strip())
        return c.alpha_2.lower()
    except LookupError:
        return None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    """One resolved (or unresolved) location."""

    query: str                     # what we sent to the geocoder
    original: str                  # the original raw text
    geometry: Optional[BaseGeometry]
    admin_level: Optional[str]     # e.g. "country", "state", "city", "neighbourhood"
    osm_type: Optional[str]        # "relation" | "way" | "node"
    display_name: Optional[str]
    importance: Optional[float]    # Nominatim importance score, 0..1
    used_fallback: int = 0         # 0 = direct hit, 1+ = which fallback
    geometry_source: Optional[str] = None  # "polygon" | "bbox" | "point_buffer"
    is_country_fallback: bool = False  # True if last-resort owner-only step won
    error: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.geometry is not None

    @property
    def is_polygon(self) -> bool:
        return isinstance(self.geometry, (Polygon, MultiPolygon))


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

_LEADING_QUALIFIERS = [
    r"surrounding areas of ",
    r"areas around ",
    r"area around ",
    r"the wider ",
    r"central regions of ",
    r"central ",
    r"northern ",
    r"southern ",
    r"eastern ",
    r"western ",
    r"near ",
    r"close to ",
    r"around ",
]

_QUALIFIER_RE = re.compile(
    r"^\s*(?:" + "|".join(_LEADING_QUALIFIERS) + r")\s*",
    re.IGNORECASE,
)

# Trailing facility nouns that block a name match in OSM
# (e.g., "Beirut seaport" — OSM has "Port of Beirut", not "Beirut seaport").
# Ordered longest-first so the regex prefers multi-word matches like
# "railway station" over the bare "station".
_FACILITY_NOUNS = [
    "international airport",
    "railway station",
    "train station",
    "metro station",
    "bus station",
    "underground station",
    "air base",
    "seaport",
    "harbour",
    "harbor",
    "airport",
    "airfield",
    "airbase",
    "terminal",
    "station",
    "docks",
    "dock",
    "port",
]
_TRAILING_FACILITY_RE = re.compile(
    r"^(.+?)\s+(" + "|".join(_FACILITY_NOUNS) + r")\s*$",
    re.IGNORECASE,
)
# These rewrite to "Port of X" (canonical OSM name for water-port facilities).
_PORT_LIKE = {"port", "seaport", "harbour", "harbor", "dock", "docks"}

# Possessive prefix at the start of the string, e.g. "Thailand's Andaman coast".
# Limited to a 1-3-word, capitalised possessor so we don't strip things like
# "the country's coast".
_POSSESSIVE_RE = re.compile(
    r"^([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,2})'s\s+(.+)$"
)


def _strip_leading_qualifier(text: str) -> str:
    """`surrounding areas of Port-au-Prince` -> `Port-au-Prince`."""
    return _QUALIFIER_RE.sub("", text).strip()


def _trailing_fragment(text: str) -> str:
    """`Hin Look Dieu, Phuket` -> `Phuket`. Returns text unchanged if no comma."""
    if "," not in text:
        return text
    return text.rsplit(",", 1)[-1].strip()


def _possessive_rewrite(text: str) -> Optional[tuple[str, Optional[str], str]]:
    """
    `Thailand's Andaman coast` -> ("Andaman coast", "th", "Thailand").
    Returns (feature, country_code_or_None, owner_name). None if no possessive
    matched. The country code may be None even on a match (pycountry not
    installed, or possessor not a recognised country).
    `owner_name` is the original possessor string and is used by the ladder as
    a last-resort coarse-grained fallback when the more specific feature query
    fails — e.g., "Andaman coast" isn't an OSM entity, but "Thailand" is.
    """
    m = _POSSESSIVE_RE.match(text.strip())
    if not m:
        return None
    owner, feature = m.group(1).strip(), m.group(2).strip()
    if not feature:
        return None
    return feature, _country_code(owner), owner


def _facility_rewrites(text: str) -> list[str]:
    """`Beirut seaport` -> [`Port of Beirut`, `Beirut`]. Empty if no trailing
    facility noun. For non-port facility nouns, just returns the bare prefix."""
    m = _TRAILING_FACILITY_RE.match(text.strip())
    if not m:
        return []
    prefix = m.group(1).strip()
    noun = m.group(2).lower()
    out: list[str] = []
    if noun in _PORT_LIKE:
        out.append(f"Port of {prefix}")
    out.append(prefix)
    return out


def fallback_ladder(text: str) -> list[tuple[str, Optional[str], bool]]:
    """
    Ordered list of (query, country_override, is_country_fallback_step)
    triples to try, progressively coarser. `country_override` is None
    unless a rewrite has supplied a stronger hint than the user passed in
    — currently only the possessive rewrite does (e.g., "Thailand's
    Andaman coast" yields country='th'). `is_country_fallback_step` is
    True only on the synthesised owner-only step (last-resort coarse
    geometry); it is set explicitly here, not inferred from list index,
    so callers don't need to second-guess what dedup did.

    The original text always comes first so already-cached hits stay hot
    when the ladder is extended over time.
    """
    candidates: list[tuple[str, Optional[str], bool]] = []
    seen: set[tuple[str, str]] = set()

    def add(
        q: Optional[str],
        hint: Optional[str] = None,
        is_country_fallback: bool = False,
    ) -> None:
        if not q:
            return
        q = re.sub(r"\s+", " ", q).strip()
        key = (q.lower(), (hint or "").lower())
        if q and key not in seen:
            seen.add(key)
            candidates.append((q, hint, is_country_fallback))

    add(text)
    poss = _possessive_rewrite(text)
    if poss is not None:
        feature, hint, _owner = poss
        add(feature, hint)
    for r in _facility_rewrites(text):
        add(r)
    add(_trailing_fragment(text))
    add(_strip_leading_qualifier(text))
    add(_strip_leading_qualifier(_trailing_fragment(text)))

    # Last resort: if the input had a possessive owner, try the owner alone
    # (country-level fallback). Coarse, but better than dropping the record;
    # downstream consumers can filter on resolution_admin_level=='country'
    # plus a non-country location_text to flag this case. The flag is set
    # on this entry so the resolver doesn't have to detect it positionally
    # — dedup may push the owner string to an earlier slot if it coincides
    # with another rewrite, in which case this final add() is a no-op and
    # no step in the ladder is flagged (which is correct: the owner won
    # before we needed the country fallback).
    if poss is not None:
        _feature, _hint, owner = poss
        add(owner, is_country_fallback=True)
    return candidates


# ---------------------------------------------------------------------------
# Nominatim resolver
# ---------------------------------------------------------------------------

class NominatimResolver:
    """
    Resolve location text via the public Nominatim endpoint.

    Parameters
    ----------
    user_agent
        Descriptive UA per Nominatim's usage policy. Required.
    cache_path
        On-disk JSON cache so reruns are free. Set to None to disable.
    min_interval_s
        Minimum seconds between requests (Nominatim asks for >= 1.0).
    endpoint
        Override for a self-hosted Nominatim instance.
    """

    DEFAULT_ENDPOINT = "https://nominatim.openstreetmap.org/search"

    # Map OSM `addresstype` -> coarse admin label we'll expose to downstream.
    _ADMIN_LEVEL_MAP = {
        "country": "country",
        "state": "admin1",
        "region": "admin1",
        "province": "admin1",
        "county": "admin2",
        "district": "admin2",
        "municipality": "admin2",
        "city": "city",
        "town": "city",
        "village": "city",
        "hamlet": "city",
        "suburb": "neighbourhood",
        "neighbourhood": "neighbourhood",
        "quarter": "neighbourhood",
    }

    def __init__(
        self,
        user_agent: str,
        cache_path: Optional[Path] = None,
        min_interval_s: float = 1.1,
        endpoint: Optional[str] = None,
        timeout_s: float = 30.0,
        allow_non_polygon: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        allow_non_polygon
            When False (default), only OSM hits whose geojson is a true
            (Multi)Polygon are accepted. Bounding-box and centroid-buffer
            fallbacks are rejected per ladder step, so the ladder keeps
            descending instead of returning an axis-aligned rectangle.
            This avoids "generic square polygon" outputs for tiny named
            places (sub-villages, hamlets stored as OSM nodes) — they
            either resolve to their parent admin polygon or stay
            unresolved. Set True only if you specifically need bbox/point
            geometries (e.g., for visualization placeholders); the
            geometry_source field tells you which kind you got.
        """
        if not user_agent or "example" in user_agent.lower():
            raise ValueError(
                "Provide a real, descriptive User-Agent (e.g. "
                "'undrr-groundsource/0.1 (camilla@undrr.org)') per Nominatim policy."
            )
        self.user_agent = user_agent
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.allow_non_polygon = allow_non_polygon
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except Exception as e:
                LOG.warning("Could not read cache %s: %s", self.cache_path, e)
        self._last_request_at = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = self.user_agent

    # -- public API ---------------------------------------------------------

    def resolve(
        self, text: str, country_hint: Optional[str] = None
    ) -> ResolutionResult:
        """Resolve `text` to a geometry, walking the fallback ladder."""
        if not text or not text.strip():
            return ResolutionResult(
                query="", original=text, geometry=None, admin_level=None,
                osm_type=None, display_name=None, importance=None,
                error="empty input",
            )

        last_err: Optional[str] = None
        ladder = fallback_ladder(text)

        for i, (q, override, is_country_fallback_step) in enumerate(ladder):
            # Per-step override (from possessive rewrite) wins; otherwise the
            # caller-supplied hint applies to every step.
            effective_hint = override if override is not None else country_hint
            try:
                hit = self._search_one(q, country_hint=effective_hint)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                LOG.warning("Nominatim error for %r: %s", q, last_err)
                continue
            if hit is None:
                continue
            geom_pair = self._geometry_from_hit(hit)
            if geom_pair is None:
                continue
            geom, source = geom_pair
            if source != "polygon" and not self.allow_non_polygon:
                # Reject bbox / point_buffer rectangles — these are
                # axis-aligned generic shapes, not real footprints. Walk
                # to the next ladder step instead.
                LOG.debug(
                    "Skipping non-polygon hit for %r (source=%s); "
                    "continuing fallback ladder.", q, source,
                )
                continue
            return ResolutionResult(
                query=q,
                original=text,
                geometry=geom,
                admin_level=self._admin_level_from_hit(hit),
                osm_type=hit.get("osm_type"),
                display_name=hit.get("display_name"),
                importance=hit.get("importance"),
                used_fallback=i,
                geometry_source=source,
                is_country_fallback=is_country_fallback_step,
            )

        return ResolutionResult(
            query=text, original=text, geometry=None, admin_level=None,
            osm_type=None, display_name=None, importance=None,
            error=last_err or "no match",
        )

    def flush_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))

    # -- internals ----------------------------------------------------------

    # Bumped when the request shape changes (e.g., adding accept-language=en).
    # Cache entries written under a different version are simply ignored, not
    # incorrectly reused — old entries can be left in the file and are
    # harmless.
    _CACHE_VERSION = 2

    def _cache_key(self, q: str, country_hint: Optional[str]) -> str:
        return json.dumps({
            "v": self._CACHE_VERSION,
            "q": q.lower().strip(),
            "c": (country_hint or "").lower(),
        })

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request_at = time.monotonic()

    def _search_one(self, q: str, country_hint: Optional[str]) -> Optional[dict]:
        key = self._cache_key(q, country_hint)
        if key in self._cache:
            cached = self._cache[key]
            return cached if cached else None  # cached miss = falsy

        params = {
            "q": q,
            "format": "jsonv2",
            "polygon_geojson": 1,
            "limit": 1,
            "addressdetails": 1,
            # Ask Nominatim for English display names where available — without
            # this we get the local-language name (e.g. Arabic "المرفأ" for the
            # Port of Beirut), which is technically correct but unhelpful for
            # English-speaking reviewers and for downstream filtering.
            "accept-language": "en",
        }
        if country_hint:
            params["countrycodes"] = country_hint.lower()

        self._throttle()
        resp = self._session.get(self.endpoint, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        results = resp.json() or []
        hit = results[0] if results else None
        # cache positive AND negative results
        self._cache[key] = hit or {}
        return hit

    def _geometry_from_hit(self, hit: dict):
        """
        Prefer GeoJSON polygon; fall back to bbox; last resort: point buffer.
        Returns (geometry, source) where source is "polygon" | "bbox" | "point_buffer",
        or None if nothing usable is in the hit.
        """
        gj = hit.get("geojson")
        if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
            try:
                geom = shape(gj)
                if geom.is_valid and not geom.is_empty:
                    return geom, "polygon"
            except Exception:
                pass

        bb = hit.get("boundingbox")  # [south, north, west, east] as strings
        if bb and len(bb) == 4:
            try:
                south, north, west, east = (float(x) for x in bb)
                if north > south and east > west:
                    return box(west, south, east, north), "bbox"
            except Exception:
                pass

        # Point fallback: ±0.01° box around the centroid. Note this is a
        # square in DEGREES, not km — at the equator ~1.1 km on a side, but
        # at 60° latitude the east-west axis collapses to ~0.55 km, so the
        # box is elongated north-south. Rough placeholder, not a real
        # footprint; only emitted when allow_non_polygon=True. For a real
        # buffer use a population-aware radius reprojected via EPSG:6933.
        try:
            lat = float(hit["lat"])
            lon = float(hit["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        return box(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01), "point_buffer"

    def _admin_level_from_hit(self, hit: dict) -> Optional[str]:
        for key in ("addresstype", "type", "category"):
            v = hit.get(key)
            if isinstance(v, str) and v in self._ADMIN_LEVEL_MAP:
                return self._ADMIN_LEVEL_MAP[v]
        return "other"
