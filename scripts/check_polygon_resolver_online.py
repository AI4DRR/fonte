"""
check_polygon_resolver_online.py
------------------------

Lightweight self-test for the polygon-resolver prototype. Hits the public
Nominatim endpoint, so it'll only run online and respects the ~1 req/sec
rate limit (the resolver enforces this internally).

Strings below come from the actual extractions in
outputs/test_400/event_extractions.jsonl, including the messy ones (sub-city
qualifiers, geological features, multi-comma fragments).

Run:
    python scripts/check_polygon_resolver_online.py --user-agent "undrr-test/0.1 (you@undrr.org)"
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from groundsource.polygon_resolver import NominatimResolver, fallback_ladder

# Mix of granularities + difficulty levels. (resolved, tag) — `resolved` is
# the expected outcome under the v2 ladder (possessive + facility rewrites).
SAMPLES = [
    ("Maldives",                                       True,  "country"),
    ("Brisbane",                                       True,  "city"),
    ("Port-au-Prince",                                 True,  "city"),
    ("Beirut seaport",                                 True,  "facility-noun rewrite"),
    ("Thailand's Andaman coast",                       True,  "possessive + country hint"),
    ("Miyagi Prefecture",                              True,  "admin1"),
    ("Mindanao",                                       True,  "region"),
    ("Hin Look Dieu, Phuket",                          True,  "settlement-fallback"),
    ("Villa N°6, Cildañez Stream Basin, Buenos Aires", True,  "messy-fallback"),
    ("surrounding areas of Port-au-Prince",            True,  "qualifier-fallback"),
    ("Hellenic Subduction Zone",                       False, "non-admin / out of OSM scope"),
    ("Warehouse Number 12",                            False, "sub-building / extractor's job"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--user-agent", required=True,
                   help="Descriptive UA, e.g. 'undrr-test/0.1 (you@undrr.org)'")
    p.add_argument("--cache", type=Path,
                   default=Path(tempfile.gettempdir()) / "polygon_resolver_test_cache.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    resolver = NominatimResolver(user_agent=args.user_agent, cache_path=args.cache)

    width = max(len(s) for s, _, _ in SAMPLES)
    n_resolved = 0
    regressions: list[str] = []
    print(f"\n{'INPUT':<{width}}  EXPECT  TAG                              RESULT")
    print("-" * (width + 80))
    for text, expected, tag in SAMPLES:
        print(f"  fallback ladder: {fallback_ladder(text)}")
        res = resolver.resolve(text)
        got = res.resolved
        match = "OK" if got == expected else "FAIL"
        if got != expected:
            regressions.append(text)
        if res.resolved:
            n_resolved += 1
            kind = "POLY" if res.is_polygon else "BBOX"
            outcome = (f"OK  [{kind} / admin={res.admin_level} / "
                       f"fb={res.used_fallback}] -> "
                       f"{(res.display_name or '')[:80]}")
        else:
            outcome = f"MISS  ({res.error})"
        print(f"{text:<{width}}  {str(expected):<6}  {tag:<32}  [{match}] {outcome}")
        print()

    resolver.flush_cache()
    print(f"\nResolved {n_resolved}/{len(SAMPLES)}  "
          f"(cache: {args.cache})\n")
    if regressions:
        print(f"FAIL: outcome did not match expectation for: {regressions}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
