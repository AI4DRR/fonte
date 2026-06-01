import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an event geometries parquet by resolution_confidence range."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input geometries parquet.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output parquet.")
    parser.add_argument("--min", type=float, default=0.7, help="Minimum resolution_confidence (inclusive).")
    parser.add_argument("--max", type=float, default=0.98, help="Maximum resolution_confidence (inclusive).")
    args = parser.parse_args()

    gdf = gpd.read_parquet(args.input)
    gdf["resolution_confidence"] = pd.to_numeric(gdf["resolution_confidence"], errors="coerce")
    mask = (gdf["resolution_confidence"] >= args.min) & (gdf["resolution_confidence"] <= args.max)
    filtered = gdf[mask].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output)
    print(f"{len(gdf)} -> {len(filtered)} events (confidence {args.min}-{args.max})")


if __name__ == "__main__":
    main()
