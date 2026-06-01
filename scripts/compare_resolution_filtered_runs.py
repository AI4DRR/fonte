import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


EVENT_KEY_FIELDS = [
    "asset_key",
    "hazard",
    "location_text",
    "start_date",
    "end_date",
]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter prompt geometry outputs by resolution_confidence, build a "
            "deduplicated prompt01-04 union layer, and compare it with a unified prompt."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs"),
        help="Root used with --baseline-pattern to discover baseline prompt parquet files.",
    )
    parser.add_argument(
        "--baseline-pattern",
        default="test_400_gpt5_medium_prompt0[1-4]/event_geometries.parquet",
        help="Glob pattern under --baseline-root for prompt01-04 geometry files.",
    )
    parser.add_argument(
        "--unified",
        type=Path,
        required=True,
        help="Unified prompt geometry parquet file, usually prompt05.",
    )
    parser.add_argument(
        "--min-resolution-confidence",
        type=float,
        default=0.7,
        help="Inclusive lower resolution_confidence bound.",
    )
    parser.add_argument(
        "--max-resolution-confidence",
        type=float,
        default=0.98,
        help="Inclusive upper resolution_confidence bound.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where comparison tables and map layers are written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def load_prompt_names() -> dict[str, str]:
    try:
        from groundsource.prompts import PROMPTS
    except Exception as exc:  # pragma: no cover - optional convenience only
        logging.warning("Could not import prompt names from groundsource.prompts: %s", exc)
        return {}

    return {f"prompt{prompt.id:02d}": prompt.name for prompt in PROMPTS}


def discover_baselines(root: Path, pattern: str) -> list[Path]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise SystemExit(f"No baseline geometry files found under {root} with {pattern!r}")
    return paths


def prompt_label_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.search(r"prompt(\d+)", part)
        if match:
            return f"prompt{int(match.group(1)):02d}"
    return path.parent.name


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def norm_text(value) -> str:
    return clean_text(value).lower()


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def load_layer(path: Path, prompt_names: dict[str, str]) -> gpd.GeoDataFrame:
    label = prompt_label_from_path(path)
    gdf = gpd.read_parquet(path)
    if gdf.empty:
        logging.warning("Loaded empty geometry file: %s", path)
        return gdf

    gdf = ensure_wgs84(gdf).copy()
    gdf["prompt_label"] = label
    gdf["prompt_name"] = prompt_names.get(label, "")
    gdf["source_path"] = str(path)
    gdf["source_prompts"] = label
    gdf["source_prompt_count"] = 1
    return add_event_keys(gdf)


def add_event_keys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for field in EVENT_KEY_FIELDS:
        if field not in gdf.columns:
            gdf[field] = ""

    gdf["asset_key_norm"] = gdf["asset_key"].map(norm_text)
    gdf["hazard_norm"] = gdf["hazard"].map(norm_text)
    gdf["location_text_norm"] = gdf["location_text"].map(norm_text)
    gdf["start_date_norm"] = gdf["start_date"].map(norm_text)
    gdf["end_date_norm"] = gdf["end_date"].map(norm_text)
    gdf["event_location_key"] = (
        gdf["asset_key_norm"]
        + "||"
        + gdf["hazard_norm"]
        + "||"
        + gdf["location_text_norm"]
        + "||"
        + gdf["start_date_norm"]
        + "||"
        + gdf["end_date_norm"]
    )
    return gdf


def filter_confidence(
    gdf: gpd.GeoDataFrame,
    min_confidence: float,
    max_confidence: float,
) -> gpd.GeoDataFrame:
    if "resolution_confidence" not in gdf.columns:
        raise ValueError("Missing required column: resolution_confidence")

    confidence = pd.to_numeric(gdf["resolution_confidence"], errors="coerce")
    filtered = gdf[confidence.between(min_confidence, max_confidence, inclusive="both")].copy()
    filtered["resolution_confidence"] = pd.to_numeric(
        filtered["resolution_confidence"], errors="coerce"
    )
    return filtered


def sorted_prompts(values: Iterable[str]) -> list[str]:
    return sorted({clean_text(value) for value in values if clean_text(value)})


def build_union_layer(baseline_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if baseline_gdf.empty:
        return baseline_gdf.copy()

    rows = []
    for _, group in baseline_gdf.groupby("event_location_key", sort=True):
        best_idx = group["resolution_confidence"].astype(float).idxmax()
        row = group.loc[best_idx].copy()
        prompts = sorted_prompts(group["prompt_label"])
        row["prompt_label"] = "prompt01-04_union"
        row["prompt_name"] = "prompt01-04_union"
        row["source_prompts"] = " | ".join(prompts)
        row["source_prompt_count"] = len(prompts)
        rows.append(row)

    union = gpd.GeoDataFrame(rows, geometry=baseline_gdf.geometry.name, crs=baseline_gdf.crs)
    return union.reset_index(drop=True)


def write_layer(gdf: gpd.GeoDataFrame, label: str, output_dir: Path) -> Path:
    layer_dir = output_dir / "map_layers" / label
    layer_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = layer_dir / "event_geometries.parquet"
    gdf.to_parquet(parquet_path, index=False)

    if not gdf.empty:
        gdf.drop(columns=[gdf.geometry.name]).to_csv(
            layer_dir / "event_geometries.csv",
            index=False,
        )
    logging.info("Wrote %s (%d rows)", parquet_path, len(gdf))
    return parquet_path


def selection_summary_row(
    label: str,
    source_path: str,
    source_rows: int,
    selected: gpd.GeoDataFrame,
) -> dict[str, object]:
    return {
        "prompt_label": label,
        "source_path": source_path,
        "source_geometry_rows": source_rows,
        "selected_geometry_rows": int(len(selected)),
        "selected_unique_assets": int(selected["asset_key"].nunique()) if len(selected) else 0,
        "min_resolution_confidence": round(float(selected["resolution_confidence"].min()), 6)
        if len(selected)
        else "",
        "max_resolution_confidence": round(float(selected["resolution_confidence"].max()), 6)
        if len(selected)
        else "",
        "avg_resolution_confidence": round(float(selected["resolution_confidence"].mean()), 6)
        if len(selected)
        else "",
    }


def representative_rows(gdf: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    if gdf.empty:
        return pd.DataFrame(columns=["event_location_key"])

    rows = []
    for key, group in gdf.groupby("event_location_key", sort=True):
        best_idx = group["resolution_confidence"].astype(float).idxmax()
        row = group.loc[best_idx]
        rows.append(
            {
                "event_location_key": key,
                f"{prefix}_asset_key": row.get("asset_key", ""),
                f"{prefix}_title": row.get("title", ""),
                f"{prefix}_hazard": row.get("hazard", ""),
                f"{prefix}_location_text": row.get("location_text", ""),
                f"{prefix}_start_date": row.get("start_date", ""),
                f"{prefix}_end_date": row.get("end_date", ""),
                f"{prefix}_resolution_confidence": row.get("resolution_confidence", ""),
                f"{prefix}_resolution_admin_level": row.get("resolution_admin_level", ""),
                f"{prefix}_source_prompts": row.get("source_prompts", ""),
                f"{prefix}_content_url": row.get("content_url", ""),
            }
        )
    return pd.DataFrame(rows)


def build_overlap_tables(
    baseline_union: gpd.GeoDataFrame,
    unified: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_keys = set(baseline_union["event_location_key"]) if len(baseline_union) else set()
    unified_keys = set(unified["event_location_key"]) if len(unified) else set()
    all_keys = sorted(baseline_keys | unified_keys)

    detail = pd.DataFrame({"event_location_key": all_keys})
    detail["in_four_prompt_union"] = detail["event_location_key"].isin(baseline_keys)
    detail["in_unified"] = detail["event_location_key"].isin(unified_keys)
    detail["comparison_status"] = detail.apply(
        lambda row: "both"
        if row["in_four_prompt_union"] and row["in_unified"]
        else "four_prompt_union_only"
        if row["in_four_prompt_union"]
        else "unified_only",
        axis=1,
    )

    baseline_reps = representative_rows(baseline_union, "four_prompt_union")
    unified_reps = representative_rows(unified, "unified")
    detail = detail.merge(baseline_reps, on="event_location_key", how="left")
    detail = detail.merge(unified_reps, on="event_location_key", how="left")

    both = len(baseline_keys & unified_keys)
    union_total = len(baseline_keys | unified_keys)
    summary = pd.DataFrame(
        [
            {
                "four_prompt_union_count": len(baseline_keys),
                "unified_count": len(unified_keys),
                "both": both,
                "unified_only": len(unified_keys - baseline_keys),
                "four_prompt_union_only": len(baseline_keys - unified_keys),
                "all_unique_event_locations": union_total,
                "jaccard_overlap": round((both / union_total) if union_total else 0.0, 6),
            }
        ]
    )
    return detail, summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.min_resolution_confidence > args.max_resolution_confidence:
        raise SystemExit("--min-resolution-confidence must be <= --max-resolution-confidence")
    if not args.unified.exists():
        raise SystemExit(f"Unified geometry file not found: {args.unified}")

    prompt_names = load_prompt_names()
    baseline_paths = discover_baselines(args.baseline_root, args.baseline_pattern)
    layers: dict[str, gpd.GeoDataFrame] = {}
    selection_rows: list[dict[str, object]] = []

    for path in baseline_paths:
        gdf = load_layer(path, prompt_names)
        label = prompt_label_from_path(path)
        filtered = filter_confidence(
            gdf,
            args.min_resolution_confidence,
            args.max_resolution_confidence,
        )
        layers[label] = filtered
        write_layer(filtered, label, args.output_dir)
        selection_rows.append(selection_summary_row(label, str(path), len(gdf), filtered))

    unified_gdf = load_layer(args.unified, prompt_names)
    unified_label = prompt_label_from_path(args.unified)
    unified_filtered = filter_confidence(
        unified_gdf,
        args.min_resolution_confidence,
        args.max_resolution_confidence,
    )
    layers[unified_label] = unified_filtered
    write_layer(unified_filtered, unified_label, args.output_dir)
    selection_rows.append(
        selection_summary_row(unified_label, str(args.unified), len(unified_gdf), unified_filtered)
    )

    baseline_filtered = pd.concat(
        [gdf for label, gdf in layers.items() if label != unified_label],
        ignore_index=True,
    )
    baseline_filtered = gpd.GeoDataFrame(
        baseline_filtered,
        geometry=unified_filtered.geometry.name if not unified_filtered.empty else "geometry",
        crs="EPSG:4326",
    )
    baseline_union = build_union_layer(baseline_filtered)
    write_layer(baseline_union, "prompt01-04_union", args.output_dir)
    selection_rows.append(
        selection_summary_row(
            "prompt01-04_union",
            "deduplicated prompt01-04 filtered rows",
            len(baseline_filtered),
            baseline_union,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_summary = pd.DataFrame(selection_rows)
    selection_summary.to_csv(args.output_dir / "selection_summary.csv", index=False)

    overlap_detail, overlap_summary = build_overlap_tables(baseline_union, unified_filtered)
    overlap_detail.to_csv(args.output_dir / "retrieval_overlap_by_event_location.csv", index=False)
    overlap_summary.to_csv(args.output_dir / "retrieval_overlap_summary.csv", index=False)

    logging.info("Wrote %s", args.output_dir / "selection_summary.csv")
    logging.info("Wrote %s", args.output_dir / "retrieval_overlap_by_event_location.csv")
    logging.info("Wrote %s", args.output_dir / "retrieval_overlap_summary.csv")
    logging.info("Overlap summary: %s", overlap_summary.to_dict(orient="records")[0])


if __name__ == "__main__":
    main()
