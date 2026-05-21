import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import Fullscreen


COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
]

POPUP_FIELDS = [
    "prompt_label",
    "prompt_name",
    "asset_key",
    "title",
    "hazard",
    "location_text",
    "start_date",
    "end_date",
    "area_km2",
    "resolution_admin_level",
    "resolution_confidence",
    "geometry_postprocess",
    "geometry_overshoot",
    "content_url",
]

TOOLTIP_FIELDS = [
    "prompt_label",
    "hazard",
    "location_text",
    "resolution_confidence",
]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a toggle-layer map comparing prompt polygon outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder used to search for prompt geometry parquet files.",
    )
    parser.add_argument(
        "--pattern",
        default="test_*_prompt*/event_geometries.parquet",
        help="Glob pattern under --input-root for prompt geometry parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prompt_polygon_comparison_map.html"),
        help="HTML map path to write.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV summary path. Defaults next to --output.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all prompt layers initially. By default only prompt01 is visible.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def load_prompt_names() -> Dict[int, str]:
    try:
        from groundsource.prompts import PROMPTS
    except Exception as exc:  # pragma: no cover - optional convenience only
        logging.warning("Could not import prompt names from groundsource.prompts: %s", exc)
        return {}
    return {prompt.id: prompt.name for prompt in PROMPTS}


def detect_prompt_id(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = re.search(r"prompt(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def prompt_label(prompt_id: Optional[int], path: Path) -> str:
    if prompt_id is not None:
        return f"prompt{prompt_id:02d}"
    return path.parent.name


def discover_inputs(input_root: Path, pattern: str) -> List[Path]:
    paths = sorted(input_root.glob(pattern))
    return [path for path in paths if path.is_file()]


def load_prompt_gdf(path: Path, prompt_names: Dict[int, str]) -> gpd.GeoDataFrame:
    prompt_id = detect_prompt_id(path)
    label = prompt_label(prompt_id, path)
    name = prompt_names.get(prompt_id, "") if prompt_id is not None else ""

    gdf = gpd.read_parquet(path)
    if gdf.empty:
        logging.warning("Skipping empty geometry file: %s", path)
        return gdf

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.copy()
    gdf["prompt_id"] = prompt_id if prompt_id is not None else ""
    gdf["prompt_label"] = label
    gdf["prompt_name"] = name
    gdf["source_path"] = str(path)
    return gdf


def clean_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        if pd.api.types.is_bool_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)
        elif pd.api.types.is_numeric_dtype(gdf[col]):
            continue
        else:
            gdf[col] = gdf[col].fillna("").astype(str)

    for col in ["area_km2", "resolution_confidence"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").round(4)

    return gdf


def build_summary(gdfs: List[gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []
    for gdf in gdfs:
        if gdf.empty:
            continue
        label = str(gdf["prompt_label"].iloc[0])
        name = str(gdf["prompt_name"].iloc[0])
        rows.append(
            {
                "prompt_label": label,
                "prompt_name": name,
                "geometry_rows": int(len(gdf)),
                "unique_assets": int(gdf["asset_key"].nunique()) if "asset_key" in gdf else 0,
                "unique_locations": int(gdf["location_text"].nunique()) if "location_text" in gdf else 0,
                "avg_area_km2": round(float(gdf["area_km2"].mean()), 3) if "area_km2" in gdf else 0.0,
                "avg_resolution_confidence": round(float(gdf["resolution_confidence"].mean()), 3)
                if "resolution_confidence" in gdf
                else 0.0,
                "geometry_overshoot_count": int(gdf["geometry_overshoot"].sum())
                if "geometry_overshoot" in gdf
                else 0,
                "postprocess_counts": (
                    gdf["geometry_postprocess"].fillna("").astype(str).value_counts().to_dict()
                    if "geometry_postprocess" in gdf
                    else {}
                ),
            }
        )
    return pd.DataFrame(rows)


def add_prompt_layer(
    fmap: folium.Map,
    gdf: gpd.GeoDataFrame,
    color: str,
    show: bool,
) -> None:
    label = str(gdf["prompt_label"].iloc[0])
    name = str(gdf["prompt_name"].iloc[0])
    layer_name = f"{label}: {name}" if name else label
    layer = folium.FeatureGroup(name=layer_name, show=show)
    present_popup_fields = [field for field in POPUP_FIELDS if field in gdf.columns]
    present_tooltip_fields = [field for field in TOOLTIP_FIELDS if field in gdf.columns]

    folium.GeoJson(
        clean_for_geojson(gdf),
        name=layer_name,
        style_function=lambda _feature, layer_color=color: {
            "color": layer_color,
            "weight": 3,
            "fillColor": layer_color,
            "fillOpacity": 0.18,
        },
        highlight_function=lambda _feature: {
            "weight": 5,
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=present_tooltip_fields,
            aliases=[field.replace("_", " ").title() for field in present_tooltip_fields],
            sticky=True,
        )
        if present_tooltip_fields
        else None,
        popup=folium.GeoJsonPopup(
            fields=present_popup_fields,
            aliases=[field.replace("_", " ").title() for field in present_popup_fields],
            max_width=520,
        )
        if present_popup_fields
        else None,
    ).add_to(layer)
    layer.add_to(fmap)


def create_map(gdfs: List[gpd.GeoDataFrame], output: Path, show_all: bool) -> None:
    combined = pd.concat(gdfs, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    minx, miny, maxx, maxy = combined.total_bounds

    fmap = folium.Map(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        max_zoom=19,
    )
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])
    Fullscreen().add_to(fmap)

    for idx, gdf in enumerate(gdfs):
        add_prompt_layer(
            fmap=fmap,
            gdf=gdf,
            color=COLORS[idx % len(COLORS)],
            show=show_all or idx == 0,
        )

    folium.LayerControl(collapsed=False).add_to(fmap)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output)
    logging.info("Wrote %s", output)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    paths = discover_inputs(args.input_root, args.pattern)
    if not paths:
        raise SystemExit(
            f"No geometry parquet files found under {args.input_root} with pattern {args.pattern!r}"
        )

    prompt_names = load_prompt_names()
    gdfs = [load_prompt_gdf(path, prompt_names) for path in paths]
    gdfs = [gdf for gdf in gdfs if not gdf.empty]
    if not gdfs:
        raise SystemExit("No non-empty geometry files found.")

    gdfs = sorted(
        gdfs,
        key=lambda gdf: (
            int(gdf["prompt_id"].iloc[0]) if str(gdf["prompt_id"].iloc[0]).isdigit() else 999999,
            str(gdf["source_path"].iloc[0]),
        ),
    )

    for gdf in gdfs:
        logging.info(
            "%s | %d geometries | %s",
            gdf["prompt_label"].iloc[0],
            len(gdf),
            gdf["source_path"].iloc[0],
        )

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_name(args.output.stem + "_summary.csv")

    summary = build_summary(gdfs)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    logging.info("Wrote %s", summary_output)

    create_map(gdfs, args.output, show_all=args.show_all)


if __name__ == "__main__":
    main()
