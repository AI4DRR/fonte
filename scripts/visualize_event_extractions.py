import argparse
import html
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "undrr-groundsource-mplconfig"),
)

import folium
import geopandas as gpd
import pandas as pd
from branca.element import Element
from folium.plugins import Fullscreen


LIST_COLUMNS = ["event_dates", "affected_locations", "key_impacts"]
META_SPLIT_COLUMNS = ["countries", "themes", "hazards", "organizations"]
MAP_POPUP_FIELDS = [
    "asset_key",
    "title",
    "hazard",
    "location_text",
    "start_date",
    "end_date",
    "area_km2",
    "resolution_admin_level",
    "resolution_geometry_source",
    "resolution_confidence",
    "geometry_postprocess",
    "geometry_overshoot",
    "content_url",
]
MAP_TOOLTIP_FIELDS = [
    "hazard",
    "location_text",
    "resolution_confidence",
]
HAZARD_UNSPECIFIED_LABEL = "(no hazard)"
CARTO_LIGHT_TILES = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual checks and QA summaries for event extraction outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/event_extractions/event_extractions.csv"),
        help="Path to event_extractions.csv or .jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/event_extractions_checks"),
        help="Directory for plots and QA tables.",
    )
    parser.add_argument(
        "--geometries",
        type=Path,
        default=Path("outputs/event_extractions/event_geometries.parquet"),
        help="Path to event_geometries.parquet for the interactive polygon map.",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=None,
        help="HTML map path to write. Defaults to <output-dir>/polygon_map.html.",
    )
    parser.add_argument(
        "--require-geometries",
        action="store_true",
        help="Fail instead of warning when --geometries does not exist.",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Only write QA summaries and static plots.",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Only write the interactive polygon map and map summary.",
    )
    parser.add_argument(
        "--min-map-confidence",
        type=float,
        default=None,
        help="Optional minimum resolution_confidence to include on the map.",
    )
    parser.add_argument(
        "--map-simplify-tolerance",
        type=float,
        default=0.02,
        help="Simplify map geometries by this many degrees before writing HTML; use 0 for exact polygons.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many top categories to show in frequency plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=40,
        help="How many rows to export for manual spot checks.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()



def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input must be a .csv or .jsonl file")

    logging.info("Loaded %s rows and %s columns from %s", len(df), len(df.columns), path)
    return df



def ensure_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        True: True,
        False: False,
    }
    return series.map(lambda x: mapping.get(str(x).strip().lower(), pd.NA) if pd.notna(x) else pd.NA)



def split_pipe_values(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]



def count_items(series: pd.Series) -> pd.Series:
    return series.map(lambda x: len(split_pipe_values(x)))



def explode_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    exploded = df[[column]].copy()
    exploded[column] = exploded[column].map(split_pipe_values)
    exploded = exploded.explode(column)
    exploded[column] = exploded[column].astype("string").str.strip()
    exploded = exploded[exploded[column].notna() & (exploded[column] != "")]
    return exploded



def explode_meta(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column])
    exploded = df[[column]].copy()
    exploded[column] = exploded[column].fillna("").astype(str).str.replace(";", "|", regex=False)
    exploded[column] = exploded[column].map(split_pipe_values)
    exploded = exploded.explode(column)
    exploded[column] = exploded[column].astype("string").str.strip()
    exploded = exploded[exploded[column].notna() & (exploded[column] != "")]
    return exploded



def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s", path)


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def read_geometries(
    path: Path,
    min_confidence: Optional[float],
    simplify_tolerance: float,
) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    gdf = gpd.read_parquet(path)
    if gdf.empty:
        raise ValueError(f"Geometry file has no rows: {path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"Geometry file has no non-empty geometries: {path}")

    if min_confidence is not None:
        if "resolution_confidence" not in gdf.columns:
            logging.warning("--min-map-confidence ignored; resolution_confidence column is missing.")
        else:
            gdf["resolution_confidence"] = pd.to_numeric(
                gdf["resolution_confidence"], errors="coerce"
            )
            gdf = gdf[gdf["resolution_confidence"] >= min_confidence].copy()
            if gdf.empty:
                raise ValueError(
                    f"No geometries remain after applying min confidence {min_confidence}."
                )

    if simplify_tolerance > 0:
        logging.info("Simplifying map geometries with tolerance %s degrees", simplify_tolerance)
        gdf["geometry"] = gdf.geometry.simplify(
            simplify_tolerance,
            preserve_topology=True,
        )
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            raise ValueError("No geometries remain after map simplification.")

    logging.info("Loaded %s geometries from %s", len(gdf), path)
    return gdf


def clean_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = gdf.copy()
    for col in cleaned.columns:
        if col == cleaned.geometry.name:
            continue
        if pd.api.types.is_bool_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype(str)
        elif pd.api.types.is_numeric_dtype(cleaned[col]):
            continue
        else:
            cleaned[col] = cleaned[col].fillna("").astype(str)

    for col in ["area_km2", "resolution_confidence"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").round(4)
    return cleaned


def polygon_style(feature: dict) -> dict:
    confidence_raw = feature.get("properties", {}).get("resolution_confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = None

    if confidence is None:
        color = "#636363"
    elif confidence >= 0.7:
        color = "#238b45"
    elif confidence >= 0.4:
        color = "#fdae61"
    else:
        color = "#d73027"

    return {
        "color": color,
        "weight": 2,
        "fillColor": color,
        "fillOpacity": 0.2,
    }


def build_map_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "geometry_rows": int(len(gdf)),
                "unique_assets": int(gdf["asset_key"].nunique()) if "asset_key" in gdf else 0,
                "unique_locations": int(gdf["location_text"].nunique()) if "location_text" in gdf else 0,
                "avg_area_km2": round(float(gdf["area_km2"].mean()), 3)
                if "area_km2" in gdf
                else 0.0,
                "avg_resolution_confidence": round(float(gdf["resolution_confidence"].mean()), 3)
                if "resolution_confidence" in gdf
                else 0.0,
                "geometry_overshoot_count": int(gdf["geometry_overshoot"].sum())
                if "geometry_overshoot" in gdf
                else 0,
                "postprocess_counts": json.dumps(
                    gdf["geometry_postprocess"].fillna("").astype(str).value_counts().to_dict(),
                    sort_keys=True,
                )
                if "geometry_postprocess" in gdf
                else "{}",
            }
        ]
    )


def hazard_options(gdf: gpd.GeoDataFrame) -> List[Tuple[str, int]]:
    """Distinct hazard kinds on the map, most frequent first.

    `hazard` holds pipe-separated kinds for multi-hazard events, so a row can
    contribute to several options. Rows without a hazard get their own bucket
    so they stay toggleable instead of silently vanishing.
    """
    if "hazard" not in gdf.columns:
        return []

    counts: dict = {}
    for value in gdf["hazard"]:
        kinds = split_pipe_values(value) or [HAZARD_UNSPECIFIED_LABEL]
        for kind in dict.fromkeys(kinds):
            counts[kind] = counts.get(kind, 0) + 1

    return sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))


def hazard_control_html(hazards: List[Tuple[str, int]]) -> str:
    if not hazards:
        return ""

    checkboxes = "\n".join(
        '        <label class="hazard-option">'
        f'<input type="checkbox" class="hazard-checkbox" value="{html.escape(kind, quote=True)}" checked>'
        f'<span>{html.escape(kind)}</span>'
        f'<span class="hazard-count">{count:,}</span>'
        "</label>"
        for kind, count in hazards
    )
    return f"""
      <div class="filter-divider"></div>
      <label>Hazard kind</label>
      <div class="hazard-actions">
        <button type="button" id="hazard-select-all">All</button>
        <button type="button" id="hazard-select-none">None</button>
      </div>
      <div id="hazard-list">
{checkboxes}
      </div>
"""


def add_map_filters(
    fmap: folium.Map,
    geojson_name: str,
    hazards: List[Tuple[str, int]],
) -> None:
    control_html = """
    <div id="confidence-filter" class="leaflet-bar">
      <label>Confidence range</label>
      <div class="confidence-values">
        <span>Min <output id="confidence-min-value">0.00</output></span>
        <span>Max <output id="confidence-max-value">1.00</output></span>
      </div>
      <div class="confidence-range">
        <div class="confidence-track"></div>
        <div id="confidence-fill"></div>
        <input id="confidence-min-slider" type="range" min="0" max="1" step="0.01" value="0">
        <input id="confidence-max-slider" type="range" min="0" max="1" step="0.01" value="1">
      </div>
      __HAZARD_CONTROL__
      <div id="confidence-count"></div>
    </div>
    <style>
      #confidence-filter {
        position: fixed;
        top: 86px;
        right: 12px;
        z-index: 9999;
        width: 240px;
        max-height: calc(100vh - 120px);
        overflow-y: auto;
        padding: 10px 12px;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #9d9d9d;
        border-radius: 4px;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.22);
        color: #222;
        font: 13px/1.35 Arial, sans-serif;
      }
      #confidence-filter label {
        display: block;
        margin-bottom: 6px;
        font-weight: 700;
      }
      #confidence-filter .filter-divider {
        margin: 10px 0 8px;
        border-top: 1px solid #d8d8d8;
      }
      #confidence-filter .hazard-actions {
        display: flex;
        gap: 6px;
        margin-bottom: 6px;
      }
      #confidence-filter .hazard-actions button {
        flex: 1;
        padding: 3px 0;
        border: 1px solid #9d9d9d;
        border-radius: 3px;
        background: #f4f4f4;
        color: #222;
        font: inherit;
        cursor: pointer;
      }
      #confidence-filter .hazard-actions button:hover {
        background: #e6e6e6;
      }
      #hazard-list {
        max-height: 220px;
        overflow-y: auto;
        padding-right: 2px;
      }
      #confidence-filter .hazard-option {
        display: flex;
        align-items: flex-start;
        gap: 6px;
        margin-bottom: 2px;
        font-weight: 400;
        cursor: pointer;
      }
      #confidence-filter .hazard-option input {
        margin: 2px 0 0;
        flex: none;
      }
      #confidence-filter .hazard-option span:first-of-type {
        flex: 1;
      }
      #confidence-filter .hazard-count {
        color: #777;
        font-variant-numeric: tabular-nums;
      }
      #confidence-filter .confidence-values {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-variant-numeric: tabular-nums;
      }
      #confidence-filter .confidence-range {
        position: relative;
        height: 28px;
      }
      #confidence-filter .confidence-track,
      #confidence-fill {
        position: absolute;
        top: 12px;
        left: 0;
        right: 0;
        height: 4px;
        border-radius: 999px;
      }
      #confidence-filter .confidence-track {
        background: #d8d8d8;
      }
      #confidence-fill {
        background: #238b45;
      }
      #confidence-filter input[type="range"] {
        position: absolute;
        top: 3px;
        left: 0;
        width: 100%;
        margin: 0;
        background: none;
        pointer-events: none;
        -webkit-appearance: none;
        appearance: none;
      }
      #confidence-filter input[type="range"]::-webkit-slider-thumb {
        width: 16px;
        height: 16px;
        border: 2px solid #1f1f1f;
        border-radius: 50%;
        background: #ffffff;
        cursor: pointer;
        pointer-events: auto;
        -webkit-appearance: none;
        appearance: none;
      }
      #confidence-filter input[type="range"]::-moz-range-thumb {
        width: 14px;
        height: 14px;
        border: 2px solid #1f1f1f;
        border-radius: 50%;
        background: #ffffff;
        cursor: pointer;
        pointer-events: auto;
      }
      #confidence-filter input[type="range"]::-moz-range-track {
        background: transparent;
      }
      #confidence-count {
        margin-top: 5px;
        color: #555;
      }
    </style>
    """
    script = f"""
    window.addEventListener("load", function() {{
      var polygonLayer = {geojson_name};
      var minSlider = document.getElementById("confidence-min-slider");
      var maxSlider = document.getElementById("confidence-max-slider");
      var minValue = document.getElementById("confidence-min-value");
      var maxValue = document.getElementById("confidence-max-value");
      var fill = document.getElementById("confidence-fill");
      var count = document.getElementById("confidence-count");
      var hazardBoxes = Array.prototype.slice.call(
        document.querySelectorAll(".hazard-checkbox")
      );
      var unspecifiedHazard = {json.dumps(HAZARD_UNSPECIFIED_LABEL)};
      var allLayers = [];

      if (!polygonLayer || !minSlider || !maxSlider || !minValue || !maxValue || !fill || !count) {{
        return;
      }}

      polygonLayer.eachLayer(function(layer) {{
        allLayers.push(layer);
      }});

      function confidenceFor(layer) {{
        var raw = layer.feature && layer.feature.properties
          ? layer.feature.properties.resolution_confidence
          : null;
        var parsed = Number(raw);
        return Number.isFinite(parsed) ? parsed : 0;
      }}

      function hazardsFor(layer) {{
        var raw = layer.feature && layer.feature.properties
          ? layer.feature.properties.hazard
          : null;
        var kinds = String(raw == null ? "" : raw)
          .split("|")
          .map(function(part) {{ return part.trim(); }})
          .filter(function(part) {{ return part.length > 0; }});
        return kinds.length ? kinds : [unspecifiedHazard];
      }}

      function selectedHazards() {{
        var selected = {{}};
        hazardBoxes.forEach(function(box) {{
          if (box.checked) {{
            selected[box.value] = true;
          }}
        }});
        return selected;
      }}

      function applyFilter() {{
        var minConfidence = Number(minSlider.value);
        var maxConfidence = Number(maxSlider.value);
        var hazardFilter = selectedHazards();
        var visible = 0;

        if (minConfidence > maxConfidence) {{
          if (document.activeElement === minSlider) {{
            maxConfidence = minConfidence;
            maxSlider.value = String(maxConfidence);
          }} else {{
            minConfidence = maxConfidence;
            minSlider.value = String(minConfidence);
          }}
        }}

        minValue.textContent = minConfidence.toFixed(2);
        maxValue.textContent = maxConfidence.toFixed(2);
        fill.style.left = (minConfidence * 100).toFixed(1) + "%";
        fill.style.right = ((1 - maxConfidence) * 100).toFixed(1) + "%";

        allLayers.forEach(function(layer) {{
          var confidence = confidenceFor(layer);
          var inConfidence = confidence >= minConfidence && confidence <= maxConfidence;
          var inHazard = !hazardBoxes.length || hazardsFor(layer).some(function(kind) {{
            return hazardFilter[kind] === true;
          }});

          if (inConfidence && inHazard) {{
            if (!polygonLayer.hasLayer(layer)) {{
              polygonLayer.addLayer(layer);
            }}
            visible += 1;
          }} else if (polygonLayer.hasLayer(layer)) {{
            polygonLayer.removeLayer(layer);
          }}
        }});

        count.textContent = visible.toLocaleString() + " / "
          + allLayers.length.toLocaleString() + " polygons";
      }}

      function setAllHazards(checked) {{
        hazardBoxes.forEach(function(box) {{
          box.checked = checked;
        }});
        applyFilter();
      }}

      minSlider.addEventListener("input", applyFilter);
      maxSlider.addEventListener("input", applyFilter);
      hazardBoxes.forEach(function(box) {{
        box.addEventListener("change", applyFilter);
      }});

      var selectAll = document.getElementById("hazard-select-all");
      var selectNone = document.getElementById("hazard-select-none");
      if (selectAll) {{
        selectAll.addEventListener("click", function() {{ setAllHazards(true); }});
      }}
      if (selectNone) {{
        selectNone.addEventListener("click", function() {{ setAllHazards(false); }});
      }}

      applyFilter();
    }});
    """
    fmap.get_root().html.add_child(
        Element(control_html.replace("__HAZARD_CONTROL__", hazard_control_html(hazards)))
    )
    fmap.get_root().script.add_child(Element(script))


def save_polygon_map(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    fmap = folium.Map(
        tiles=CARTO_LIGHT_TILES,
        attr=CARTO_ATTRIBUTION,
        max_zoom=19,
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    if pd.notna([minx, miny, maxx, maxy]).all():
        fmap.fit_bounds([[miny, minx], [maxy, maxx]])

    Fullscreen().add_to(fmap)
    present_popup_fields = [field for field in MAP_POPUP_FIELDS if field in gdf.columns]
    present_tooltip_fields = [field for field in MAP_TOOLTIP_FIELDS if field in gdf.columns]

    geojson = folium.GeoJson(
        clean_for_geojson(gdf).to_json(drop_id=True),
        name="Event polygons",
        style_function=polygon_style,
        highlight_function=lambda _feature: {
            "weight": 4,
            "fillOpacity": 0.38,
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
            max_width=560,
        )
        if present_popup_fields
        else None,
    )
    geojson.add_to(fmap)
    add_map_filters(fmap, geojson.get_name(), hazard_options(gdf))

    folium.LayerControl(collapsed=False).add_to(fmap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_path)
    logging.info("Wrote %s", output_path)


def write_map_outputs(args: argparse.Namespace) -> Path:
    gdf = read_geometries(
        args.geometries,
        args.min_map_confidence,
        args.map_simplify_tolerance,
    )
    map_output = args.map_output or args.output_dir / "polygon_map.html"
    save_polygon_map(gdf, map_output)
    write_table(build_map_summary(gdf), map_output.with_name(map_output.stem + "_summary.csv"))
    return map_output



def save_bar(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    plt = get_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info("Wrote %s", output_path)



def save_hist(series: pd.Series, title: str, xlabel: str, output_path: Path, bins: int = 20) -> None:
    plt = get_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info("Wrote %s", output_path)



def build_summary(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }
    if "asset_key" in df.columns:
        summary["n_unique_asset_key"] = int(df["asset_key"].nunique(dropna=True))
        summary["n_duplicate_asset_key_rows"] = int(df.duplicated(subset=["asset_key"]).sum())

    for col in ["is_single_actual_event", "is_verifiable_event"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            summary[f"{col}_counts"] = {str(k): int(v) for k, v in vc.items()}

    for col in LIST_COLUMNS:
        if col in df.columns:
            counts = count_items(df[col])
            summary[f"avg_{col}_count"] = float(counts.mean()) if len(counts) else 0.0
            summary[f"max_{col}_count"] = int(counts.max()) if len(counts) else 0

    return summary



def generate_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = pd.DataFrame(index=df.index)
    flagged["asset_key"] = df.get("asset_key")
    flagged["title"] = df.get("title")
    flagged["content_url"] = df.get("content_url")

    event_counts = {
        col: count_items(df[col]) if col in df.columns else pd.Series([0] * len(df), index=df.index)
        for col in LIST_COLUMNS
    }

    key_impacts_text = df.get("key_impacts", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

    flags = pd.DataFrame(index=df.index)
    flags["error_in_key_impacts"] = key_impacts_text.str.startswith("ERROR:")
    flags["single_but_not_verifiable"] = (
        df.get("is_single_actual_event", pd.Series([pd.NA] * len(df), index=df.index)).fillna(False)
        & ~df.get("is_verifiable_event", pd.Series([pd.NA] * len(df), index=df.index)).fillna(False)
    )
    flags["verifiable_but_missing_hazard"] = (
        df.get("is_verifiable_event", pd.Series([pd.NA] * len(df), index=df.index)).fillna(False)
        & df.get("event_hazard", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip().eq("")
    )
    flags["too_many_dates"] = event_counts["event_dates"] > 5
    flags["too_many_locations"] = event_counts["affected_locations"] > 10
    flags["too_many_impacts"] = event_counts["key_impacts"] > 5
    flags["missing_title"] = df.get("title", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip().eq("")
    if "asset_key" in df.columns:
        flags["duplicate_asset_key"] = df.duplicated(subset=["asset_key"], keep=False)
    else:
        flags["duplicate_asset_key"] = False

    flagged["flag_count"] = flags.sum(axis=1)
    for col in flags.columns:
        flagged[col] = flags[col]

    flagged = flagged[flagged["flag_count"] > 0].sort_values(["flag_count", "asset_key"], ascending=[False, True])
    return flagged



def make_manual_samples(df: pd.DataFrame, flagged: pd.DataFrame, sample_size: int, output_dir: Path) -> None:
    base_cols = [
        c for c in [
            "asset_key",
            "title",
            "language",
            "countries",
            "themes",
            "hazards",
            "content_url",
            "is_single_actual_event",
            "is_verifiable_event",
            "event_hazard",
            "event_dates",
            "affected_locations",
            "key_impacts",
        ] if c in df.columns
    ]

    if len(df) > 0:
        sample_all = df[base_cols].sample(n=min(sample_size, len(df)), random_state=42)
        write_table(sample_all, output_dir / "manual_review_random_sample.csv")

    if len(flagged) > 0:
        merged = flagged.merge(df, on=[c for c in ["asset_key", "title", "content_url"] if c in flagged.columns and c in df.columns], how="left")
        cols = [c for c in flagged.columns.tolist() + base_cols if c in merged.columns]
        write_table(merged[cols].head(sample_size), output_dir / "manual_review_flagged_sample.csv")



def save_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", output_path)



def save_report(df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    lines = [
        "# Event extraction QA report",
        "",
        f"- Rows: {summary.get('n_rows', 0)}",
        f"- Columns: {summary.get('n_columns', 0)}",
    ]
    if "n_unique_asset_key" in summary:
        lines.append(f"- Unique asset_key: {summary['n_unique_asset_key']}")
        lines.append(f"- Duplicate asset_key rows: {summary['n_duplicate_asset_key_rows']}")

    for col in ["is_single_actual_event", "is_verifiable_event"]:
        if f"{col}_counts" in summary:
            lines.append(f"- {col}: {summary[f'{col}_counts']}")

    lines.extend([
        "",
        "## Main QA questions",
        "",
        "- Are there duplicate asset_keys?",
        "- Are there many rows marked as single events but not verifiable?",
        "- Are there extraction errors in key_impacts?",
        "- Are list fields unusually long?",
        "- Do the hazard, language, and country distributions look plausible?",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote %s", output_path)



def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.map_only:
        try:
            write_map_outputs(args)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        logging.info("Done. Map outputs written from %s", args.geometries)
        return

    df = read_input(args.input)

    for col in ["is_single_actual_event", "is_verifiable_event"]:
        if col in df.columns:
            df[col] = ensure_bool(df[col])

    for col in LIST_COLUMNS:
        if col in df.columns:
            df[f"{col}_count"] = count_items(df[col])

    summary = build_summary(df)
    save_summary_json(summary, args.output_dir / "summary.json")
    save_report(df, summary, args.output_dir / "qa_report.md")

    flagged = generate_flags(df)
    write_table(flagged, args.output_dir / "flagged_rows.csv")

    if "is_single_actual_event" in df.columns:
        series = df["is_single_actual_event"].fillna("NA").astype(str).value_counts()
        save_bar(series, "Single actual event classification", "Value", args.output_dir / "single_event_counts.png")

    if "is_verifiable_event" in df.columns:
        series = df["is_verifiable_event"].fillna("NA").astype(str).value_counts()
        save_bar(series, "Verifiable event classification", "Value", args.output_dir / "verifiable_event_counts.png")

    if "event_hazard" in df.columns:
        hazard_counts = (
            df["event_hazard"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(args.top_n)
        )
        if len(hazard_counts) > 0:
            save_bar(hazard_counts, "Top extracted hazards", "Hazard", args.output_dir / "top_hazards.png")

    if "language" in df.columns:
        lang_counts = (
            df["language"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(args.top_n)
        )
        if len(lang_counts) > 0:
            save_bar(lang_counts, "Top document languages", "Language", args.output_dir / "top_languages.png")

    for col in META_SPLIT_COLUMNS:
        exploded = explode_meta(df, col)
        if len(exploded) > 0:
            counts = exploded[col].value_counts().head(args.top_n)
            save_bar(counts, f"Top {col}", col, args.output_dir / f"top_{col}.png")

    for col in LIST_COLUMNS:
        count_col = f"{col}_count"
        if count_col in df.columns:
            save_hist(df[count_col], f"Distribution of {col} counts", count_col, args.output_dir / f"{count_col}_hist.png")

    if not args.skip_map:
        map_output = None
        try:
            map_output = write_map_outputs(args)
        except FileNotFoundError as exc:
            if args.require_geometries:
                raise SystemExit(str(exc)) from exc
            logging.warning("%s; skipping polygon map.", exc)

    make_manual_samples(df, flagged, args.sample_size, args.output_dir)
    logging.info("Done. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
