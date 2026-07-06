"""
Local Streamlit reviewer for a polygon_validation_sample.csv.

Run from the repository root:

    streamlit run eval/polygon_validation_review_app.py

By default it edits the test_2000 prompt05 sample:

    outputs/test_2000_gpt5_medium_prompt05_polygon_validation/polygon_validation_sample.csv

Point it at a different sample with the POLYGON_VALIDATION_CSV env var, or by
passing a path after `--`:

    POLYGON_VALIDATION_CSV=outputs/.../polygon_validation_sample.csv \
        streamlit run eval/polygon_validation_review_app.py
    streamlit run eval/polygon_validation_review_app.py -- outputs/.../polygon_validation_sample.csv

The app edits the CSV in place, creating one timestamped backup per Streamlit
session before the first write. It autosaves the current row when you press
"Save & Next".

For each row the reviewer judges:
* location_correct  — did the EXTRACTOR pick a real affected place? (Y/N/Partial)
* geometry_correct  — does the RESOLVER's polygon match the text? (Y/N/Partial/NA)
* correct_admin_level — free text: what admin tier the place really is
* date_correct      — did the EXTRACTOR get the event dates right? (Y/N/Partial)
* correct_dates     — free text: what the dates should have been
* notes             — free text
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = (
    REPO_ROOT
    / "outputs"
    / "test_2000_gpt5_medium_prompt05_polygon_validation"
    / "polygon_validation_sample.csv"
)

# Columns the resolver/sample builder writes; we never edit these.
REFERENCE_COLUMNS = (
    "stratum",
    "review_id",
    "hazard",
    "start_date",
    "end_date",
    "event_dates_raw",
    "location_text",
    "title",
    "content_url",
    "resolution_admin_level",
    "resolution_geometry_source",
    "resolution_confidence",
    "resolution_used_fallback",
    "resolution_display_name",
    "resolution_query",
    "area_km2",
    "map_url",
    "uuid",
    "asset_key",
)

# Reviewer columns we edit.
LABEL_COLUMNS = (
    "location_correct",
    "geometry_correct",
    "correct_admin_level",
    "date_correct",
    "correct_dates",
    "notes",
)

# The "reviewed?" gate. A row counts as done once location_correct is set.
PRIMARY_LABEL = "location_correct"

YN_OPTIONS = ("", "Y", "N", "Partial")
YN_NA_OPTIONS = ("", "Y", "N", "Partial", "NA")


class ValidationCsvError(ValueError):
    """Raised when the validation CSV is missing or not reviewable."""


# ---------------------------------------------------------------------------
# CSV IO (atomic, backup-on-first-write) — mirrors eval/gold_review_app.py
# ---------------------------------------------------------------------------

def _clean(value: object) -> str:
    """Normalise a CSV cell to a stripped string, treating NaN/None as blank."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def read_validation_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValidationCsvError(f"Validation CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [c for c in REFERENCE_COLUMNS if c not in fieldnames]
        if missing:
            raise ValidationCsvError(
                "Validation CSV is missing required columns: " + ", ".join(missing)
            )
        # Make sure the label columns exist so we can write them back.
        for col in LABEL_COLUMNS:
            if col not in fieldnames:
                fieldnames.append(col)
        rows = [{k: _clean(row.get(k)) for k in fieldnames} for row in reader]

    return fieldnames, rows


def write_validation_csv_atomic(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, str]],
) -> None:
    path = path.resolve()
    fieldnames = list(fieldnames)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in fieldnames})
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def create_backup(path: Path, timestamp: str | None = None) -> Path:
    if not path.exists():
        raise ValidationCsvError(f"Cannot back up missing CSV: {path}")
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    counter = 2
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak.{stamp}.{counter}")
        counter += 1
    shutil.copy2(path, backup)
    return backup


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def first_blank_index(rows: list[dict[str, str]]) -> int:
    for i, row in enumerate(rows):
        if not row.get(PRIMARY_LABEL, "").strip():
            return i
    return max(0, len(rows) - 1)


def remaining_blank_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not row.get(PRIMARY_LABEL, "").strip())


def normalize_choice(value: str, options: tuple[str, ...]) -> str:
    value = (value or "").strip()
    return value if value in options else ""


# ---------------------------------------------------------------------------
# Map embedding — parse the OSM map_url into an embeddable iframe
# ---------------------------------------------------------------------------

_MAP_RE = re.compile(r"#map=([0-9.]+)/(-?[0-9.]+)/(-?[0-9.]+)")


def parse_map_url(url: str) -> tuple[float, float, float] | None:
    """Return (lat, lon, zoom) parsed from an OSM map_url, or None."""
    if not url:
        return None
    m = _MAP_RE.search(url)
    if not m:
        return None
    zoom, lat, lon = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return lat, lon, zoom


def osm_embed_url(lat: float, lon: float, zoom: float) -> str:
    """Build an iframe-friendly OSM embed URL with a marker, deriving a bbox
    from the centroid + zoom (openstreetmap.org's main view blocks framing,
    but the export/embed endpoint allows it)."""
    half_lon = 360.0 / (2 ** zoom)
    half_lat = half_lon * 0.6  # rough Mercator/aspect correction
    min_lon, max_lon = lon - half_lon, lon + half_lon
    min_lat, max_lat = lat - half_lat, lat + half_lat
    return (
        "https://www.openstreetmap.org/export/embed.html"
        f"?bbox={min_lon:.5f},{min_lat:.5f},{max_lon:.5f},{max_lat:.5f}"
        f"&layer=mapnik&marker={lat:.5f},{lon:.5f}"
    )


# ---------------------------------------------------------------------------
# Streamlit state
# ---------------------------------------------------------------------------

def resolve_csv_path() -> Path:
    env = os.environ.get("POLYGON_VALIDATION_CSV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Streamlit forwards args after `--` into sys.argv.
    for arg in sys.argv[1:]:
        if arg and not arg.startswith("-"):
            return Path(arg).expanduser().resolve()
    return DEFAULT_CSV_PATH


def _set_row_index(index: int, rows: list[dict[str, str]]) -> None:
    import streamlit as st

    if not rows:
        st.session_state.row_index = 0
        return
    st.session_state.row_index = max(0, min(index, len(rows) - 1))


def _load_state(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import streamlit as st

    fieldnames, rows = read_validation_csv(path)
    if "row_index" not in st.session_state:
        st.session_state.row_index = first_blank_index(rows)
    return fieldnames, rows


def _ensure_session_backup(path: Path) -> None:
    import streamlit as st

    key = f"backup_path::{path.resolve()}"
    if key not in st.session_state:
        st.session_state[key] = str(create_backup(path))


def _save_current_row(
    *,
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    row_index: int,
    values: dict[str, str],
) -> None:
    rows[row_index].update(values)
    write_validation_csv_atomic(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# UI panels
# ---------------------------------------------------------------------------

STRATUM_BADGE = {
    "high_confidence": "🟢 high_confidence",
    "low_confidence": "🟡 low_confidence",
    "unresolved": "🔴 unresolved",
}


def _resolver_panel(row: dict[str, str]) -> None:
    import streamlit as st

    st.subheader("What the resolver decided")
    cols = st.columns(3)
    cols[0].metric("Admin level", row.get("resolution_admin_level", "") or "-")
    cols[1].metric("Geometry source", row.get("resolution_geometry_source", "") or "-")
    conf = row.get("resolution_confidence", "")
    cols[2].metric("Confidence", f"{float(conf):.3f}" if conf else "-")

    cols2 = st.columns(3)
    cols2[0].metric("Used fallback", row.get("resolution_used_fallback", "") or "-")
    area = row.get("area_km2", "")
    cols2[1].metric("Area (km²)", f"{float(area):,.0f}" if area else "-")
    cols2[2].metric("Stratum", "")  # filled via caption below for full text

    st.caption(
        f"**Nominatim display_name:** {row.get('resolution_display_name', '') or '-'}  \n"
        f"**Resolver query:** `{row.get('resolution_query', '') or '-'}`"
    )


def _map_panel(row: dict[str, str]) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.subheader("Resolved geometry")
    map_url = row.get("map_url", "").strip()
    parsed = parse_map_url(map_url)
    if not parsed:
        st.info("No geometry for this row (unresolved). Judge location & dates only.")
        return

    lat, lon, zoom = parsed
    components.iframe(osm_embed_url(lat, lon, zoom), height=420, scrolling=False)
    st.link_button("Open larger map in OpenStreetMap", map_url)


def _source_panel(row: dict[str, str]) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    url = row.get("content_url", "").strip()
    st.subheader("Source document")
    if not url:
        st.warning("This row has no content_url.")
        return
    st.link_button("Open source in browser", url)
    with st.expander("Embedded preview", expanded=False):
        st.caption("Some publishers block embedding. Use the browser button if blank.")
        components.iframe(url, height=600, scrolling=True)


def _review_form(row: dict[str, str], row_index: int) -> dict[str, str] | None:
    import streamlit as st

    st.subheader("Your review")
    widget_key = row.get("uuid", "").strip() or str(row_index)
    is_unresolved = row.get("stratum", "") == "unresolved"

    loc_idx = YN_OPTIONS.index(normalize_choice(row.get("location_correct", ""), YN_OPTIONS))
    geom_idx = YN_NA_OPTIONS.index(
        normalize_choice(row.get("geometry_correct", ""), YN_NA_OPTIONS)
    )
    date_idx = YN_OPTIONS.index(normalize_choice(row.get("date_correct", ""), YN_OPTIONS))

    with st.form(f"review_form_{widget_key}", clear_on_submit=False):
        location_correct = st.radio(
            "location_correct — is this a real affected place? (judges the extractor)",
            YN_OPTIONS,
            index=loc_idx,
            horizontal=True,
            format_func=lambda x: "blank" if x == "" else x,
            key=f"loc_{widget_key}",
        )

        geometry_correct = st.radio(
            "geometry_correct — does the polygon match the text? (judges the resolver)"
            + (" — NA for unresolved" if is_unresolved else ""),
            YN_NA_OPTIONS,
            index=geom_idx,
            horizontal=True,
            format_func=lambda x: "blank" if x == "" else x,
            key=f"geom_{widget_key}",
        )

        correct_admin_level = st.text_input(
            "correct_admin_level — what tier is it really? (country/admin1/admin2/city/…)",
            value=row.get("correct_admin_level", ""),
            key=f"adm_{widget_key}",
        )

        date_correct = st.radio(
            "date_correct — did the extractor get the dates right?",
            YN_OPTIONS,
            index=date_idx,
            horizontal=True,
            format_func=lambda x: "blank" if x == "" else x,
            key=f"date_{widget_key}",
        )
        correct_dates = st.text_input(
            "correct_dates — what they should have been (YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD)",
            value=row.get("correct_dates", ""),
            key=f"cdates_{widget_key}",
        )

        notes = st.text_area(
            "notes",
            value=row.get("notes", ""),
            height=110,
            key=f"notes_{widget_key}",
        )

        submitted = st.form_submit_button("Save & Next", type="primary")

    if not submitted:
        return None

    return {
        "location_correct": location_correct,
        "geometry_correct": geometry_correct,
        "correct_admin_level": correct_admin_level.strip(),
        "date_correct": date_correct,
        "correct_dates": correct_dates.strip(),
        "notes": notes.strip(),
    }


def _navigation(path: Path, rows: list[dict[str, str]]) -> None:
    import streamlit as st

    row_count = len(rows)
    current = st.session_state.row_index
    done = row_count - remaining_blank_count(rows)

    st.sidebar.header("Progress")
    st.sidebar.progress(0 if row_count == 0 else done / row_count)
    st.sidebar.write(f"{done} / {row_count} rows reviewed")
    st.sidebar.write(f"{remaining_blank_count(rows)} rows still blank")

    # Per-stratum progress.
    st.sidebar.divider()
    for stratum in ("high_confidence", "low_confidence", "unresolved"):
        pool = [r for r in rows if r.get("stratum") == stratum]
        if not pool:
            continue
        reviewed = sum(1 for r in pool if r.get(PRIMARY_LABEL, "").strip())
        st.sidebar.caption(
            f"{STRATUM_BADGE.get(stratum, stratum)}: {reviewed}/{len(pool)}"
        )

    st.sidebar.divider()
    jump_to = st.sidebar.number_input(
        "Jump to row",
        min_value=1,
        max_value=max(1, row_count),
        value=min(current + 1, max(1, row_count)),
        step=1,
    )
    if st.sidebar.button("Go to row"):
        _set_row_index(int(jump_to) - 1, rows)
        st.rerun()

    nav_cols = st.sidebar.columns(2)
    if nav_cols[0].button("Previous", disabled=current <= 0):
        _set_row_index(current - 1, rows)
        st.rerun()
    if nav_cols[1].button("Skip", disabled=current >= row_count - 1):
        _set_row_index(current + 1, rows)
        st.rerun()

    if st.sidebar.button("First blank"):
        _set_row_index(first_blank_index(rows), rows)
        st.rerun()

    backup_key = f"backup_path::{path.resolve()}"
    if backup_key in st.session_state:
        st.sidebar.divider()
        st.sidebar.caption(f"Editing: {path}")
        st.sidebar.caption(f"Backup: {st.session_state[backup_key]}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Polygon Validation Reviewer", layout="wide")
    st.title("Polygon Validation Reviewer")

    path = resolve_csv_path()
    try:
        _ensure_session_backup(path)
        fieldnames, rows = _load_state(path)
    except ValidationCsvError as exc:
        st.error(str(exc))
        st.stop()

    if not rows:
        st.warning(f"No rows found in {path}.")
        st.stop()

    _navigation(path, rows)
    row_index = st.session_state.row_index
    row = rows[row_index]

    stratum = row.get("stratum", "")
    st.caption(
        f"Review {row.get('review_id', row_index + 1)} · "
        f"row {row_index + 1} of {len(rows)} · {STRATUM_BADGE.get(stratum, stratum)}"
    )
    st.header(row.get("location_text", "") or "(no location text)")
    st.write(
        f"**Hazard:** {row.get('hazard', '') or '-'}  |  "
        f"**Extracted dates:** {row.get('event_dates_raw', '') or row.get('start_date', '') or '-'}"
    )
    st.code(row.get("title", "") or "(untitled)", language=None)

    _resolver_panel(row)

    left, mid, right = st.columns([1.0, 1.0, 1.0], gap="large")
    with left:
        _map_panel(row)
    with mid:
        _source_panel(row)
    with right:
        values = _review_form(row, row_index)

    if values is not None:
        _save_current_row(
            path=path,
            fieldnames=fieldnames,
            rows=rows,
            row_index=row_index,
            values=values,
        )
        _set_row_index(row_index + 1, rows)
        st.success("Saved.")
        st.rerun()


if __name__ == "__main__":
    main()
