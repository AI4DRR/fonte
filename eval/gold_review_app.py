"""
Local Streamlit reviewer for eval/gold.csv.

Run from the repository root:

    streamlit run eval/gold_review_app.py

The app edits eval/gold.csv in place, creating one timestamped backup per
Streamlit session before the first write.
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_GOLD_PATH = APP_DIR / "gold.csv"

REFERENCE_COLUMNS = (
    "asset_key",
    "sample_bucket",
    "hazard_family",
    "title",
    "content_url",
    "language",
    "runs_hazard_summary",
)

LABEL_COLUMNS = (
    "is_real_event",
    "true_hazard",
    "true_country",
    "true_locations",
    "true_date_start",
    "true_date_end",
    "notes",
)

REQUIRED_COLUMNS = (*REFERENCE_COLUMNS, *LABEL_COLUMNS)
REAL_EVENT_OPTIONS = ("", "T", "F", "unclear")


class GoldCsvError(ValueError):
    """Raised when eval/gold.csv is missing or not reviewable."""


def read_gold_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a gold CSV as strings and validate the expected review schema."""
    if not path.exists():
        raise GoldCsvError(f"Gold CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise GoldCsvError(
                "Gold CSV is missing required columns: " + ", ".join(missing)
            )
        rows = [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]

    return fieldnames, rows


def write_gold_csv_atomic(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, str]],
) -> None:
    """Atomically replace the gold CSV while preserving column order."""
    path = path.resolve()
    fieldnames = list(fieldnames)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
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
    """Create a timestamped backup next to the CSV and return its path."""
    if not path.exists():
        raise GoldCsvError(f"Cannot back up missing CSV: {path}")

    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    counter = 2
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak.{stamp}.{counter}")
        counter += 1

    shutil.copy2(path, backup)
    return backup


def first_blank_index(rows: list[dict[str, str]]) -> int:
    """Return the first row whose is_real_event label is blank."""
    for i, row in enumerate(rows):
        if not row.get("is_real_event", "").strip():
            return i
    return max(0, len(rows) - 1)


def remaining_blank_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not row.get("is_real_event", "").strip())


def is_valid_date_or_blank(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def normalize_real_event(value: str) -> str:
    value = (value or "").strip()
    return value if value in REAL_EVENT_OPTIONS else ""


def _set_row_index(index: int, rows: list[dict[str, str]]) -> None:
    import streamlit as st

    if not rows:
        st.session_state.row_index = 0
        return
    st.session_state.row_index = max(0, min(index, len(rows) - 1))


def _load_state(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import streamlit as st

    fieldnames, rows = read_gold_csv(path)
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
    write_gold_csv_atomic(path, fieldnames, rows)


def _row_badges(row: dict[str, str]) -> None:
    import streamlit as st

    cols = st.columns(4)
    cols[0].metric("Bucket", row.get("sample_bucket", "") or "-")
    cols[1].metric("Family", row.get("hazard_family", "") or "-")
    cols[2].metric("Language", row.get("language", "") or "-")
    cols[3].metric("Event label", row.get("is_real_event", "") or "blank")


def _source_panel(row: dict[str, str]) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    url = row.get("content_url", "").strip()
    st.subheader("Source")
    if not url:
        st.warning("This row has no content_url.")
        return

    st.link_button("Open source in browser", url)
    with st.expander("Embedded preview", expanded=True):
        st.caption("Some publishers block embedding. Use the browser button if this panel is blank.")
        components.iframe(url, height=720, scrolling=True)


def _label_form(row: dict[str, str], row_index: int) -> dict[str, str] | None:
    import streamlit as st

    st.subheader("Gold Labels")
    current_real = normalize_real_event(row.get("is_real_event", ""))
    current_real_index = REAL_EVENT_OPTIONS.index(current_real)
    widget_key = row.get("asset_key", "").strip() or str(row_index)

    with st.form(f"gold_label_form_{widget_key}", clear_on_submit=False):
        is_real_event = st.radio(
            "is_real_event",
            REAL_EVENT_OPTIONS,
            index=current_real_index,
            horizontal=True,
            format_func=lambda x: "blank" if x == "" else x,
            key=f"is_real_event_{widget_key}",
        )

        event_required = is_real_event == "T"
        if event_required:
            st.info("Event fields are required for scoring when is_real_event is T.")

        true_hazard = st.text_input(
            "true_hazard" + (" *" if event_required else ""),
            value=row.get("true_hazard", ""),
            placeholder="e.g. Flood",
            key=f"true_hazard_{widget_key}",
        )
        true_country = st.text_input(
            "true_country" + (" *" if event_required else ""),
            value=row.get("true_country", ""),
            placeholder="Country name as written in the document",
            key=f"true_country_{widget_key}",
        )
        true_locations = st.text_area(
            "true_locations" + (" *" if event_required else ""),
            value=row.get("true_locations", ""),
            placeholder="Semicolon-separated, e.g. Bihar; Patna",
            height=90,
            key=f"true_locations_{widget_key}",
        )

        date_cols = st.columns(2)
        true_date_start = date_cols[0].text_input(
            "true_date_start" + (" *" if event_required else ""),
            value=row.get("true_date_start", ""),
            placeholder="YYYY-MM-DD",
            key=f"true_date_start_{widget_key}",
        )
        true_date_end = date_cols[1].text_input(
            "true_date_end" + (" *" if event_required else ""),
            value=row.get("true_date_end", ""),
            placeholder="YYYY-MM-DD",
            key=f"true_date_end_{widget_key}",
        )

        start_valid = is_valid_date_or_blank(true_date_start)
        end_valid = is_valid_date_or_blank(true_date_end)
        if not start_valid or not end_valid:
            st.warning("Date fields should be YYYY-MM-DD. You can still save if the source is ambiguous.")

        notes = st.text_area(
            "notes",
            value=row.get("notes", ""),
            height=130,
            key=f"notes_{widget_key}",
        )

        submitted = st.form_submit_button("Save & Next", type="primary")

    if not submitted:
        return None

    return {
        "is_real_event": is_real_event,
        "true_hazard": true_hazard.strip(),
        "true_country": true_country.strip(),
        "true_locations": true_locations.strip(),
        "true_date_start": true_date_start.strip(),
        "true_date_end": true_date_end.strip(),
        "notes": notes.strip(),
    }


def _navigation(path: Path, rows: list[dict[str, str]]) -> None:
    import streamlit as st

    row_count = len(rows)
    current = st.session_state.row_index

    st.sidebar.header("Navigation")
    st.sidebar.progress(0 if row_count == 0 else (row_count - remaining_blank_count(rows)) / row_count)
    st.sidebar.write(f"{row_count - remaining_blank_count(rows)} / {row_count} rows labelled")
    st.sidebar.write(f"{remaining_blank_count(rows)} rows still blank")

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
        st.sidebar.caption(f"Backup: {st.session_state[backup_key]}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Gold CSV Reviewer", layout="wide")
    st.title("Gold CSV Reviewer")

    path = DEFAULT_GOLD_PATH
    try:
        _ensure_session_backup(path)
        fieldnames, rows = _load_state(path)
    except GoldCsvError as exc:
        st.error(str(exc))
        st.stop()

    if not rows:
        st.warning("No rows found in eval/gold.csv.")
        st.stop()

    _navigation(path, rows)
    row_index = st.session_state.row_index
    row = rows[row_index]

    st.caption(f"Row {row_index + 1} of {len(rows)}")
    st.header(row.get("title", "") or "(untitled)")
    st.code(row.get("asset_key", ""), language=None)
    _row_badges(row)

    st.subheader("Model Hazard Summary")
    st.write(row.get("runs_hazard_summary", "") or "-")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        _source_panel(row)
    with right:
        values = _label_form(row, row_index)

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
