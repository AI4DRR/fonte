"""
Easier validation reviewer for the gold prompt-5 run.

Run from the repository root:

    streamlit run eval/gold_validation_app.py
    # or: python3 -m streamlit run eval/gold_validation_app.py

For each queued event it shows prompt 5's full extraction and the source, then
you give ONE verdict — Correct / Partial / Wrong — with a single click that
saves and jumps to the next unreviewed event. Optional notes per event.

Reads/writes eval/gold_validation.csv (built by prefill_gold_validation.py),
editing in place with one timestamped backup per session before the first write.
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
DEFAULT_VALIDATION_PATH = APP_DIR / "gold_validation.csv"

VERDICTS = ("correct", "partial", "wrong")
VERDICT_LABELS = {"correct": "Correct ✓", "partial": "Partial ~", "wrong": "Wrong ✗"}

LIST_FIELDS = ("affected_locations", "key_impacts", "evidence_snippets", "missing_information")
REQUIRED_COLUMNS = ("asset_key", "event_index", "in_queue", "review_rank", "verdict", "notes")


class ValidationCsvError(ValueError):
    """Raised when the validation CSV is missing or malformed."""


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValidationCsvError(
            f"Validation CSV not found: {path}\n"
            "Build it first:  python3 eval/prefill_gold_validation.py"
        )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValidationCsvError("CSV missing required columns: " + ", ".join(missing))
        rows = [{k: (v if v is not None else "") for k, v in r.items()} for r in reader]
    return fieldnames, rows


def write_csv_atomic(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    path = path.resolve()
    fieldnames = list(fieldnames)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
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


def create_backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    counter = 2
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak.{stamp}.{counter}")
        counter += 1
    shutil.copy2(path, backup)
    return backup


def queue_rows(rows: list[dict[str, str]]) -> list[int]:
    """Indices of in_queue rows, ordered by review_rank."""
    def _rank(i: int) -> int:
        try:
            return int(rows[i].get("review_rank", "") or 10**9)
        except (TypeError, ValueError):
            return 10**9
    idxs = [i for i, r in enumerate(rows) if str(r.get("in_queue", "")).strip().upper() == "Y"]
    idxs.sort(key=_rank)
    return idxs


def first_unjudged(rows: list[dict[str, str]], queue: list[int]) -> int:
    for pos, i in enumerate(queue):
        if not str(rows[i].get("verdict", "")).strip():
            return pos
    return 0


def _fmt_list_field(value: str) -> str:
    parts = [p.strip() for p in str(value or "").split("|") if p.strip()]
    return "\n".join(f"- {p}" for p in parts) if parts else "_(none)_"


def _ensure_backup(path: Path) -> None:
    import streamlit as st
    key = f"backup_path::{path.resolve()}"
    if key not in st.session_state:
        st.session_state[key] = str(create_backup(path))


def _save(path: Path, fieldnames: list[str], rows: list[dict[str, str]], row_index: int, values: dict) -> None:
    rows[row_index].update(values)
    write_csv_atomic(path, fieldnames, rows)


def _event_panel(row: dict[str, str]) -> None:
    import streamlit as st

    hazard = row.get("event_hazard") or "(no hazard)"
    conf = row.get("event_confidence") or "-"
    prob = row.get("event_confidence_prob")
    conf_label = conf
    try:
        if str(prob).strip():
            conf_label = f"{conf} · p={float(prob):.2f}"
    except (TypeError, ValueError):
        pass

    st.markdown(f"### {hazard}")
    st.caption(
        f"Event {row.get('event_index','?')} of {row.get('event_count','?')} · "
        f"confidence: {conf_label} · risk: {row.get('auto_risk','-')}"
    )
    st.markdown(f"**Dates:** {row.get('event_dates') or '-'}")
    st.markdown(f"**Admin levels:** {row.get('location_admin_levels') or '-'}")
    st.markdown("**Affected locations:**\n" + _fmt_list_field(row.get("affected_locations")))
    st.markdown("**Key impacts:**\n" + _fmt_list_field(row.get("key_impacts")))
    with st.expander("Evidence & missing information", expanded=True):
        st.markdown("**Evidence snippets:**\n" + _fmt_list_field(row.get("evidence_snippets")))
        st.markdown("**Missing information:**\n" + _fmt_list_field(row.get("missing_information")))


def _source_panel(row: dict[str, str]) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    url = (row.get("content_url") or "").strip()
    st.subheader("Source")
    if not url:
        st.warning("This event has no content_url.")
        return
    st.link_button("Open source in browser", url)
    with st.expander("Embedded preview", expanded=True):
        st.caption("Some publishers block embedding. Use the browser button if blank.")
        components.iframe(url, height=640, scrolling=True)


def _advance(rows: list[dict[str, str]], queue: list[int]) -> None:
    import streamlit as st
    pos = st.session_state.queue_pos
    nxt = first_unjudged(rows, queue)
    # If everything judged, just step forward; else jump to next unjudged after current.
    after = [p for p in range(pos + 1, len(queue)) if not str(rows[queue[p]].get("verdict", "")).strip()]
    st.session_state.queue_pos = after[0] if after else (nxt if nxt != pos else min(pos + 1, len(queue) - 1))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Gold Validation", layout="wide")
    st.title("Gold Validation — prompt 5")

    path = DEFAULT_VALIDATION_PATH
    try:
        _ensure_backup(path)
        fieldnames, rows = read_csv(path)
    except ValidationCsvError as exc:
        st.error(str(exc))
        st.stop()

    queue = queue_rows(rows)
    if not queue:
        st.warning("No rows have in_queue == Y. Run prefill_gold_validation.py first.")
        st.stop()

    if "queue_pos" not in st.session_state:
        st.session_state.queue_pos = first_unjudged(rows, queue)

    judged = sum(1 for i in queue if str(rows[i].get("verdict", "")).strip())
    st.sidebar.header("Progress")
    st.sidebar.progress(judged / len(queue))
    st.sidebar.write(f"{judged} / {len(queue)} judged")
    counts = {v: sum(1 for i in queue if rows[i].get("verdict") == v) for v in VERDICTS}
    st.sidebar.write(
        f"✓ {counts['correct']}  ·  ~ {counts['partial']}  ·  ✗ {counts['wrong']}"
    )
    if st.sidebar.button("Jump to next unjudged"):
        st.session_state.queue_pos = first_unjudged(rows, queue)
        st.rerun()
    nav = st.sidebar.columns(2)
    if nav[0].button("Previous", disabled=st.session_state.queue_pos <= 0):
        st.session_state.queue_pos -= 1
        st.rerun()
    if nav[1].button("Next", disabled=st.session_state.queue_pos >= len(queue) - 1):
        st.session_state.queue_pos += 1
        st.rerun()
    backup_key = f"backup_path::{path.resolve()}"
    if backup_key in st.session_state:
        st.sidebar.caption(f"Backup: {st.session_state[backup_key]}")

    pos = max(0, min(st.session_state.queue_pos, len(queue) - 1))
    st.session_state.queue_pos = pos
    row_index = queue[pos]
    row = rows[row_index]

    st.caption(f"Queue {pos + 1} of {len(queue)}  ·  review_rank {row.get('review_rank','-')}")
    st.header(row.get("title", "") or "(untitled)")
    st.code(row.get("asset_key", ""), language=None)
    current = (row.get("verdict") or "").strip()
    if current:
        st.success(f"Current verdict: {VERDICT_LABELS.get(current, current)}")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        _event_panel(row)
    with right:
        _source_panel(row)

    st.divider()
    st.subheader("Verdict")
    bcols = st.columns(3)
    clicked = None
    if bcols[0].button(VERDICT_LABELS["correct"], use_container_width=True, type="primary"):
        clicked = "correct"
    if bcols[1].button(VERDICT_LABELS["partial"], use_container_width=True):
        clicked = "partial"
    if bcols[2].button(VERDICT_LABELS["wrong"], use_container_width=True):
        clicked = "wrong"

    notes = st.text_input("Notes (optional)", value=row.get("notes", ""), key=f"notes_{row_index}")

    if clicked is not None:
        _save(path, fieldnames, rows, row_index, {"verdict": clicked, "notes": notes.strip()})
        _advance(rows, queue)
        st.rerun()

    if st.button("Save notes only"):
        _save(path, fieldnames, rows, row_index, {"notes": notes.strip()})
        st.toast("Notes saved.")


if __name__ == "__main__":
    main()
