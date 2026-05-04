import argparse
import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


CORE_FIELDS = [
    "is_single_actual_event",
    "is_verifiable_event",
    "event_confidence",
    "event_hazard",
    "event_dates",
    "affected_locations",
]

LIST_FIELDS = {
    "event_dates",
    "affected_locations",
    "location_admin_levels",
    "key_impacts",
    "evidence_snippets",
    "missing_information",
}

NULLISH = {"", "nan", "none", "null", "<na>"}


@dataclass
class PromptRun:
    path: Path
    prompt_id: Optional[int]
    prompt_label: str
    prompt_name: str
    rows: List[Dict[str, str]]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare event extraction CSVs produced by multiple prompt runs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder used to search for prompt output CSVs.",
    )
    parser.add_argument(
        "--pattern",
        default="test_*_prompt*/event_extractions.csv",
        help="Glob pattern under --input-root for prompt CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cross_prompt_comparison"),
        help="Directory where comparison CSVs and report will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Maximum number of hazards per prompt to write to hazard_by_prompt.csv.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def load_prompt_names() -> Dict[int, str]:
    try:
        from prompts import PROMPTS
    except Exception as exc:  # pragma: no cover - optional convenience only
        logging.warning("Could not import prompt names from prompts.py: %s", exc)
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_empty(value) -> bool:
    return clean(value).lower() in NULLISH


def compact_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_bool(value) -> Optional[bool]:
    text = clean(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def split_pipe_values(value) -> List[str]:
    if is_empty(value):
        return []
    return [compact_spaces(part) for part in clean(value).split("|") if part.strip()]


def list_count(row: Dict[str, str], field: str) -> int:
    return len(split_pipe_values(row.get(field, "")))


def rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 4)


def avg(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def bool_true_count(rows: Iterable[Dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if normalize_bool(row.get(field)) is True)


def duplicate_asset_rows(rows: Sequence[Dict[str, str]]) -> int:
    counts = Counter(
        clean(row.get("asset_key"))
        for row in rows
        if not is_empty(row.get("asset_key"))
    )
    return sum(count - 1 for count in counts.values() if count > 1)


def build_asset_index(run: PromptRun) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for row in run.rows:
        asset_key = clean(row.get("asset_key"))
        if not asset_key:
            continue
        if asset_key not in index:
            index[asset_key] = row
    return index


def normalized_field_value(row: Optional[Dict[str, str]], field: str) -> str:
    if row is None:
        return "<missing row>"

    value = row.get(field, "")
    if field in {"is_single_actual_event", "is_verifiable_event"}:
        normalized = normalize_bool(value)
        if normalized is True:
            return "true"
        if normalized is False:
            return "false"
        return ""

    if field in LIST_FIELDS:
        values = [compact_spaces(part).lower() for part in split_pipe_values(value)]
        return " | ".join(values)

    if is_empty(value):
        return ""
    return compact_spaces(clean(value)).lower()


def prompt_value_summary(
    runs: Sequence[PromptRun],
    indexes: Dict[str, Dict[str, Dict[str, str]]],
    asset_key: str,
    field: str,
) -> str:
    parts = []
    for run in runs:
        row = indexes[run.prompt_label].get(asset_key)
        parts.append(f"{run.prompt_label}={normalized_field_value(row, field)}")
    return " || ".join(parts)


def build_runs(paths: Sequence[Path]) -> List[PromptRun]:
    prompt_names = load_prompt_names()
    runs = []
    for path in paths:
        prompt_id = detect_prompt_id(path)
        label = prompt_label(prompt_id, path)
        name = prompt_names.get(prompt_id, "") if prompt_id is not None else ""
        rows = read_csv_rows(path)
        runs.append(
            PromptRun(
                path=path,
                prompt_id=prompt_id,
                prompt_label=label,
                prompt_name=name,
                rows=rows,
            )
        )

    return sorted(
        runs,
        key=lambda run: (
            run.prompt_id if run.prompt_id is not None else 999999,
            str(run.path),
        ),
    )


def build_prompt_summary(runs: Sequence[PromptRun]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for run in runs:
        rows = run.rows
        row_count = len(rows)
        unique_assets = len({
            clean(row.get("asset_key"))
            for row in rows
            if not is_empty(row.get("asset_key"))
        })
        confidence_counts = Counter(
            normalized_field_value(row, "event_confidence") or "blank"
            for row in rows
        )
        dates_counts = [list_count(row, "event_dates") for row in rows]
        location_counts = [list_count(row, "affected_locations") for row in rows]
        admin_counts = [list_count(row, "location_admin_levels") for row in rows]
        impact_counts = [list_count(row, "key_impacts") for row in rows]

        verifiable_missing_core = 0
        locations_without_dates = 0
        dates_without_locations = 0
        mismatched_location_admin = 0
        error_count = 0

        for row, date_count, location_count, admin_count in zip(
            rows, dates_counts, location_counts, admin_counts
        ):
            has_hazard = not is_empty(row.get("event_hazard"))
            has_dates = date_count > 0
            has_locations = location_count > 0
            if has_locations and not has_dates:
                locations_without_dates += 1
            if has_dates and not has_locations:
                dates_without_locations += 1
            if normalize_bool(row.get("is_verifiable_event")) is True and (
                not has_hazard or not has_dates or not has_locations
            ):
                verifiable_missing_core += 1
            if location_count != admin_count:
                mismatched_location_admin += 1
            if clean(row.get("key_impacts")).startswith("ERROR:"):
                error_count += 1

        single_true = bool_true_count(rows, "is_single_actual_event")
        verifiable_true = bool_true_count(rows, "is_verifiable_event")

        summary_rows.append(
            {
                "prompt_id": run.prompt_id if run.prompt_id is not None else "",
                "prompt_label": run.prompt_label,
                "prompt_name": run.prompt_name,
                "path": str(run.path),
                "row_count": row_count,
                "unique_asset_count": unique_assets,
                "missing_asset_key_count": sum(
                    1 for row in rows if is_empty(row.get("asset_key"))
                ),
                "duplicate_asset_key_rows": duplicate_asset_rows(rows),
                "is_single_true_count": single_true,
                "is_single_true_rate": rate(single_true, row_count),
                "is_verifiable_true_count": verifiable_true,
                "is_verifiable_true_rate": rate(verifiable_true, row_count),
                "nonempty_event_hazard_count": sum(
                    1 for row in rows if not is_empty(row.get("event_hazard"))
                ),
                "nonempty_event_dates_count": sum(1 for count in dates_counts if count > 0),
                "nonempty_affected_locations_count": sum(
                    1 for count in location_counts if count > 0
                ),
                "nonempty_key_impacts_count": sum(1 for count in impact_counts if count > 0),
                "confidence_high_count": confidence_counts["high"],
                "confidence_medium_count": confidence_counts["medium"],
                "confidence_low_count": confidence_counts["low"],
                "confidence_blank_count": confidence_counts["blank"],
                "error_count": error_count,
                "avg_event_dates_count": avg(dates_counts),
                "avg_affected_locations_count": avg(location_counts),
                "avg_key_impacts_count": avg(impact_counts),
                "locations_without_dates_count": locations_without_dates,
                "dates_without_locations_count": dates_without_locations,
                "verifiable_missing_core_count": verifiable_missing_core,
                "mismatched_location_admin_count": mismatched_location_admin,
            }
        )
    return summary_rows


def build_field_agreement(
    runs: Sequence[PromptRun],
    indexes: Dict[str, Dict[str, Dict[str, str]]],
    asset_keys: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    for field in CORE_FIELDS:
        agreement_count = 0
        all_empty_count = 0
        max_unique_values = 0

        for asset_key in asset_keys:
            values = [
                normalized_field_value(indexes[run.prompt_label].get(asset_key), field)
                for run in runs
            ]
            unique_values = set(values)
            max_unique_values = max(max_unique_values, len(unique_values))
            if len(unique_values) == 1:
                agreement_count += 1
            if unique_values == {""}:
                all_empty_count += 1

        compared = len(asset_keys)
        disagreement_count = compared - agreement_count
        rows.append(
            {
                "field": field,
                "compared_asset_count": compared,
                "agreement_count": agreement_count,
                "disagreement_count": disagreement_count,
                "disagreement_rate": rate(disagreement_count, compared),
                "all_empty_count": all_empty_count,
                "max_unique_values": max_unique_values,
            }
        )
    return rows


def first_nonempty_metadata(
    asset_key: str,
    runs: Sequence[PromptRun],
    indexes: Dict[str, Dict[str, Dict[str, str]]],
    field: str,
) -> str:
    for run in runs:
        row = indexes[run.prompt_label].get(asset_key)
        if row is not None and not is_empty(row.get(field)):
            return clean(row.get(field))
    return ""


def build_asset_disagreements(
    runs: Sequence[PromptRun],
    indexes: Dict[str, Dict[str, Dict[str, str]]],
    asset_keys: Sequence[str],
) -> List[Dict[str, object]]:
    disagreement_rows: List[Dict[str, object]] = []
    for asset_key in asset_keys:
        disagreed_fields = []
        value_summaries = {}

        for field in CORE_FIELDS:
            values = {
                normalized_field_value(indexes[run.prompt_label].get(asset_key), field)
                for run in runs
            }
            if len(values) > 1:
                disagreed_fields.append(field)
                value_summaries[f"{field}_values"] = prompt_value_summary(
                    runs, indexes, asset_key, field
                )
            else:
                value_summaries[f"{field}_values"] = ""

        if not disagreed_fields:
            continue

        disagreement_rows.append(
            {
                "asset_key": asset_key,
                "title": first_nonempty_metadata(asset_key, runs, indexes, "title"),
                "content_url": first_nonempty_metadata(asset_key, runs, indexes, "content_url"),
                "n_disagreed_fields": len(disagreed_fields),
                "disagreed_fields": " | ".join(disagreed_fields),
                **value_summaries,
            }
        )

    return sorted(
        disagreement_rows,
        key=lambda row: (-int(row["n_disagreed_fields"]), str(row["asset_key"])),
    )


def build_hazard_by_prompt(
    runs: Sequence[PromptRun],
    top_n: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in runs:
        counts = Counter(
            compact_spaces(clean(row.get("event_hazard")))
            for row in run.rows
            if not is_empty(row.get("event_hazard"))
        )
        for rank, (hazard, count) in enumerate(counts.most_common(top_n), start=1):
            rows.append(
                {
                    "prompt_id": run.prompt_id if run.prompt_id is not None else "",
                    "prompt_label": run.prompt_label,
                    "prompt_name": run.prompt_name,
                    "rank": rank,
                    "event_hazard": hazard,
                    "count": count,
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote %s", path)


def markdown_table(rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> List[str]:
    if not rows:
        return ["_No rows._"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")).replace("|", "/") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def save_report(
    output_path: Path,
    runs: Sequence[PromptRun],
    summary_rows: Sequence[Dict[str, object]],
    agreement_rows: Sequence[Dict[str, object]],
    disagreement_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    best_verifiable = max(
        summary_rows,
        key=lambda row: (float(row["is_verifiable_true_rate"]), int(row["is_verifiable_true_count"])),
    )
    most_single = max(
        summary_rows,
        key=lambda row: (float(row["is_single_true_rate"]), int(row["is_single_true_count"])),
    )
    most_disagreed_field = max(
        agreement_rows,
        key=lambda row: int(row["disagreement_count"]),
    )

    lines = [
        "# Cross-prompt event extraction comparison",
        "",
        f"- Prompt runs compared: {len(runs)}",
        f"- Assets with any disagreement: {len(disagreement_rows)}",
        f"- Highest verifiable rate: {best_verifiable['prompt_label']} "
        f"({best_verifiable['is_verifiable_true_count']}/{best_verifiable['row_count']})",
        f"- Highest single-event rate: {most_single['prompt_label']} "
        f"({most_single['is_single_true_count']}/{most_single['row_count']})",
        f"- Most disagreed field: {most_disagreed_field['field']} "
        f"({most_disagreed_field['disagreement_count']} assets)",
        "",
        "## Prompt Summary",
        "",
    ]

    lines.extend(
        markdown_table(
            summary_rows,
            [
                "prompt_label",
                "prompt_name",
                "row_count",
                "is_verifiable_true_count",
                "is_verifiable_true_rate",
                "is_single_true_count",
                "is_single_true_rate",
                "confidence_high_count",
                "confidence_medium_count",
                "confidence_low_count",
                "confidence_blank_count",
                "avg_affected_locations_count",
                "error_count",
            ],
        )
    )

    lines.extend([
        "",
        "## Field Agreement",
        "",
    ])
    lines.extend(
        markdown_table(
            agreement_rows,
            [
                "field",
                "compared_asset_count",
                "agreement_count",
                "disagreement_count",
                "disagreement_rate",
                "all_empty_count",
            ],
        )
    )

    lines.extend([
        "",
        "## What To Inspect Next",
        "",
        "- Use `asset_disagreements.csv` to manually review documents where prompts changed the answer.",
        "- Compare verifiable rates with confidence counts: more verifiable events can mean better recall, but only if disagreements look valid.",
        "- Compare average affected-location counts: higher counts may indicate richer extraction or location overshoot.",
        "- Check `mismatched_location_admin_count`, `verifiable_missing_core_count`, and `error_count` before trusting a prompt variant.",
        "",
        "## Files Written",
        "",
        f"- `{output_dir / 'prompt_summary.csv'}`",
        f"- `{output_dir / 'field_agreement.csv'}`",
        f"- `{output_dir / 'asset_disagreements.csv'}`",
        f"- `{output_dir / 'hazard_by_prompt.csv'}`",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    paths = discover_inputs(args.input_root, args.pattern)
    if not paths:
        raise SystemExit(
            f"No prompt CSVs found under {args.input_root} with pattern {args.pattern!r}"
        )

    runs = build_runs(paths)
    logging.info("Loaded %s prompt runs", len(runs))
    for run in runs:
        logging.info("  %s | %s rows | %s", run.prompt_label, len(run.rows), run.path)

    indexes = {run.prompt_label: build_asset_index(run) for run in runs}
    asset_keys = sorted({
        asset_key
        for index in indexes.values()
        for asset_key in index
    })

    summary_rows = build_prompt_summary(runs)
    agreement_rows = build_field_agreement(runs, indexes, asset_keys)
    disagreement_rows = build_asset_disagreements(runs, indexes, asset_keys)
    hazard_rows = build_hazard_by_prompt(runs, args.top_n)

    write_csv(
        args.output_dir / "prompt_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    write_csv(
        args.output_dir / "field_agreement.csv",
        agreement_rows,
        list(agreement_rows[0].keys()) if agreement_rows else [],
    )
    disagreement_headers = [
        "asset_key",
        "title",
        "content_url",
        "n_disagreed_fields",
        "disagreed_fields",
        *[f"{field}_values" for field in CORE_FIELDS],
    ]
    write_csv(
        args.output_dir / "asset_disagreements.csv",
        disagreement_rows,
        disagreement_headers,
    )
    hazard_headers = [
        "prompt_id",
        "prompt_label",
        "prompt_name",
        "rank",
        "event_hazard",
        "count",
    ]
    write_csv(args.output_dir / "hazard_by_prompt.csv", hazard_rows, hazard_headers)
    save_report(
        args.output_dir / "comparison_report.md",
        runs,
        summary_rows,
        agreement_rows,
        disagreement_rows,
        args.output_dir,
    )

    logging.info("Done. Comparison outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
