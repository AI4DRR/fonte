import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


LIST_COLUMNS = ["event_dates", "affected_locations", "key_impacts"]
META_SPLIT_COLUMNS = ["countries", "themes", "hazards", "organizations"]


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
        default=Path("outputs/test_1000/event_extractions.csv"),
        help="Path to event_extractions.csv or .jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/test_1000_checks"),
        help="Directory for plots and QA tables.",
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



def save_bar(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
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

    make_manual_samples(df, flagged, args.sample_size, args.output_dir)
    logging.info("Done. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
