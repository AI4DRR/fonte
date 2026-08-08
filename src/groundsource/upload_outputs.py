"""
groundsource.upload_outputs
----------------------------

Standalone uploader: recursively pushes every file under a local output
directory (default: $EXTRACT_OUTPUT_DIR) to Azure Blob Storage, preserving
the relative directory structure beneath a configurable blob prefix.

Uses the container-scoped SAS token in STORAGEACCOUNT_TOKEN directly against
the Azure Blob REST API (Put Blob) via `requests`, so no Azure SDK dependency
is required.

Usage
-----

    groundsource-upload
    groundsource-upload --source outputs/gold

Required environment variables (see .env):

    STORAGEACCOUNT_ENDPOINT           e.g. https://myaccount.blob.core.windows.net
    STORAGEACCOUNT_TOKEN              container-scoped SAS token (query string, no leading '?')
    STORAGEACCOUNT_OUTPUT_CONTAINER   e.g. groundsource-output
    STORAGEACCOUNT_OUTPUT_PREFIX      e.g. demo
    EXTRACT_OUTPUT_DIR                local dir to upload; only required when --source is omitted
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

LOG = logging.getLogger("upload_outputs")

REQUIRED_ENV_VARS = (
    "STORAGEACCOUNT_ENDPOINT",
    "STORAGEACCOUNT_TOKEN",
    "STORAGEACCOUNT_OUTPUT_CONTAINER",
    "STORAGEACCOUNT_OUTPUT_PREFIX",
)

# Azure Storage REST API version compatible with the `sv=` param on modern
# container SAS tokens. Required on every Put Blob request.
BLOB_API_VERSION = "2021-08-06"


def require_env_vars(names: tuple) -> dict:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Check your .env file."
        )
    return {name: os.getenv(name) for name in names}


def iter_files(source: Path) -> List[Path]:
    return sorted(p for p in source.rglob("*") if p.is_file())


def signed_url(*, endpoint: str, container: str, blob_path: str, sas_token: str) -> str:
    return f"{endpoint.rstrip('/')}/{container}/{blob_path}?{sas_token}"


def upload_file(
    *,
    local_path: Path,
    blob_path: str,
    endpoint: str,
    container: str,
    sas_token: str,
) -> None:
    url = signed_url(
        endpoint=endpoint, container=container, blob_path=blob_path, sas_token=sas_token
    )
    with open(local_path, "rb") as fh:
        data = fh.read()
    response = requests.put(
        url,
        data=data,
        headers={
            "x-ms-blob-type": "BlockBlob",
            "x-ms-version": BLOB_API_VERSION,
            "Content-Length": str(len(data)),
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(
            f"Upload failed for {local_path} -> {blob_path}: "
            f"HTTP {response.status_code} {response.text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local output files to Azure Blob Storage."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Local directory to upload recursively. Defaults to $EXTRACT_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO, or $LOG_LEVEL).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    required = REQUIRED_ENV_VARS if args.source is not None else REQUIRED_ENV_VARS + ("EXTRACT_OUTPUT_DIR",)
    try:
        env = require_env_vars(required)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 1

    source = args.source if args.source is not None else Path(env["EXTRACT_OUTPUT_DIR"])
    if not source.is_dir():
        LOG.error("Source directory not found: %s", source)
        return 1

    files = iter_files(source)
    if not files:
        LOG.warning("No files found under %s; nothing to upload.", source)
        return 0

    endpoint = env["STORAGEACCOUNT_ENDPOINT"]
    container = env["STORAGEACCOUNT_OUTPUT_CONTAINER"]
    prefix = env["STORAGEACCOUNT_OUTPUT_PREFIX"].strip("/")
    sas_token = env["STORAGEACCOUNT_TOKEN"]

    uploaded = 0
    uploaded_blobs: List[tuple] = []
    for local_path in files:
        relative = local_path.relative_to(source).as_posix()
        blob_path = f"{prefix}/{relative}" if prefix else relative
        LOG.info("Uploading %s -> %s/%s", local_path, container, blob_path)
        upload_file(
            local_path=local_path,
            blob_path=blob_path,
            endpoint=endpoint,
            container=container,
            sas_token=sas_token,
        )
        uploaded += 1
        uploaded_blobs.append((relative, blob_path))

    LOG.info(
        "Uploaded %d file(s) from %s to %s/%s/",
        uploaded,
        source,
        container,
        prefix,
    )

    report_lines = [
        "",
        "Upload complete.",
        "",
        f"Uploaded {uploaded} file(s) from {source} to {container}/{prefix}/",
        "",
        "Validation URLs:",
        "",
    ]
    url_blocks = [
        f"{relative}\n"
        + signed_url(
            endpoint=endpoint, container=container, blob_path=blob_path, sas_token=sas_token
        )
        for relative, blob_path in uploaded_blobs
    ]
    report_lines.append("\n\n".join(url_blocks))
    print("\n".join(report_lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
