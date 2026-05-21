# Reproducibility

This repo is now organized around an installable Python package in
`src/groundsource`. That means scripts import the same code whether you run them
from the repo root, from Docker, or from a fresh virtual environment.

## Local Environment

Recommended baseline:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt -c constraints.txt
```

`requirements.txt` installs the package from `pyproject.toml`. Use
`requirements-dev.txt` when you also want the local test tools. Add
`-c constraints.txt` to pin the direct dependencies to the known-good versions
captured from this workspace.

## Environment Variables

Start from:

```bash
cp .env.example .env
```

Required values are:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `API_VERSION`
- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_TOKEN`

Optional but commonly useful:

- `AZURE_OPENAI_REASONING_EFFORT`
- `DATABRICKS_SOURCE_ASSET_TABLE`
- `DATABRICKS_SOURCE_CHUNK_TABLE`
- `DATABRICKS_CA_FILE`
- `LOG_LEVEL`

## Data And Generated Artifacts

Tracked inputs:

- `data/` contains small reference geodata.
- `eval/` contains the evaluation runbook, gold labels, and scoring inputs.

Ignored local artifacts:

- `outputs/` for extraction runs, geometries, maps, and reports.
- `.nominatim_cache.json` for geocoder responses.
- `.env` and `*.pem` for secrets and private certificate chains.

## Determinism

The extraction step depends on Azure OpenAI and Databricks state, so full
bit-for-bit reproducibility requires recording the model deployment, reasoning
effort, prompt ID, source table snapshot, and output directory. The generated
output directory names from `--all` include model, effort, and prompt ID.

Sampling helpers use explicit seeds where sampling is involved. For example,
`scripts/make_polygon_validation_sample.py` defaults to `--seed 42`.

## Smoke Checks

Run these before scaling a batch:

```bash
make test
groundsource-extract --db-only --limit 3
groundsource-extract --limit 3 --output-dir outputs/smoke
groundsource-polygons \
  --input outputs/smoke/event_extractions.jsonl \
  --output outputs/smoke/event_geometries.parquet \
  --coastline data/ne_50m_coastline.zip \
  --user-agent "undrr-groundsource/0.1 (you@example.org)" \
  --limit 5
```
