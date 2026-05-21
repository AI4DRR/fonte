# UNDRR Groundsource Starter

Prototype pipeline for extracting disaster-event records from Databricks text,
running prompt experiments, and resolving affected locations to Groundsource-style
geometries.

## Repository Layout

```text
src/groundsource/   installable extraction and polygon-resolution package
scripts/            one-off QA, comparison, and validation helpers
eval/               prompt-evaluation runbook, gold labels, and scoring tools
tests/              lightweight local tests
data/               small reference geodata used by polygon post-processing
outputs/            generated runs and maps; gitignored
```

The main entry points are installed as console commands:

- `groundsource-extract` runs the Databricks to Azure OpenAI extraction pipeline.
- `groundsource-polygons` resolves extracted locations to geometries.

## Setup

Use Python 3.11 when possible; the Docker image uses Python 3.11. Python 3.10+
should also work.

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt -c constraints.txt
```

Fill `.env` with your Azure OpenAI and Databricks values. Keep `.env`,
`*.pem`, `outputs/`, and `.nominatim_cache.json` out of Git.

## Common Commands

```bash
# Check local tests.
make test

# Check Databricks connectivity without calling Azure OpenAI.
groundsource-extract --db-only --limit 3

# Run a small extraction.
groundsource-extract --limit 5 --output-dir outputs/event_extractions

# Resolve polygons from an extraction JSONL.
groundsource-polygons \
  --input outputs/event_extractions/event_extractions.jsonl \
  --output outputs/event_extractions/event_geometries.parquet \
  --coastline data/ne_50m_coastline.zip \
  --user-agent "undrr-groundsource/0.1 (you@example.org)"

# Open the gold-review UI.
streamlit run eval/gold_review_app.py
```

You can also use `make setup`, `make extract`, `make polygons`, and
`make review` for the same workflows.

## Docker

Docker gives you a clean Python 3.11 runtime with the same package entry points.

```bash
docker compose build
docker compose run --rm extractor groundsource-extract --db-only --limit 3
docker compose run --rm extractor groundsource-extract --limit 5
```

Polygon resolution from Docker:

```bash
docker compose run --rm extractor groundsource-polygons \
  --input outputs/event_extractions/event_extractions.jsonl \
  --output outputs/event_extractions/event_geometries.parquet \
  --coastline data/ne_50m_coastline.zip \
  --user-agent "undrr-groundsource/0.1 (you@example.org)"
```

Start the review UI:

```bash
docker compose up review
```

## Prompt And Evaluation Workflows

Edit prompt variants in `src/groundsource/prompts.py`. The extraction schema
lives in `src/groundsource/app_event_focus.py`.

Prompt-comparison and gold-labelling steps live in `eval/README.md`.

## Reproducibility Notes

Generated outputs go under `outputs/` and are intentionally ignored by Git.
The geocoder cache `.nominatim_cache.json` is also ignored because it can become
large and environment-specific.

For a fresh machine, prefer this order:

1. Install with `requirements-dev.txt -c constraints.txt`.
2. Copy `.env.example` to `.env`.
3. Run `make test`.
4. Run `groundsource-extract --db-only --limit 3`.
5. Run a tiny extraction and polygon pass before scaling up.

More details are in `docs/reproducibility.md`.

## Pushing Changes To GitHub

Review and commit locally:

```bash
git status
git add .
git commit -m "Organize project structure and reproducibility setup"
```

If the repository already has a GitHub remote:

```bash
git remote -v
git push
```

If it does not have a remote yet, create an empty GitHub repo, then run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Never push `.env`, Databricks tokens, Azure OpenAI keys, or private certificate
files. If a secret ever gets committed, rotate it immediately.
