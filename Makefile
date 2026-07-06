.PHONY: setup test extract db-check polygons review docker-build prefill recommend polygon-score \
	gold-extract gold-polygons gold-prefill-validation gold-validate gold-validation-score

PYTHON ?= python3
VENV ?= .venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt -c constraints.txt

test:
	$(VENV_PYTHON) -m unittest discover -s tests -p "test_*.py"

db-check:
	$(VENV_BIN)/groundsource-extract --db-only --limit 3

extract:
	$(VENV_BIN)/groundsource-extract --limit 5

polygons:
	$(VENV_BIN)/groundsource-polygons \
		--input outputs/event_extractions/event_extractions.jsonl \
		--output outputs/event_extractions/event_geometries.parquet \
		--coastline data/ne_50m_coastline.zip \
		--user-agent "$$NOMINATIM_USER_AGENT"

review:
	$(VENV_PYTHON) -m streamlit run eval/gold_review_app.py

# Auto pre-fill gold labels (consensus) and polygon review queue (risk-ranked).
prefill:
	$(VENV_PYTHON) eval/prefill_gold.py
	$(VENV_PYTHON) eval/prefill_polygon_review.py

# After gold.csv is verified: score the 8 runs and emit the recommendation.
recommend:
	$(VENV_PYTHON) eval/score_against_gold.py
	$(VENV_PYTHON) eval/recommend.py

# After the polygon queue is judged: accuracy overall + by stratum.
polygon-score:
	$(VENV_PYTHON) eval/score_polygon_validation.py

# ---- Gold run (prompt 5, 2000 docs) + easier validation ----------------------

# 1. Extract the gold dataset with the production prompt.
# Tunable from the command line, e.g.  make gold-extract WORKERS=2 MODEL=gpt-5
GOLD_MODEL ?= gpt-5
GOLD_WORKERS ?= 4
GOLD_REASONING ?= low
GOLD_LIMIT ?= 2000
gold-extract:
	$(VENV_BIN)/groundsource-extract \
		--prompt 5 \
		--limit $(GOLD_LIMIT) \
		--workers $(GOLD_WORKERS) \
		--model $(GOLD_MODEL) \
		--reasoning-effort $(GOLD_REASONING) \
		--output-dir outputs/gold \
		--resume

# 1b. Resolve polygons (geometries) for the gold extraction.
gold-polygons:
	$(VENV_BIN)/groundsource-polygons \
		--input outputs/gold/event_extractions.jsonl \
		--output outputs/gold/event_geometries.parquet \
		--coastline data/ne_50m_coastline.zip \
		--cache outputs/gold/.nominatim_cache.json \
		--user-agent "$$NOMINATIM_USER_AGENT"

# 2. Build the ~100-event stratified, risk-ranked validation queue.
gold-prefill-validation:
	$(VENV_PYTHON) eval/prefill_gold_validation.py

# 3. Judge the queue (Correct / Partial / Wrong) in the easier review app.
gold-validate:
	$(VENV_PYTHON) -m streamlit run eval/gold_validation_app.py

# 4. Score accuracy overall + by hazard stratum.
gold-validation-score:
	$(VENV_PYTHON) eval/score_gold_validation.py

docker-build:
	docker build -t undrr-groundsource .
