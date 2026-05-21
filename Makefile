.PHONY: setup test extract db-check polygons review docker-build

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
	$(VENV_BIN)/streamlit run eval/gold_review_app.py

docker-build:
	docker build -t undrr-groundsource .
