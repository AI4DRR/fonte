FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs needed by geopandas/shapely/pyogrio and TLS for Databricks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgeos-dev \
        libproj-dev \
        proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package first so console commands are available in the image.
COPY pyproject.toml README.md requirements.txt constraints.txt ./
COPY src ./src
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt -c constraints.txt

# Copy runnable helpers and small reference data. Large/regenerable artifacts
# are excluded via .dockerignore and mounted from the host when needed.
COPY scripts ./scripts
COPY eval ./eval
COPY data ./data

# Output and cache directories are mounted as volumes from the host in
# docker-compose.yml so artifacts survive container restarts.
RUN mkdir -p /app/outputs

EXPOSE 8501

# Default: run the main extraction pipeline. Override at `docker run` /
# `docker compose run` time to run polygon resolution or the Streamlit app.
CMD ["groundsource-extract", "--limit", "5"]
