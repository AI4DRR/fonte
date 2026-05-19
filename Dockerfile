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

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source. Anything large/regenerable is excluded via .dockerignore.
COPY . .

# Output and cache directories are mounted as volumes from the host in
# docker-compose.yml so artifacts survive container restarts.
RUN mkdir -p /app/outputs

EXPOSE 8501

# Default: run the main extraction pipeline. Override at `docker run` /
# `docker compose run` time to run polygon resolution or the Streamlit app.
CMD ["python", "app_event_focus.py", "--limit", "5"]
