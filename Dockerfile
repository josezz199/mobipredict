# Imagen única para el pipeline (Prefect + MLflow) y la API (FastAPI)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    MOBIPREDICT_HOME=/app \
    MLFLOW_ALLOW_FILE_STORE=true

# libgomp1: requerido por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY tests/ tests/
COPY run_pipeline.py .
COPY data/reference/ data/reference/
RUN mkdir -p data/raw data/interim data/processed eda/figures eda/reports models docs mlruns \
    && useradd --create-home --uid 1000 mobi && chown -R mobi:mobi /app
USER mobi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
