.PHONY: install datos pipeline fast api test notebook docker-up docker-down mlflow-ui
install:      ; pip install -r requirements-dev.txt
datos:        ; python scripts/descargar_datos.py
pipeline:     ; python run_pipeline.py
fast:         ; python run_pipeline.py --fast
api:          ; uvicorn api.main:app --reload --port 8000
test:         ; pytest -q tests
mlflow-ui:    ; MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns --port 5000
notebook:     ; jupyter lab notebooks/MobiPredict_MLOps_end_to_end.ipynb
docker-up:    ; docker compose up --build -d
docker-down:  ; docker compose down
