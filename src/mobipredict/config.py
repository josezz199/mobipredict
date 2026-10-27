"""Configuración central de MobiPredict (datos reales).

Todas las rutas, URLs de las fuentes oficiales, ventanas temporales y parámetros
viven aquí para que cada módulo (ingesta, EDA, features, entrenamiento, API) use
exactamente los mismos valores. Se pueden sobrescribir con variables de entorno.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(os.getenv("MOBIPREDICT_HOME", Path(__file__).resolve().parents[2]))

DATA_RAW = ROOT / "data" / "raw"
DATA_REFERENCE = ROOT / "data" / "reference"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
EDA_FIGURES = ROOT / "eda" / "figures"
EDA_REPORTS = ROOT / "eda" / "reports"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
MLRUNS_DIR = ROOT / "mlruns"

ALL_DIRS = [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, EDA_FIGURES, EDA_REPORTS, MODELS_DIR, DOCS_DIR, MLRUNS_DIR]

# --------------------------------------------------------------------------- #
# Fuentes oficiales (todas abiertas). Ver docs/fuentes_de_datos.md
# --------------------------------------------------------------------------- #
MEDATA = "https://medata.app.medellin.gov.co/sites/default/files/distribution"
EOD_API = os.getenv("MOBIPREDICT_EOD_API", "https://eod2025.metropol.gov.co/api/api")

FUENTES = {
    "velocidad_gt": {
        "url": f"{MEDATA}/1-023-25-000287/Velocidad_y_tiempo_de_viaje_GT.csv",
        "archivo": "medata_velocidad_tiempo_viaje_gt.csv",
        "titulo": "Velocidad y tiempo de viaje GT (SIMM, Secretaría de Movilidad de Medellín)",
        "ficha": "https://www.datos.gov.co/d/7t5n-3b3w",
    },
    "aforos": {
        "url": f"{MEDATA}/1-023-25-000301/Aforos_Vehiculares.csv",
        "archivo": "medata_aforos_vehiculares.csv",
        "titulo": "Aforos vehiculares (Secretaría de Movilidad de Medellín)",
        "ficha": "https://www.datos.gov.co/d/b9s9-jw7c",
    },
    "siniestros": {
        "url": "https://data.mendeley.com/public-files/datasets/r6g5dfnpgh/files/"
               "e992c888-5819-40c6-9538-41980ea1d429/file_downloaded",
        "archivo": "mendeley_siniestros_medellin_2008_2025.zip",
        "titulo": "Road-traffic incidents and fatalities in Medellín 2008–2025 (Mendeley Data, CC BY 4.0)",
        "ficha": "https://doi.org/10.17632/r6g5dfnpgh.1",
    },
    "eod2025": {
        "url": EOD_API,
        "archivo": "eod2025_indicadores.json",
        "titulo": "Encuesta Origen-Destino 2025 — API del visor oficial (AMVA)",
        "ficha": "https://eod2025.metropol.gov.co/",
    },
    "clima": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "archivo": "open_meteo_medellin_horario.csv",
        "titulo": "Open-Meteo Historical Weather API (reanálisis ERA5 / ERA5-Land)",
        "ficha": "https://open-meteo.com/en/docs/historical-weather-api",
    },
}
CLIMA_LAT, CLIMA_LON = 6.2442, -75.5812          # centro de Medellín
HTTP_TIMEOUT_S = float(os.getenv("MOBIPREDICT_HTTP_TIMEOUT", "120"))

MACROZONAS_FILE = DATA_REFERENCE / "eod2025_macrozonas.csv"
CORREDORES_FILE = DATA_REFERENCE / "corredores_simm.csv"
MANIFEST_FILE = DATA_RAW / "manifest.json"

# Tablas estandarizadas (salida de fuentes.py)
SENSORES_STD = DATA_INTERIM / "sensores_velocidad_horaria.parquet"
TRAMOS_STD = DATA_INTERIM / "tramos.csv"
CLIMA_STD = DATA_INTERIM / "clima_horario.parquet"
FESTIVOS_STD = DATA_INTERIM / "festivos_colombia.csv"
EOD_HORA_STD = DATA_INTERIM / "eod2025_viajes_por_hora.csv"
EOD_OD_STD = DATA_INTERIM / "eod2025_matriz_od_macrozonas.csv"
SINIESTROS_STD = DATA_INTERIM / "siniestros_2008_2025.parquet"
AFOROS_STD = DATA_INTERIM / "aforos_composicion.parquet"

# --------------------------------------------------------------------------- #
# Cifras publicadas usadas como referencia (no se recalculan con los datos)
# --------------------------------------------------------------------------- #
REFERENCIAS = {
    "muertes_viales_medellin_2025": {
        "valor": 276, "detalle": "162 motociclistas y 100 peatones; -11,8 % frente a 2024",
        "fuente": "Observatorio de Movilidad de Medellín, vía El Colombiano (8 ene 2026)",
        "url": "https://www.elcolombiano.com/medellin/balance-de-accidentalidad-en-medellin-el-ano-pasado-PF32374139"},
    "variacion_tenencia_motos_2017_2025_pct": {
        "valor": 48.0, "fuente": "Resultados EOD 2025 (AMVA), vía El Colombiano (2 may 2026)",
        "url": "https://www.elcolombiano.com/medellin/movilidad-habitantes-medellin-aburra-disminucion-teletrabajo-DD36063145"},
    "variacion_participacion_tp_2017_2025_pct": {
        "valor": -14.0, "fuente": "Resultados EOD 2025 (AMVA), vía El Colombiano (2 may 2026)",
        "url": "https://www.elcolombiano.com/medellin/movilidad-habitantes-medellin-aburra-disminucion-teletrabajo-DD36063145"},
}

# --------------------------------------------------------------------------- #
# Ventanas temporales del modelado (serie de velocidad: feb-2017 a dic-2020)
# --------------------------------------------------------------------------- #
FECHA_INICIO = "2017-02-01"
CORTE_VALIDACION = os.getenv("MOBIPREDICT_CORTE_VALID", "2019-09-01")   # train < corte
CORTE_TEST = os.getenv("MOBIPREDICT_CORTE_TEST", "2019-12-01")          # valid < corte <= test
FIN_TEST = "2020-03-20"            # inicio de las restricciones por COVID-19 en Medellín
FIN_DRIFT = "2021-01-01"           # 2020-03-20..2020-12-31 = periodo de drift (pandemia)

# --------------------------------------------------------------------------- #
# Modelado
# --------------------------------------------------------------------------- #
SEED = int(os.getenv("MOBIPREDICT_SEED", "42"))
FAST = os.getenv("MOBIPREDICT_FAST", "0") == "1"

FREQ_MIN = 60                    # resolución de la serie (horaria, 6:00-20:00)
HORIZON_STEPS = 1                # predicción a 1 hora
LAGS = [1, 2, 3]                 # rezagos en horas
MIN_HISTORY = max(LAGS) + 1      # lecturas horarias consecutivas para inferir

PCT_VEL_LIBRE = 0.95             # velocidad de referencia = percentil 95 del tramo (solo entrenamiento)
COLLAPSE_SPEED_RATIO = float(os.getenv("MOBIPREDICT_UMBRAL_COLAPSO", "0.50"))  # colapso: v < 50 % de la referencia

N_TRIALS_OPTUNA = int(os.getenv("MOBIPREDICT_TRIALS", "8" if FAST else "25"))
N_ITER_RANDOM = int(os.getenv("MOBIPREDICT_RANDOM_ITER", "4" if FAST else "10"))
SEARCH_SAMPLE = 25_000 if FAST else 50_000     # filas para la búsqueda de hiperparámetros
FIT_MAX_ROWS = 80_000 if FAST else 200_000     # tope de filas para RF y MLP (los más recientes)

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
# Backend de archivos en mlruns/ (requisito del proyecto). MLflow 3.x lo exige habilitar.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.resolve().as_uri())
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "MobiPredict")

# --------------------------------------------------------------------------- #
# Artefactos
# --------------------------------------------------------------------------- #
MODEL_SPEED_FILE = MODELS_DIR / "modelo_velocidad.joblib"
MODEL_COLLAPSE_FILE = MODELS_DIR / "modelo_colapso.joblib"
FEATURE_STORE_FILE = MODELS_DIR / "feature_store.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"
COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"
ESCENARIOS_FILE = MODELS_DIR / "escenarios.json"       # momentos reales del periodo de prueba (frontend)
FRONTEND_DIR = ROOT / "frontend"
DECISION_FILE = DOCS_DIR / "model_decision.md"


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
