"""API de predicción de MobiPredict (FastAPI).

Endpoints:
  GET  /                  redirige al frontend web (/app/)
  GET  /health            estado del servicio y de los modelos cargados
  GET  /tramos            catálogo de corredores disponibles
  GET  /escenarios        momentos reales del periodo de prueba (para el frontend)
  GET  /escenarios/{id}   lecturas de todos los tramos en ese momento + lo que pasó en la hora siguiente
  GET  /modelos           modelos ganadores y comparación de todos los candidatos
  POST /predict           velocidad a 1 hora + probabilidad y alerta de colapso para un tramo
  POST /predict/batch     varias peticiones en una sola llamada

Arranque local:  uvicorn api.main:app --reload --port 8000
Frontend:        http://localhost:8000/
Documentación interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobipredict import config  # noqa: E402
from mobipredict.features import features_inferencia  # noqa: E402

from .schemas import (HealthResponse, PredictBatchRequest, PredictBatchResponse,  # noqa: E402
                      PredictRequest, PredictResponse)

ESTADO: dict = {"cargado": False}


def cargar_modelos() -> None:
    """Carga el modelo ganador de cada tarea, el feature store y la metadata."""
    faltan = [p for p in [config.MODEL_SPEED_FILE, config.MODEL_COLLAPSE_FILE,
                          config.FEATURE_STORE_FILE, config.METADATA_FILE] if not p.exists()]
    if faltan:
        ESTADO.update(cargado=False, error=f"Faltan artefactos: {[p.name for p in faltan]}. "
                                           "Ejecute `python run_pipeline.py` primero.")
        return
    ESTADO.update(
        cargado=True, error=None,
        modelo_velocidad=joblib.load(config.MODEL_SPEED_FILE),
        modelo_colapso=joblib.load(config.MODEL_COLLAPSE_FILE),
        store=joblib.load(config.FEATURE_STORE_FILE),
        meta=json.loads(config.METADATA_FILE.read_text(encoding="utf-8")),
        escenarios=(json.loads(config.ESCENARIOS_FILE.read_text(encoding="utf-8"))
                    if config.ESCENARIOS_FILE.exists() else []),
        comparacion=(pd.read_csv(config.COMPARISON_FILE) if config.COMPARISON_FILE.exists() else None),
        cargado_en=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    cargar_modelos()
    yield


app = FastAPI(
    title="MobiPredict API",
    description="Pronóstico de velocidad por corredor y alertas tempranas de colapso en Medellín "
                "(modelos entrenados con datos abiertos del SIMM, la EOD 2025 y Open-Meteo).",
    version="1.0.0",
    lifespan=lifespan,
)

# Frontend web y figuras del EDA (archivos estáticos servidos por la misma API)
if config.FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
config.EDA_FIGURES.mkdir(parents=True, exist_ok=True)
app.mount("/figuras", StaticFiles(directory=config.EDA_FIGURES), name="figuras")


@app.get("/", include_in_schema=False)
def inicio() -> RedirectResponse:
    return RedirectResponse(url="/app/" if config.FRONTEND_DIR.exists() else "/docs")


def _requiere_modelos() -> None:
    if not ESTADO.get("cargado"):
        raise HTTPException(status_code=503, detail=ESTADO.get("error", "Modelos no cargados"))


def _nivel(prob: float, umbral: float) -> str:
    if prob >= umbral:
        return "colapso"
    if prob >= umbral * 0.6:
        return "precaucion"
    return "normal"


def _predecir(req: PredictRequest) -> PredictResponse:
    store, meta = ESTADO["store"], ESTADO["meta"]
    obs = [o.model_dump() for o in sorted(req.observaciones, key=lambda o: o.timestamp)]
    try:
        X = features_inferencia(req.tramo_id, obs, store)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    vel = max(float(ESTADO["modelo_velocidad"].predict(X)[0]), 1.0)
    prob = float(ESTADO["modelo_colapso"].predict_proba(X)[:, 1][0])
    umbral = float(meta["modelo_colapso"]["umbral_decision"])
    tramo = store["tramos"].set_index("tramo_id").loc[req.tramo_id]
    t_pred = pd.Timestamp(obs[-1]["timestamp"]).floor("h") + pd.Timedelta(minutes=meta["horizonte_min"])
    return PredictResponse(
        tramo_id=req.tramo_id,
        corredor=str(tramo["corredor"]),
        sentido=str(tramo["sentido"]),
        timestamp_prediccion=t_pred.to_pydatetime(),
        horizonte_min=meta["horizonte_min"],
        velocidad_predicha_kmh=round(vel, 2),
        tiempo_viaje_predicho_min=round(float(tramo["longitud_km"]) / vel * 60, 2),
        velocidad_referencia_kmh=round(float(tramo["vel_referencia_kmh"]), 1),
        probabilidad_colapso=round(prob, 4),
        alerta_colapso=prob >= umbral,
        nivel=_nivel(prob, umbral),
        umbral_decision=umbral,
        modelo_velocidad=meta["modelo_velocidad"]["nombre"],
        modelo_colapso=meta["modelo_colapso"]["nombre"],
    )


@app.get("/health", response_model=HealthResponse, tags=["estado"])
def health() -> HealthResponse:
    meta = ESTADO.get("meta", {})
    return HealthResponse(
        status="ok" if ESTADO.get("cargado") else "degradado",
        modelos_cargados=bool(ESTADO.get("cargado")),
        modelo_velocidad=meta.get("modelo_velocidad", {}).get("nombre"),
        modelo_colapso=meta.get("modelo_colapso", {}).get("nombre"),
        horizonte_min=meta.get("horizonte_min"),
        entrenado_utc=meta.get("generado_utc"),
        detalle=ESTADO.get("error"),
    )


@app.get("/tramos", tags=["catalogo"])
def tramos() -> list[dict]:
    _requiere_modelos()
    cols = ["tramo_id", "corredor", "sentido", "inicio", "fin", "longitud_km", "flujo_tipo", "macrozona",
            "vel_referencia_kmh"]
    t = ESTADO["store"]["tramos"][cols].copy()
    return t.astype(object).where(t.notna(), None).to_dict(orient="records")


@app.post("/predict", response_model=PredictResponse, tags=["prediccion"])
def predict(req: PredictRequest) -> PredictResponse:
    _requiere_modelos()
    return _predecir(req)


@app.post("/predict/batch", response_model=PredictBatchResponse, tags=["prediccion"])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    _requiere_modelos()
    preds = [_predecir(r) for r in req.peticiones]
    return PredictBatchResponse(predicciones=preds, alertas=sum(p.alerta_colapso for p in preds))


@app.get("/escenarios", tags=["frontend"])
def escenarios() -> list[dict]:
    """Momentos reales del periodo de prueba (y uno de la pandemia) disponibles para simular."""
    _requiere_modelos()
    return [{k: e[k] for k in ["id", "nombre", "timestamp", "timestamp_prediccion", "periodo"]}
            | {"tramos": len(e["tramos"]), "colapsos_reales": sum(t["colapso_real"] for t in e["tramos"])}
            for e in ESTADO["escenarios"]]


@app.get("/escenarios/{escenario_id}", tags=["frontend"])
def escenario(escenario_id: str) -> dict:
    """Lecturas reales de cada tramo hasta la hora t y la velocidad medida en t + 1 h."""
    _requiere_modelos()
    for e in ESTADO["escenarios"]:
        if e["id"] == escenario_id:
            return e
    raise HTTPException(status_code=404, detail=f"Escenario desconocido: {escenario_id}")


@app.get("/modelos", tags=["frontend"])
def modelos() -> dict:
    """Modelos ganadores, reglas de decisión y métricas de todos los candidatos."""
    _requiere_modelos()
    meta = ESTADO["meta"]
    filas = []
    comp = ESTADO.get("comparacion")
    if comp is not None:
        cols = [c for c in comp.columns if c not in ("params", "run_id")]
        filas = json.loads(comp[cols].to_json(orient="records"))   # NaN -> null, tipos nativos
    return {"generado_utc": meta.get("generado_utc"), "horizonte_min": meta.get("horizonte_min"),
            "umbral_colapso_ratio_vel": meta.get("umbral_colapso_ratio_vel"),
            "modelo_velocidad": meta.get("modelo_velocidad"), "modelo_colapso": meta.get("modelo_colapso"),
            "n_features": len(meta.get("features", [])), "comparacion": filas,
            "figuras": sorted(p.name for p in config.EDA_FIGURES.glob("*.png"))}


@app.post("/admin/reload", tags=["estado"])
def reload() -> dict:
    """Recarga los modelos desde disco (tras un reentrenamiento)."""
    cargar_modelos()
    return {"recargado": ESTADO.get("cargado"), "detalle": ESTADO.get("error")}
