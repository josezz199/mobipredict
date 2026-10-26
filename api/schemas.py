"""Esquemas de entrada y salida de la API (Pydantic)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Lecturas horarias reales del tramo AVENIDA_80_NS (MEData/SIMM). Se reemplazan al correr el pipeline
# por un ejemplo tomado del conjunto de prueba (ver models/ejemplo_api.json).
EJEMPLO_TRAMO = "AVENIDA_80_NS"
EJEMPLO_OBS = [
    {"timestamp": "2020-02-11T14:00:00", "velocidad_kmh": 24.0, "lluvia_mm": 0.0},
    {"timestamp": "2020-02-11T15:00:00", "velocidad_kmh": 22.0, "lluvia_mm": 0.0},
    {"timestamp": "2020-02-11T16:00:00", "velocidad_kmh": 19.0, "lluvia_mm": 0.4},
    {"timestamp": "2020-02-11T17:00:00", "velocidad_kmh": 15.0, "lluvia_mm": 1.2},
]


class Observacion(BaseModel):
    """Lectura horaria de un tramo vial."""
    timestamp: datetime = Field(..., description="Inicio de la hora (hora local de Medellín)")
    velocidad_kmh: float = Field(..., gt=0, le=130, description="Velocidad media del tramo en esa hora (km/h)")
    lluvia_mm: float = Field(0.0, ge=0, description="Precipitación en esa hora (mm)")


class PredictRequest(BaseModel):
    tramo_id: str = Field(..., examples=[EJEMPLO_TRAMO], description="Ver GET /tramos")
    observaciones: list[Observacion] = Field(
        ..., min_length=4, description="Últimas lecturas horarias consecutivas (>= 4, la más reciente al final)",
        examples=[EJEMPLO_OBS])


class PredictResponse(BaseModel):
    tramo_id: str
    corredor: str
    sentido: str
    timestamp_prediccion: datetime
    horizonte_min: int
    velocidad_predicha_kmh: float
    tiempo_viaje_predicho_min: float
    velocidad_referencia_kmh: float
    probabilidad_colapso: float
    alerta_colapso: bool
    nivel: Literal["normal", "precaucion", "colapso"]
    umbral_decision: float
    modelo_velocidad: str
    modelo_colapso: str


class PredictBatchRequest(BaseModel):
    peticiones: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class PredictBatchResponse(BaseModel):
    predicciones: list[PredictResponse]
    alertas: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degradado"]
    modelos_cargados: bool
    modelo_velocidad: str | None = None
    modelo_colapso: str | None = None
    horizonte_min: int | None = None
    entrenado_utc: str | None = None
    detalle: str | None = None
