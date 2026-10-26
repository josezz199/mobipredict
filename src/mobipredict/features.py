"""Feature engineering espaciotemporal + demográfico y split temporal (datos reales).

Serie base: velocidad horaria (6:00-20:00) por tramo vial, MEData/SIMM 2017-2020.

Las MISMAS funciones construyen las features para entrenar (por lotes) y para la API
(las últimas lecturas de un tramo), así no hay training-serving skew.

"Feature store" local (models/feature_store.joblib):
  * tramos:  catálogo (corredor, sentido, longitud, tipo de flujo, ubicación, macrozona EOD)
             + velocidad de referencia (p95 del entrenamiento).
  * perfil:  velocidad media por tramo x tipo de día x hora (solo entrenamiento).
  * eod_hora: distribución horaria oficial de viajes EOD 2025 por grupo modal.
  * eod_mz:  viajes generados/atraídos por macrozona (matriz OD EOD 2025).
  * clima:   lluvia horaria (Open-Meteo); en la API se recibe con cada lectura.
  * festivos: calendario oficial de Colombia.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from . import config, fuentes

log = logging.getLogger(__name__)

TARGET_REG = "y_velocidad_1h"
TARGET_CLF = "y_colapso_1h"
PASO = pd.Timedelta(minutes=config.FREQ_MIN)


# --------------------------------------------------------------------------- #
# Tablas de referencia
# --------------------------------------------------------------------------- #
def _tipo_dia(ts: pd.Series, festivo: pd.Series) -> np.ndarray:
    dow = ts.dt.dayofweek
    return np.select([(festivo == 1) | (dow == 6), dow == 5], ["domingo_festivo", "sabado"], "laboral")


def perfil_historico(d: pd.DataFrame) -> pd.DataFrame:
    return (d.groupby(["tramo_id", "tipo_dia", "hora"])["velocidad_kmh"].mean()
            .rename("perfil_velocidad").reset_index())


def construir_store(sens: pd.DataFrame, tramos: pd.DataFrame, festivos: pd.DataFrame,
                    eod_hora: pd.DataFrame, od: pd.DataFrame, clima: pd.DataFrame, corte_train) -> dict:
    """Calcula las tablas de referencia usando SOLO datos anteriores al corte de entrenamiento."""
    train = sens[sens["timestamp"] < corte_train]
    vref = (train.groupby("tramo_id")["velocidad_kmh"].quantile(config.PCT_VEL_LIBRE)
            .rename("vel_referencia_kmh").reset_index())
    tr = tramos.merge(vref, on="tramo_id", how="left")
    base = _base(train, {"festivos": festivos})
    return {
        "tramos": tr,
        "perfil": perfil_historico(base),
        "eod_hora": eod_hora,
        "eod_mz": fuentes.priors_macrozona(od),
        "clima": clima[["timestamp", "lluvia_mm"]],
        "festivos": set(pd.to_datetime(festivos["fecha"]).dt.normalize()),
    }


# --------------------------------------------------------------------------- #
# Construcción de features (compartida entrenamiento / inferencia)
# --------------------------------------------------------------------------- #
def _base(d: pd.DataFrame, store: dict) -> pd.DataFrame:
    d = d.copy()
    fest = store["festivos"]
    if not isinstance(fest, set):
        fest = set(pd.to_datetime(fest["fecha"]).dt.normalize())
    d["es_festivo"] = d["timestamp"].dt.normalize().isin(fest).astype(int)
    d["tipo_dia"] = _tipo_dia(d["timestamp"], d["es_festivo"])
    d["hora"] = d["timestamp"].dt.hour
    return d


def _lag_seguro(d: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Rezago de k horas solo si la lectura previa está exactamente k horas antes (sin cruzar noches)."""
    g = d.groupby("tramo_id", sort=False)
    val, ts = g[col].shift(k), g["timestamp"].shift(k)
    return val.where(d["timestamp"] - ts == k * PASO)


def construir_features(d: pd.DataFrame, store: dict) -> pd.DataFrame:
    """`d` limpio y ordenado por tramo y tiempo: columnas timestamp, tramo_id, velocidad_kmh y,
    si existe, lluvia_mm (si no, se toma de la tabla de clima)."""
    d = _base(d.sort_values(["tramo_id", "timestamp"]), store)
    if "lluvia_mm" not in d.columns:
        d = d.merge(store["clima"], on="timestamp", how="left")
    d["lluvia_mm"] = d["lluvia_mm"].fillna(0)

    tr = store["tramos"][["tramo_id", "longitud_km", "flujo_continuo", "lat", "lon", "id_macrozona",
                          "vel_referencia_kmh"]]
    d = d.merge(tr, on="tramo_id", how="left")
    d["ratio_vel"] = d["velocidad_kmh"] / d["vel_referencia_kmh"]

    h = d["hora"]
    d["hora_sin"], d["hora_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    dow = d["timestamp"].dt.dayofweek
    d["dow_sin"], d["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d["es_fin_semana"] = (dow >= 5).astype(int)
    d["es_pico"] = h.isin([6, 7, 8, 17, 18, 19]).astype(int)
    d["mes_sin"] = np.sin(2 * np.pi * d["timestamp"].dt.month / 12)
    d["mes_cos"] = np.cos(2 * np.pi * d["timestamp"].dt.month / 12)

    for col in ["velocidad_kmh", "ratio_vel", "lluvia_mm"]:
        for k in config.LAGS:
            d[f"{col}_lag{k}"] = _lag_seguro(d, col, k)
    d["velocidad_media_3h"] = d[["velocidad_kmh", "velocidad_kmh_lag1", "velocidad_kmh_lag2"]].mean(axis=1)
    d["velocidad_delta1"] = d["velocidad_kmh"] - d["velocidad_kmh_lag1"]
    d["ratio_vel_delta1"] = d["ratio_vel"] - d["ratio_vel_lag1"]
    d["lluvia_3h"] = d[["lluvia_mm", "lluvia_mm_lag1", "lluvia_mm_lag2"]].sum(axis=1, min_count=1)
    d["n_lecturas_previas"] = d[[f"velocidad_kmh_lag{k}" for k in config.LAGS]].notna().sum(axis=1)

    # Perfil histórico y priors EOD en la hora OBJETIVO (t + 1 h)
    t_obj = d["timestamp"] + config.HORIZON_STEPS * PASO
    d["hora_obj"] = t_obj.dt.hour
    perfil = store["perfil"].rename(columns={"hora": "hora_obj"})
    d = d.merge(perfil, on=["tramo_id", "tipo_dia", "hora_obj"], how="left")
    d["desvio_perfil"] = d["velocidad_kmh"] - d["perfil_velocidad"]

    eh = store["eod_hora"][["hora", "share_viajes_privado", "share_viajes_publico", "share_viajes_total"]]
    d = d.merge(eh.rename(columns={"hora": "hora_obj", "share_viajes_privado": "eod_share_privado_hora",
                                   "share_viajes_publico": "eod_share_publico_hora",
                                   "share_viajes_total": "eod_share_total_hora"}), on="hora_obj", how="left")
    d = d.merge(store["eod_mz"], on="id_macrozona", how="left")
    return d


FEATURES_NUM = (
    ["hora_sin", "hora_cos", "dow_sin", "dow_cos", "mes_sin", "mes_cos", "es_fin_semana", "es_pico", "es_festivo",
     "velocidad_kmh", "ratio_vel", "velocidad_media_3h", "velocidad_delta1", "ratio_vel_delta1",
     "lluvia_mm", "lluvia_3h", "n_lecturas_previas",
     "longitud_km", "flujo_continuo", "lat", "lon", "vel_referencia_kmh",
     "perfil_velocidad", "desvio_perfil",
     "eod_share_privado_hora", "eod_share_publico_hora", "eod_share_total_hora",
     "eod_viajes_generados_mz", "eod_viajes_atraidos_mz"]
    + [f"{c}_lag{k}" for c in ["velocidad_kmh", "ratio_vel"] for k in config.LAGS]
)


def agregar_targets(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("tramo_id", sort=False)
    v_fut, ts_fut = g["velocidad_kmh"].shift(-config.HORIZON_STEPS), g["timestamp"].shift(-config.HORIZON_STEPS)
    ok = (ts_fut - d["timestamp"]) == config.HORIZON_STEPS * PASO
    d[TARGET_REG] = v_fut.where(ok)
    d[TARGET_CLF] = np.where(d[TARGET_REG].isna(), np.nan,
                             (d[TARGET_REG] / d["vel_referencia_kmh"] < config.COLLAPSE_SPEED_RATIO).astype(float))
    return d


# --------------------------------------------------------------------------- #
# Pipeline de features + split temporal
# --------------------------------------------------------------------------- #
def preparar_dataset() -> dict:
    config.ensure_dirs()
    sens = pd.read_parquet(config.SENSORES_STD)
    tramos = pd.read_csv(config.TRAMOS_STD)
    festivos = pd.read_csv(config.FESTIVOS_STD, parse_dates=["fecha"])
    eod_hora = pd.read_csv(config.EOD_HORA_STD)
    od = pd.read_csv(config.EOD_OD_STD)
    clima = pd.read_parquet(config.CLIMA_STD)

    c_valid, c_test = pd.Timestamp(config.CORTE_VALIDACION), pd.Timestamp(config.CORTE_TEST)
    fin_test, fin_drift = pd.Timestamp(config.FIN_TEST), pd.Timestamp(config.FIN_DRIFT)
    sens = sens[sens["timestamp"] >= pd.Timestamp(config.FECHA_INICIO)]

    store = construir_store(sens, tramos, festivos, eod_hora, od, clima, corte_train=c_valid)
    d = agregar_targets(construir_features(sens, store))
    d = d.dropna(subset=[TARGET_REG, TARGET_CLF])
    d[TARGET_CLF] = d[TARGET_CLF].astype(int)

    # Train < corte_valid <= valid < corte_test <= test < 2020-03-20 <= drift (pandemia)
    d["particion"] = np.select(
        [d["timestamp"] < c_valid, d["timestamp"] < c_test, d["timestamp"] < fin_test, d["timestamp"] < fin_drift],
        ["train", "valid", "test", "drift"], "fuera")
    # El objetivo de la última hora de cada partición cae en la siguiente: se descarta (sin fuga)
    t_obj = d["timestamp"] + config.HORIZON_STEPS * PASO
    frontera = (((d["timestamp"] < c_valid) & (t_obj >= c_valid)) | ((d["timestamp"] < c_test) & (t_obj >= c_test))
                | ((d["timestamp"] < fin_test) & (t_obj >= fin_test)))
    d = d[~frontera & (d["particion"] != "fuera")].reset_index(drop=True)

    cols = ["timestamp", "tramo_id", "particion"] + FEATURES_NUM + [TARGET_REG, TARGET_CLF]
    d = d[cols]
    for part in ["train", "valid", "test", "drift"]:
        d[d["particion"] == part].to_parquet(config.DATA_PROCESSED / f"{part}.parquet", index=False)

    joblib.dump(store, config.FEATURE_STORE_FILE)
    spec = {
        "fuente_sensores": config.FUENTES["velocidad_gt"]["titulo"],
        "features": FEATURES_NUM, "target_regresion": TARGET_REG, "target_clasificacion": TARGET_CLF,
        "horizonte_min": config.FREQ_MIN * config.HORIZON_STEPS, "lags_h": config.LAGS,
        "umbral_colapso_ratio_vel": config.COLLAPSE_SPEED_RATIO, "percentil_vel_referencia": config.PCT_VEL_LIBRE,
        "corte_validacion": str(c_valid.date()), "corte_test": str(c_test.date()), "fin_test": str(fin_test.date()),
        "ventana_drift": [str(fin_test.date()), "2020-12-31"],
        "filas": d["particion"].value_counts().to_dict(),
        "tasa_colapso": d.groupby("particion")[TARGET_CLF].mean().round(4).to_dict(),
        "tramos": int(d["tramo_id"].nunique()),
    }
    (config.DATA_PROCESSED / "feature_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                                                             encoding="utf-8")
    log.info("Dataset listo: %s | tasa de colapso: %s", spec["filas"], spec["tasa_colapso"])
    return spec


def cargar_particiones() -> dict[str, pd.DataFrame]:
    return {p: pd.read_parquet(config.DATA_PROCESSED / f"{p}.parquet") for p in ["train", "valid", "test", "drift"]}


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
def features_inferencia(tramo_id: str, observaciones: list[dict], store: dict) -> pd.DataFrame:
    """Fila de features para la última de las lecturas horarias recibidas de un tramo."""
    if tramo_id not in set(store["tramos"]["tramo_id"]):
        raise ValueError(f"tramo_id desconocido: {tramo_id}")
    if len(observaciones) < config.MIN_HISTORY:
        raise ValueError(f"Se requieren al menos {config.MIN_HISTORY} lecturas horarias; llegaron {len(observaciones)}")
    h = pd.DataFrame(observaciones)
    h["timestamp"] = pd.to_datetime(h["timestamp"]).dt.tz_localize(None).dt.floor("h")
    h["tramo_id"] = tramo_id
    if "lluvia_mm" not in h.columns:
        h["lluvia_mm"] = 0.0
    h = h.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return construir_features(h, store).iloc[[-1]][FEATURES_NUM]
