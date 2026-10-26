"""Estandarización de las fuentes reales a tablas limpias en data/interim/.

Cada función lee el archivo crudo descargado en data/raw/, valida su estructura y
escribe una tabla con nombres de columna uniformes. El resto del proyecto (EDA,
features, API) solo lee estas tablas estandarizadas.
"""
from __future__ import annotations

import io
import json
import logging
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _slug_tramo(corredor: str, sentido: str) -> str:
    return f"{_norm(corredor).upper()}_{sentido.strip().upper()}"


# --------------------------------------------------------------------------- #
# Sensores: velocidad y tiempo de viaje por corredor (MEData / SIMM, Google Traffic)
# --------------------------------------------------------------------------- #
COLS_GT = ["id", "fecha_hora", "tv_s", "corredor_id", "dia", "mes", "anio", "hora", "fecha", "corredor",
           "flujo_tipo", "sentido", "inicio", "fin", "longitud_km", "tv_min", "velocidad_kmh", "tv_min_km",
           "diasem", "nombre_dia"]


def leer_velocidad_gt(ruta) -> pd.DataFrame:
    """Lee el CSV crudo de MEData y devuelve lecturas con timestamp confiable.

    Hallazgos de calidad (verificados sobre las 682.502 filas):
      * Las columnas DÍA y MES no son confiables en parte de 2020; FECHA_HORA (d/mm/aaaa H:MM)
        coincide con DIASEM en el 100 % de las filas, así que la fecha se toma de FECHA_HORA.
      * Desde 2020 hay lecturas cada 30 min; antes, cada hora. Se agregan al promedio horario.
      * ~9.700 claves (tramo, FECHA_HORA) están duplicadas; se promedian.
    """
    raw = pd.read_csv(ruta, encoding="utf-8", encoding_errors="replace", low_memory=False)
    if raw.shape[1] != len(COLS_GT):
        raise ValueError(f"Velocidad GT: se esperaban {len(COLS_GT)} columnas y llegaron {raw.shape[1]}")
    raw.columns = COLS_GT
    for c in ["velocidad_kmh", "tv_min", "longitud_km", "diasem"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["timestamp_raw"] = pd.to_datetime(raw["fecha_hora"].astype(str).str.strip(), format="%d/%m/%Y %H:%M",
                                          errors="coerce")
    raw = raw.dropna(subset=["timestamp_raw", "velocidad_kmh"])
    incoherentes = raw["timestamp_raw"].dt.dayofweek + 1 != raw["diasem"]
    if incoherentes.any():
        log.warning("Sensores: %d filas con FECHA_HORA incoherente con DIASEM; se descartan", incoherentes.sum())
        raw = raw[~incoherentes]
    raw["timestamp"] = raw["timestamp_raw"].dt.floor("h")
    return raw


def estandarizar_sensores(ruta=None, guardar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Serie horaria de velocidad por tramo (corredor + sentido) y catálogo de tramos."""
    raw = leer_velocidad_gt(ruta or config.DATA_RAW / config.FUENTES["velocidad_gt"]["archivo"])
    raw["corredor"] = raw["corredor"].str.strip()
    raw["sentido"] = raw["sentido"].str.strip().str.upper()
    raw["tramo_id"] = [_slug_tramo(c, s) for c, s in zip(raw["corredor"], raw["sentido"])]

    # Valores físicamente imposibles se descartan (velocidad urbana entre 2 y 110 km/h)
    n0 = len(raw)
    raw = raw[(raw["velocidad_kmh"] >= 2) & (raw["velocidad_kmh"] <= 110)]
    log.info("Sensores: %d filas descartadas por velocidad fuera de rango", n0 - len(raw))

    sens = (raw.groupby(["tramo_id", "timestamp"], as_index=False)
            .agg(velocidad_kmh=("velocidad_kmh", "mean"), tiempo_viaje_min=("tv_min", "mean"))
            .sort_values(["tramo_id", "timestamp"]))

    moda = lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]  # noqa: E731
    tramos = (raw.groupby("tramo_id")
              .agg(corredor=("corredor", moda), sentido=("sentido", moda), inicio=("inicio", moda),
                   fin=("fin", moda), longitud_km=("longitud_km", "median"), flujo_tipo=("flujo_tipo", moda),
                   registros=("velocidad_kmh", "size"))
              .reset_index())
    tramos["flujo_continuo"] = (tramos["flujo_tipo"].str.lower().str.startswith("contin")).astype(int)

    # Ubicación (cámaras SIMM) y macrozona EOD 2025 más cercana
    cor = pd.read_csv(config.CORREDORES_FILE)
    tramos = tramos.merge(cor[["corredor", "lat", "lon"]], on="corredor", how="left")
    mz = pd.read_csv(config.MACROZONAS_FILE).dropna(subset=["lat", "lon"])
    d = ((tramos["lat"].to_numpy()[:, None] - mz["lat"].to_numpy()[None, :]) ** 2
         + (tramos["lon"].to_numpy()[:, None] - mz["lon"].to_numpy()[None, :]) ** 2)
    idx = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
    tramos["id_macrozona"] = mz["id_macrozona"].to_numpy()[idx]
    tramos["macrozona"] = (mz["municipio"] + " - " + mz["macrozona"]).to_numpy()[idx]
    tramos.loc[tramos["lat"].isna(), ["id_macrozona", "macrozona"]] = [np.nan, None]

    if guardar:
        config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        sens.to_parquet(config.SENSORES_STD, index=False)
        tramos.to_csv(config.TRAMOS_STD, index=False)
    return sens, tramos


# --------------------------------------------------------------------------- #
# Clima y festivos
# --------------------------------------------------------------------------- #
def estandarizar_clima() -> pd.DataFrame:
    c = pd.read_csv(config.DATA_RAW / config.FUENTES["clima"]["archivo"], parse_dates=["timestamp"])
    c["lluvia_mm"] = c["lluvia_mm"].fillna(0).clip(lower=0)
    c.to_parquet(config.CLIMA_STD, index=False)
    return c


def estandarizar_festivos(anios=range(2017, 2027)) -> pd.DataFrame:
    import holidays
    f = pd.DataFrame([(pd.Timestamp(d), n) for d, n in holidays.Colombia(years=list(anios)).items()],
                     columns=["fecha", "festivo"]).sort_values("fecha")
    f.to_csv(config.FESTIVOS_STD, index=False)
    return f


# --------------------------------------------------------------------------- #
# EOD 2025 (agregados oficiales del visor del AMVA)
# --------------------------------------------------------------------------- #
GRUPOS_HORA = {"16": "informal", "17": "privado", "18": "publico", "19": "no_motorizado"}


def cargar_eod_json() -> dict:
    return json.loads((config.DATA_RAW / config.FUENTES["eod2025"]["archivo"]).read_text(encoding="utf-8"))


def estandarizar_eod() -> tuple[pd.DataFrame, pd.DataFrame]:
    eod = cargar_eod_json()["indicadores"]
    horas = pd.DataFrame({"hora": range(24)})
    for i, nombre in GRUPOS_HORA.items():
        g = pd.DataFrame(eod[i]["grupos"]).rename(columns={"criterio": "hora", "valor": f"viajes_{nombre}"})
        g["hora"] = g["hora"].astype(int)
        horas = horas.merge(g, on="hora", how="left")
    horas = horas.fillna(0)
    cols = [f"viajes_{n}" for n in GRUPOS_HORA.values()]
    horas["viajes_total"] = horas[cols].sum(axis=1)
    for c in cols + ["viajes_total"]:
        horas[f"share_{c}"] = horas[c] / horas[c].sum()
    horas.to_csv(config.EOD_HORA_STD, index=False)

    od = pd.DataFrame(eod["15"]["matriz"]).rename(columns={"agrupa_mz_origen": "origen", "agrupa_mz_destino": "destino",
                                                          "valor": "viajes"})
    od[["origen", "destino"]] = od[["origen", "destino"]].astype(int)
    od.to_csv(config.EOD_OD_STD, index=False)
    return horas, od


def priors_macrozona(od: pd.DataFrame) -> pd.DataFrame:
    gen = od.groupby("origen")["viajes"].sum().rename("eod_viajes_generados_mz")
    atr = od.groupby("destino")["viajes"].sum().rename("eod_viajes_atraidos_mz")
    p = pd.concat([gen, atr], axis=1).fillna(0)
    p.index.name = "id_macrozona"
    return p.reset_index()


# --------------------------------------------------------------------------- #
# Siniestralidad (Mendeley Data, 2008-2025)
# --------------------------------------------------------------------------- #
def _hora(x) -> float:
    if pd.isna(x):
        return np.nan
    if hasattr(x, "hour"):
        return x.hour
    if isinstance(x, (int, float)):
        return int((x % 1) * 24) if x < 1 else (int(x) if x < 24 else np.nan)
    m = re.match(r"\s*(\d{1,2})", str(x))
    return int(m.group(1)) if m and int(m.group(1)) < 24 else np.nan


def estandarizar_siniestros() -> pd.DataFrame | None:
    z = config.DATA_RAW / config.FUENTES["siniestros"]["archivo"]
    if not z.exists():
        log.warning("Siniestros: archivo no disponible; se omite.")
        return None
    if config.SINIESTROS_STD.exists() and config.SINIESTROS_STD.stat().st_mtime > z.stat().st_mtime:
        return pd.read_parquet(config.SINIESTROS_STD)
    with zipfile.ZipFile(z) as zf:
        nombre = next(n for n in zf.namelist() if n.endswith("Fatal_Road_Traffic.xlsx"))
        log.info("Siniestros: leyendo %s (puede tardar unos minutos)", nombre)
        df = pd.read_excel(io.BytesIO(zf.read(nombre)), sheet_name="BD", engine="openpyxl")
    df.columns = [_norm(c) for c in df.columns]
    col = lambda *ops: next((c for c in df.columns if any(o == c or o in c for o in ops)), None)  # noqa: E731
    out = pd.DataFrame({
        "anio": pd.to_numeric(df[col("ano", "anio")], errors="coerce"),
        "fecha": pd.to_datetime(df[col("fecha_incidente")], errors="coerce"),
        "hora": df[col("hora_incidente")].map(_hora),
        "clase": df[col("clase_incidente")].astype(str).str.strip().str.upper(),
        "gravedad": df[col("gravedad_incidente")].astype(str).str.strip().str.upper(),
        "lat": pd.to_numeric(df[col("latitud")], errors="coerce"),
        "lon": pd.to_numeric(df[col("longitud")], errors="coerce"),
        "comuna": df[col("comuna_planeacion")].astype(str) if col("comuna_planeacion") else None,
    })
    out["fatal"] = out["gravedad"].str.startswith("MUERT").astype(int)
    out.to_parquet(config.SINIESTROS_STD, index=False)
    return out


# --------------------------------------------------------------------------- #
# Aforos: composición vehicular por hora (2017-2019)
# --------------------------------------------------------------------------- #
def estandarizar_aforos() -> pd.DataFrame | None:
    p = config.DATA_RAW / config.FUENTES["aforos"]["archivo"]
    if not p.exists():
        return None
    a = pd.read_csv(p, encoding="latin-1", low_memory=False)
    if a.shape[1] < 24:
        log.warning("Aforos: estructura inesperada; se omite.")
        return None
    a = a.iloc[:, [0, 7, 8, 9, 10, 11, 20, 23]]
    a.columns = ["nodo", "autos", "buses", "camiones", "motos", "bicicletas", "anio", "hora"]
    for c in a.columns[1:]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["hora", "anio"])
    comp = a.groupby(["anio", "hora"], as_index=False)[["autos", "buses", "camiones", "motos", "bicicletas"]].sum()
    comp["nodos"] = a.groupby(["anio", "hora"])["nodo"].nunique().to_numpy()
    comp.to_parquet(config.AFOROS_STD, index=False)
    return comp


def estandarizar_todo() -> dict:
    config.ensure_dirs()
    sens, tramos = estandarizar_sensores()
    clima = estandarizar_clima()
    fest = estandarizar_festivos()
    horas, od = estandarizar_eod()
    sin = estandarizar_siniestros()
    afo = estandarizar_aforos()
    resumen = {
        "sensores_filas": len(sens), "tramos": len(tramos),
        "periodo_sensores": [str(sens["timestamp"].min()), str(sens["timestamp"].max())],
        "clima_horas": len(clima), "festivos": len(fest),
        "eod_viajes_total": float(od["viajes"].sum()), "eod_pares_od": len(od),
        "siniestros_filas": None if sin is None else len(sin),
        "aforos_filas_agregadas": None if afo is None else len(afo),
    }
    log.info("Fuentes estandarizadas: %s", resumen)
    return resumen
