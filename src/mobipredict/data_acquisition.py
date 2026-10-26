"""Adquisición automática de las fuentes oficiales.

Para cada fuente:
  1. Si el archivo ya está en data/raw/ (descargado antes o puesto a mano), se reutiliza.
  2. Si no, se descarga desde la URL oficial (con reintentos).
  3. Si la descarga falla y no hay archivo local, el pipeline se detiene con un mensaje
     claro. NO hay datos simulados: el proyecto trabaja solo con datos reales.

Cada ejecución deja data/raw/manifest.json con origen, URL, tamaño y hash de cada archivo.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from . import config

log = logging.getLogger(__name__)
UA = {"User-Agent": "MobiPredict/1.0 (proyecto academico de movilidad; contacto en README)"}


class FuenteNoDisponible(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _descargar(url: str, destino: Path, intentos: int = 3, **kw) -> None:
    tmp = destino.with_suffix(destino.suffix + ".part")
    ultimo = None
    for i in range(1, intentos + 1):
        try:
            with httpx.stream("GET", url, timeout=config.HTTP_TIMEOUT_S, follow_redirects=True,
                              headers=UA, **kw) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.replace(destino)
            return
        except Exception as exc:  # red, 4xx/5xx, corte de conexión
            ultimo = exc
            log.warning("Intento %d/%d fallido para %s: %s", i, intentos, url, exc)
            time.sleep(2 * i)
    tmp.unlink(missing_ok=True)
    raise FuenteNoDisponible(f"No se pudo descargar {url}: {ultimo}")


def _get_json(url: str, **params):
    r = httpx.get(url, params=params or None, timeout=config.HTTP_TIMEOUT_S, follow_redirects=True, headers=UA)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Fuentes
# --------------------------------------------------------------------------- #
def _archivo_simple(clave: str, forzar: bool) -> tuple[Path, str]:
    f = config.FUENTES[clave]
    destino = config.DATA_RAW / f["archivo"]
    if destino.exists() and destino.stat().st_size > 0 and not forzar:
        return destino, "local:data/raw"
    _descargar(f["url"], destino)
    return destino, f"descargado:{f['url']}"


def _eod_desde_referencia(destino: Path) -> None:
    """Respaldo: arma el JSON de la EOD 2025 con los agregados oficiales versionados en data/reference/eod2025/."""
    ref = config.DATA_REFERENCE / "eod2025"
    base = json.loads((ref / "indicadores_api.json").read_text(encoding="utf-8"))
    od = pd.read_csv(ref / "matriz_od_macrozonas.csv")
    base["indicadores"]["15"] = {
        "id_indicador": 15, "nombre": "Mapas",
        "matriz": [{"agrupa_mz_origen": int(o), "agrupa_mz_destino": int(d), "valor": float(v)}
                   for o, d, v in od[["origen", "destino", "viajes"]].itertuples(index=False)]}
    base["api"] = config.EOD_API
    destino.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")


def _eod2025(forzar: bool) -> tuple[Path, str]:
    """Descarga todos los indicadores agregados de la EOD 2025; si la API no responde, usa la copia
    oficial versionada en data/reference/eod2025/ (extraída de la misma API el 2026-10-23)."""
    destino = config.DATA_RAW / config.FUENTES["eod2025"]["archivo"]
    if destino.exists() and not forzar:
        return destino, "local:data/raw"
    base = config.EOD_API
    try:
        ids = _get_json(f"{base}/metadata/indicadores")["ids_indicadores"]
        indicadores = {i: _get_json(f"{base}/indicadores/{i}/agregado") for i in ids}
        por_municipio = {m: _get_json(f"{base}/indicadores/1/agregado", municipio=m)["valor"]
                         for m in _get_json(f"{base}/metadata/municipios")["municipios"]}
    except Exception as exc:
        log.warning("API EOD 2025 no disponible (%s); se usa la copia de data/reference/eod2025/", exc)
        _eod_desde_referencia(destino)
        return destino, "referencia:data/reference/eod2025 (API oficial, extraído 2026-10-23)"
    payload = {"descargado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "api": base, "indicadores": indicadores, "viajes_por_municipio_residencia": por_municipio}
    destino.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:   # polígonos de macrozonas (opcional; el EDA usa los centroides de data/reference)
        _descargar(config.FUENTES["eod2025"]["ficha"].rstrip("/") + "/macrozonas_actualizado.geo.json",
                   config.DATA_RAW / "eod2025_macrozonas.geo.json", intentos=1)
    except FuenteNoDisponible as exc:
        log.warning("GeoJSON de macrozonas no descargado: %s", exc)
    return destino, f"descargado:{base}"


def _clima(forzar: bool, inicio: str, fin: str) -> tuple[Path, str]:
    destino = config.DATA_RAW / config.FUENTES["clima"]["archivo"]
    if destino.exists() and not forzar:
        return destino, "local:data/raw"
    manual = destino.with_suffix(".json")      # descargado con scripts/descargar_datos.ps1
    if manual.exists() and not forzar:
        data = json.loads(manual.read_text(encoding="utf-8"))
        h = data["hourly"]
        pd.DataFrame({"timestamp": h["time"], "lluvia_mm": h["precipitation"],
                      "temperatura_c": h["temperature_2m"]}).to_csv(destino, index=False)
        return destino, "local:data/raw (json Open-Meteo)"
    data = _get_json(config.FUENTES["clima"]["url"], latitude=config.CLIMA_LAT, longitude=config.CLIMA_LON,
                     start_date=inicio, end_date=fin, hourly="precipitation,temperature_2m",
                     timezone="America/Bogota")
    h = data["hourly"]
    pd.DataFrame({"timestamp": h["time"], "lluvia_mm": h["precipitation"],
                  "temperatura_c": h["temperature_2m"]}).to_csv(destino, index=False)
    return destino, "descargado:open-meteo"


def adquirir_datos(forzar: bool = False) -> dict:
    """Obtiene todas las fuentes reales y escribe el manifiesto. Lanza FuenteNoDisponible si falta alguna."""
    config.ensure_dirs()
    resultados, errores = {}, {}
    tareas = {
        "velocidad_gt": lambda: _archivo_simple("velocidad_gt", forzar),
        "aforos": lambda: _archivo_simple("aforos", forzar),
        "siniestros": lambda: _archivo_simple("siniestros", forzar),
        "eod2025": lambda: _eod2025(forzar),
        "clima": lambda: _clima(forzar, "2017-01-01", "2020-12-31"),
    }
    for clave, fn in tareas.items():
        try:
            resultados[clave] = fn()
            log.info("Fuente %s: %s", clave, resultados[clave][1])
        except Exception as exc:
            errores[clave] = str(exc)
            log.error("Fuente %s no disponible: %s", clave, exc)

    manifest = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuentes": {
            k: {"titulo": config.FUENTES[k]["titulo"], "url": config.FUENTES[k]["url"],
                "ficha": config.FUENTES[k]["ficha"], "archivo": str(p.relative_to(config.ROOT)),
                "origen": o, "bytes": p.stat().st_size, "sha256_16": _sha256(p)}
            for k, (p, o) in resultados.items()
        },
        "errores": errores,
    }
    config.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    obligatorias = {"velocidad_gt", "eod2025", "clima"}
    if obligatorias & set(errores):
        faltan = sorted(obligatorias & set(errores))
        raise FuenteNoDisponible(
            f"Faltan fuentes obligatorias: {faltan}. Descárguelas a data/raw/ con los nombres de "
            "config.FUENTES o revise la conexión (ver README, sección Datos).")
    return manifest
