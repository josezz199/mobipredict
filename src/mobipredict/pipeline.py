"""Orquestación con Prefect.

Adquisición -> Estandarización -> EDA -> Features -> Entrenamiento/Optimización (+ MLflow) -> Selección.
"""
from __future__ import annotations

import time

import pandas as pd
from prefect import flow, get_run_logger, task

from . import config, data_acquisition, eda, features, fuentes, selection, training


@task(name="adquisicion-datos", retries=1, retry_delay_seconds=10)
def tarea_adquisicion(forzar_descarga: bool = False) -> dict:
    manifest = data_acquisition.adquirir_datos(forzar=forzar_descarga)
    get_run_logger().info("Fuentes: %s", {k: v["origen"] for k, v in manifest["fuentes"].items()})
    return manifest


@task(name="estandarizacion-fuentes")
def tarea_estandarizacion() -> dict:
    resumen = fuentes.estandarizar_todo()
    get_run_logger().info("Tablas estandarizadas: %s", resumen)
    return resumen


@task(name="eda")
def tarea_eda() -> dict:
    resumen = eda.ejecutar_eda()
    get_run_logger().info("EDA: %d figuras en eda/figures/", len(resumen["figuras"]))
    return resumen


@task(name="feature-engineering")
def tarea_features() -> dict:
    spec = features.preparar_dataset()
    get_run_logger().info("Particiones: %s", spec["filas"])
    return spec


@task(name="entrenamiento-optimizacion-mlflow")
def tarea_entrenamiento(spec: dict) -> pd.DataFrame:
    tabla = training.entrenar_y_registrar(features.cargar_particiones(), spec)
    get_run_logger().info("Candidatos entrenados: %d (MLflow: %s)", len(tabla), config.MLFLOW_TRACKING_URI)
    return tabla


@task(name="seleccion-modelo")
def tarea_seleccion(tabla: pd.DataFrame, spec: dict) -> dict:
    meta = selection.seleccionar(tabla, spec)
    get_run_logger().info("Ganadores: velocidad=%s, colapso=%s",
                          meta["modelo_velocidad"]["nombre"], meta["modelo_colapso"]["nombre"])
    return meta


@flow(name="mobipredict-pipeline", log_prints=True)
def mobipredict_pipeline(ejecutar_eda: bool = True, forzar_descarga: bool = False) -> dict:
    t0 = time.time()
    tarea_adquisicion(forzar_descarga)
    tarea_estandarizacion()
    resumen_eda = tarea_eda() if ejecutar_eda else None
    spec = tarea_features()
    tabla = tarea_entrenamiento(spec)
    meta = tarea_seleccion(tabla, spec)
    print(f"Pipeline completo en {time.time() - t0:.0f} s")
    return {"eda": resumen_eda, "spec": spec, "metadata": meta}
