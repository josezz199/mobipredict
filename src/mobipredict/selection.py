"""Selección razonada del modelo ganador por tarea.

Reglas (definidas antes de ver resultados y aplicadas sobre VALIDACIÓN; test solo se reporta):

Velocidad a 1 hora
  1. Debe mejorar el WAPE de la línea base (perfil histórico) en al menos 10 %.
  2. Gana el menor RMSE de validación.
  3. Desempate: si otro candidato está a menos de 1 % de RMSE, se prefiere el de menor latencia.

Colapso a 1 hora
  1. Elegibles: recall de validación >= 0,70 con el umbral F1-óptimo.
     Si ninguno cumple, se consideran todos y se advierte.
  2. Gana la mayor PR-AUC de validación (robusta al desbalance).
  3. Desempate: diferencia de PR-AUC < 0,005 -> menor Brier y luego menor latencia.

El periodo de pandemia (drift) se reporta para todos los modelos, pero NO se usa para elegir.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from . import config
from .training import CAND_DIR

log = logging.getLogger(__name__)

BASE_VEL, BASE_COLAPSO = "perfil_historico", "regla_umbral"
MEJORA_MIN_WAPE = 0.10
TOL_RMSE = 0.01
RECALL_MIN = 0.70
TOL_PRAUC = 0.005


def _elegir_velocidad(t: pd.DataFrame) -> tuple[str, list[str]]:
    razones = []
    base = t.loc[t["modelo"] == BASE_VEL].iloc[0]
    cand = t[t["modelo"] != BASE_VEL].copy()
    cand["mejora_wape_vs_base"] = 1 - cand["valid_wape"] / base["valid_wape"]
    aptos = cand[cand["mejora_wape_vs_base"] >= MEJORA_MIN_WAPE]
    if aptos.empty:
        razones.append("Ningún modelo superó a la línea base en 10 % de WAPE; se elige el mejor disponible y se "
                       "recomienda revisar features.")
        aptos = cand
    else:
        fuera = sorted(set(cand["modelo"]) - set(aptos["modelo"]))
        if fuera:
            razones.append(f"Descartados por no mejorar 10 % el WAPE de la línea base: {', '.join(fuera)}.")
    aptos = aptos.sort_values("valid_rmse")
    mejor = aptos.iloc[0]
    cercanos = aptos[aptos["valid_rmse"] <= mejor["valid_rmse"] * (1 + TOL_RMSE)]
    ganador = cercanos.sort_values("latencia_ms_1000").iloc[0] if len(cercanos) > 1 else mejor
    if ganador["modelo"] != mejor["modelo"]:
        razones.append(f"{ganador['modelo']} queda a menos de 1 % del RMSE de {mejor['modelo']} y es más rápido; "
                       "se prefiere por latencia.")
    razones.append(f"{ganador['modelo']} obtiene el menor RMSE de validación ({ganador['valid_rmse']:.2f} km/h) y "
                   f"reduce el WAPE {ganador['mejora_wape_vs_base'] * 100:.1f} % frente al perfil histórico.")
    return ganador["modelo"], razones


def _elegir_colapso(t: pd.DataFrame) -> tuple[str, list[str]]:
    razones = []
    cand = t[t["modelo"] != BASE_COLAPSO]
    aptos = cand[cand["valid_recall_colapso"] >= RECALL_MIN]
    if aptos.empty:
        razones.append(f"Ningún modelo alcanza recall >= {RECALL_MIN} en validación; se elige por PR-AUC entre todos.")
        aptos = cand
    else:
        fuera = sorted(set(cand["modelo"]) - set(aptos["modelo"]))
        if fuera:
            razones.append(f"Descartados por recall < {RECALL_MIN}: {', '.join(fuera)}.")
    aptos = aptos.sort_values("valid_pr_auc", ascending=False)
    mejor = aptos.iloc[0]
    cercanos = aptos[aptos["valid_pr_auc"] >= mejor["valid_pr_auc"] - TOL_PRAUC]
    ganador = cercanos.sort_values(["valid_brier", "latencia_ms_1000"]).iloc[0] if len(cercanos) > 1 else mejor
    if ganador["modelo"] != mejor["modelo"]:
        razones.append(f"{ganador['modelo']} empata en PR-AUC con {mejor['modelo']} (< 0,005) y está mejor calibrado.")
    razones.append(f"{ganador['modelo']} obtiene PR-AUC de validación {ganador['valid_pr_auc']:.3f} con recall "
                   f"{ganador['valid_recall_colapso']:.2f} y precisión {ganador['valid_precision_colapso']:.2f} "
                   f"(umbral {ganador['umbral']:.3f}).")
    return ganador["modelo"], razones


def _registrar_en_mlflow(tarea: str, modelo: str, run_id: str) -> str | None:
    """Marca el run ganador y registra el modelo en el Model Registry local con alias champion."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    client.set_tag(run_id, "seleccionado", "true")
    nombre = f"mobipredict-{tarea}"
    try:
        modelo_obj = joblib.load(CAND_DIR / f"{tarea}_{modelo}.joblib")
        with mlflow.start_run(run_id=run_id):
            info = mlflow.sklearn.log_model(modelo_obj, name="modelo_mlflow", serialization_format="cloudpickle")
        mv = mlflow.register_model(info.model_uri, nombre)
        client.set_registered_model_alias(nombre, "champion", mv.version)
        return f"{nombre} v{mv.version} (alias champion)"
    except Exception as exc:
        log.warning("No se pudo registrar %s en el Model Registry: %s", nombre, exc)
        return None


ORDEN = ["tarea", "modelo", "seleccionado",
         "valid_mae", "valid_rmse", "valid_wape", "valid_r2", "valid_mae_pico",
         "test_mae", "test_rmse", "test_wape", "test_r2", "test_mae_pico",
         "drift_mae", "drift_rmse", "drift_wape",
         "valid_precision_colapso", "valid_recall_colapso", "valid_f1_colapso", "valid_pr_auc", "valid_roc_auc",
         "valid_brier", "test_precision_colapso", "test_recall_colapso", "test_f1_colapso", "test_pr_auc",
         "test_roc_auc", "test_brier", "test_falsas_alarmas_por_dia", "test_tasa_positivos",
         "drift_precision_colapso", "drift_recall_colapso", "drift_pr_auc", "drift_falsas_alarmas_por_dia",
         "umbral", "latencia_ms_1000", "cv_score", "run_id", "params"]


def seleccionar(tabla: pd.DataFrame, spec: dict) -> dict:
    config.ensure_dirs()
    vel = tabla[tabla["tarea"] == "velocidad"].copy()
    col = tabla[tabla["tarea"] == "colapso"].copy()
    g_vel, r_vel = _elegir_velocidad(vel)
    g_col, r_col = _elegir_colapso(col)

    tabla = tabla.copy()
    tabla["seleccionado"] = (((tabla["tarea"] == "velocidad") & (tabla["modelo"] == g_vel))
                             | ((tabla["tarea"] == "colapso") & (tabla["modelo"] == g_col)))
    tabla[[c for c in ORDEN if c in tabla.columns]].round(4).to_csv(config.COMPARISON_FILE, index=False)

    fv, fc = vel.set_index("modelo").loc[g_vel], col.set_index("modelo").loc[g_col]
    shutil.copy(CAND_DIR / f"velocidad_{g_vel}.joblib", config.MODEL_SPEED_FILE)
    shutil.copy(CAND_DIR / f"colapso_{g_col}.joblib", config.MODEL_COLLAPSE_FILE)
    reg_v = _registrar_en_mlflow("velocidad", g_vel, fv["run_id"])
    reg_c = _registrar_en_mlflow("colapso", g_col, fc["run_id"])

    manifest = json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8")) if config.MANIFEST_FILE.exists() else {}
    metadata = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizonte_min": spec["horizonte_min"],
        "features": spec["features"],
        "umbral_colapso_ratio_vel": spec["umbral_colapso_ratio_vel"],
        "min_observaciones": config.MIN_HISTORY,
        "modelo_velocidad": {"nombre": g_vel, "run_id": fv["run_id"], "registro": reg_v,
                             "test_mae_kmh": round(float(fv["test_mae"]), 3),
                             "test_rmse_kmh": round(float(fv["test_rmse"]), 3),
                             "test_wape": round(float(fv["test_wape"]), 4), "razones": list(r_vel)},
        "modelo_colapso": {"nombre": g_col, "run_id": fc["run_id"], "registro": reg_c,
                           "umbral_decision": round(float(fc["umbral"]), 4),
                           "test_pr_auc": round(float(fc["test_pr_auc"]), 4),
                           "test_recall": round(float(fc["test_recall_colapso"]), 4),
                           "test_precision": round(float(fc["test_precision_colapso"]), 4),
                           "razones": list(r_col)},
        "tasa_colapso": spec.get("tasa_colapso", {}), "filas": spec.get("filas", {}),
        "fuentes": {k: {"titulo": v.get("titulo"), "origen": v.get("origen"), "ficha": v.get("ficha")}
                    for k, v in manifest.get("fuentes", {}).items()},
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _escribir_decision(vel, col, g_vel, g_col, r_vel, r_col, metadata, spec)
    log.info("Seleccionados: velocidad=%s, colapso=%s", g_vel, g_col)
    return metadata


def _tabla_md(df: pd.DataFrame, cols: dict[str, str], ganador: str, fmt: dict[str, str]) -> list[str]:
    head = "| Modelo | " + " | ".join(cols.values()) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    filas = []
    for _, r in df.iterrows():
        nombre = f"**{r['modelo']}** ✔" if r["modelo"] == ganador else r["modelo"]
        celdas = [("—" if pd.isna(r.get(c)) else format(r[c], fmt.get(c, ".3f"))) for c in cols]
        filas.append(f"| {nombre} | " + " | ".join(celdas) + " |")
    return [head, sep, *filas]


def _nota_test(df, ganador, col, menor_es_mejor, nombre_met) -> list[str]:
    mejor_test = df.sort_values(col, ascending=menor_es_mejor).iloc[0]
    if mejor_test["modelo"] == ganador:
        return [f"- En prueba el ganador también es el mejor en {nombre_met}: la decisión se sostiene fuera de muestra."]
    g = df.set_index("modelo").loc[ganador]
    return [f"- En prueba, {mejor_test['modelo']} queda ligeramente mejor en {nombre_met} "
            f"({mejor_test[col]:.3f} frente a {g[col]:.3f}). No se cambia la elección: hacerlo usaría el conjunto de "
            "prueba para seleccionar. La diferencia se vigila en producción con el esquema champion/challenger."]


def _hz(spec) -> str:
    m = spec["horizonte_min"]
    return "1 hora" if m == 60 else f"{m} min"


def _escribir_decision(vel, col, g_vel, g_col, r_vel, r_col, meta, spec) -> None:
    v = vel.sort_values("valid_rmse")
    c = col.sort_values("valid_pr_auc", ascending=False)
    gv, gc = v.set_index("modelo").loc[g_vel], c.set_index("modelo").loc[g_col]
    bv, bc = v.set_index("modelo").loc[BASE_VEL], c.set_index("modelo").loc[BASE_COLAPSO]
    tasa = spec.get("tasa_colapso", {})
    L = [
        "# Decisión de modelo — MobiPredict",
        "",
        f"Generado automáticamente el {meta['generado_utc']} por `src/mobipredict/selection.py`.",
        "",
        "## Resumen",
        "",
        f"- **Velocidad a {_hz(spec)}:** `{g_vel}` — RMSE de prueba {gv['test_rmse']:.2f} km/h, "
        f"MAE {gv['test_mae']:.2f} km/h, WAPE {gv['test_wape'] * 100:.1f} % "
        f"(perfil histórico: WAPE {bv['test_wape'] * 100:.1f} %).",
        f"- **Colapso a {_hz(spec)}:** `{g_col}` — PR-AUC de prueba {gc['test_pr_auc']:.3f}, recall "
        f"{gc['test_recall_colapso']:.2f}, precisión {gc['test_precision_colapso']:.2f} con umbral {gc['umbral']:.3f} "
        f"(regla base: PR-AUC {bc['test_pr_auc']:.3f}).",
        "",
        "## Datos y protocolo",
        "",
        f"- Sensores: {spec['fuente_sensores']}, {spec['tramos']} tramos, lecturas horarias 6:00-20:00.",
        f"- Split temporal: entrenamiento hasta {spec['corte_validacion']}, validación hasta {spec['corte_test']}, "
        f"prueba hasta {spec['fin_test']} (antes de las restricciones por COVID-19). La hora frontera de cada "
        "partición se descarta para que el objetivo no cruce de una a otra.",
        f"- Drift: {spec['ventana_drift'][0]} a {spec['ventana_drift'][1]} (pandemia), solo para diagnóstico.",
        f"- Colapso = velocidad < {spec['umbral_colapso_ratio_vel']:.0%} de la velocidad de referencia del tramo "
        f"(percentil {spec['percentil_vel_referencia'] * 100:.0f} del entrenamiento). Tasa de colapso: "
        + ", ".join(f"{k} {v:.1%}" for k, v in tasa.items()) + ".",
        "- Hiperparámetros con validación cruzada temporal (TimeSeriesSplit, 3 cortes): RandomizedSearchCV y Optuna.",
        "- Selección y umbral de decisión se fijan en **validación**; **prueba** solo se reporta.",
        "",
        f"## Velocidad a {_hz(spec)}",
        "",
        *_tabla_md(v, {"valid_rmse": "RMSE val", "valid_wape": "WAPE val", "valid_mae_pico": "MAE pico val",
                       "test_mae": "MAE test", "test_rmse": "RMSE test", "test_r2": "R² test",
                       "latencia_ms_1000": "ms/1000 filas"}, g_vel,
                   {"valid_rmse": ".2f", "valid_mae_pico": ".2f", "test_mae": ".2f", "test_rmse": ".2f",
                    "latencia_ms_1000": ".1f"}),
        "",
        "Unidades: km/h. WAPE = error absoluto total / velocidad total.",
        "",
        "**Razonamiento**",
        "",
        *[f"- {r}" for r in r_vel],
        *_nota_test(v, g_vel, "test_rmse", True, "RMSE"),
        "",
        f"## Colapso a {_hz(spec)}",
        "",
        *_tabla_md(c, {"valid_pr_auc": "PR-AUC val", "valid_recall_colapso": "Recall val",
                       "valid_precision_colapso": "Precisión val", "valid_brier": "Brier val",
                       "test_pr_auc": "PR-AUC test", "test_recall_colapso": "Recall test",
                       "test_precision_colapso": "Precisión test",
                       "test_falsas_alarmas_por_dia": "Falsas alarmas/día (red)",
                       "latencia_ms_1000": "ms/1000 filas"}, g_col,
                   {"test_falsas_alarmas_por_dia": ".1f", "latencia_ms_1000": ".1f"}),
        "",
        "**Razonamiento**",
        "",
        *[f"- {r}" for r in r_col],
        *_nota_test(c, g_col, "test_pr_auc", False, "PR-AUC"),
        "- Se prioriza recall: una alerta perdida cuesta más que una falsa alarma que un operador descarta.",
        "",
        "## Prueba de estrés: pandemia (drift)",
        "",
        *_tabla_md(v, {"test_mae": "MAE test (pre-pandemia)", "drift_mae": "MAE drift (pandemia)",
                       "drift_wape": "WAPE drift"}, g_vel, {"test_mae": ".2f", "drift_mae": ".2f"}),
        "",
        *_tabla_md(c, {"test_pr_auc": "PR-AUC test", "drift_pr_auc": "PR-AUC drift",
                       "drift_recall_colapso": "Recall drift", "drift_precision_colapso": "Precisión drift",
                       "drift_falsas_alarmas_por_dia": "Falsas alarmas/día drift"},
                   g_col, {"drift_falsas_alarmas_por_dia": ".1f"}),
        "",
        "— = no aplica: sin colapsos reales en el periodo no se pueden medir recall ni PR-AUC.",
        "",
        "Con las vías casi vacías la congestión casi desaparece: la tasa de colapso del periodo de drift es "
        f"{tasa.get('drift', float('nan')):.1%} frente a {tasa.get('test', float('nan')):.1%} en prueba. Es un cambio de "
        "régimen que el monitoreo debe detectar (drift de entrada en velocidades y de salida en la tasa de alertas) "
        "y que justifica el reentrenamiento por disparador, no solo por calendario.",
        "",
        "## Riesgos y siguientes pasos",
        "",
        "- Los sensores abiertos más recientes publicados por la Alcaldía llegan a dic-2020; antes de operar en 2026 "
        "hay que reentrenar con el feed actual del SIMM.",
        "- Revisar el umbral de colapso y el de alerta con la Secretaría de Movilidad.",
        "- Evaluar un modelo de grafos espacio-temporal como challenger con más tramos.",
    ]
    config.DECISION_FILE.write_text("\n".join(L), encoding="utf-8")
