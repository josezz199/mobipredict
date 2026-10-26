"""Entrenamiento, optimización de hiperparámetros y registro en MLflow.

Dos tareas sobre la serie real de velocidad horaria por tramo (MEData/SIMM 2017-2020):
  * Regresión: velocidad media del tramo dentro de 1 hora (km/h).
  * Clasificación: colapso dentro de 1 hora (velocidad < 50 % de la de referencia del tramo).

Cada tarea compara una línea base sin entrenamiento contra modelos lineales,
Random Forest, LightGBM (Gradient Boosting) y una red neuronal (MLP). La
búsqueda de hiperparámetros usa validación cruzada temporal (TimeSeriesSplit):
RandomizedSearchCV para scikit-learn y Optuna para LightGBM.

Protocolo: se entrena con `train`, se elige umbral y modelo con `valid` y se
reporta el desempeño final en `test` (nunca visto durante el ajuste). Además se mide el
desempeño en `drift` (marzo-diciembre de 2020, pandemia) como prueba de estrés de drift.
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES_NUM, TARGET_CLF, TARGET_REG

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
CAND_DIR = config.MODELS_DIR / "candidatos"


# --------------------------------------------------------------------------- #
# Líneas base (sin entrenamiento) con interfaz scikit-learn
# --------------------------------------------------------------------------- #
class PerfilHistoricoRegressor(BaseEstimator, RegressorMixin):
    """Predice la velocidad promedio histórica del tramo para el tipo de día y la hora objetivo."""

    def fit(self, X, y=None):
        self.fallback_ = float(np.nanmean(y)) if y is not None else 0.0
        return self

    def predict(self, X):
        p = X["perfil_velocidad"].to_numpy(dtype=float)
        return np.where(np.isnan(p), X["velocidad_kmh"].fillna(self.fallback_).to_numpy(), p)


class ReglaVelocidadClassifier(BaseEstimator, ClassifierMixin):
    """Regla operativa: riesgo de colapso según qué tan cerca está la velocidad actual del umbral."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        r = X["ratio_vel"].fillna(1).to_numpy()
        tendencia = X["ratio_vel_delta1"].fillna(0).to_numpy()
        umbral = config.COLLAPSE_SPEED_RATIO
        p = np.clip((umbral + 0.15 - (r + tendencia)) / 0.30, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metricas_regresion(y, p, es_pico) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    es_pico = np.asarray(es_pico).astype(bool)
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "wape": float(np.abs(y - p).sum() / np.abs(y).sum()),
        "r2": r2_score(y, p),
        "mae_pico": mean_absolute_error(y[es_pico], p[es_pico]) if es_pico.any() else np.nan,
    }


def umbral_optimo(y, prob) -> float:
    """Umbral que maximiza F1 en validación (se fija ANTES de mirar test)."""
    prec, rec, thr = precision_recall_curve(y, prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5


def metricas_clasificacion(y, prob, umbral: float, n_dias: float) -> dict:
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= umbral).astype(int)
    falsas = int(((pred == 1) & (y == 0)).sum())
    if y.sum() == 0:   # sin colapsos reales (p. ej. pandemia): solo tiene sentido medir falsas alarmas
        return {"precision_colapso": np.nan, "recall_colapso": np.nan, "f1_colapso": np.nan, "pr_auc": np.nan,
                "roc_auc": np.nan, "brier": brier_score_loss(y, prob),
                "falsas_alarmas_por_dia": falsas / max(n_dias, 1e-9), "tasa_positivos": 0.0}
    return {
        "precision_colapso": precision_score(y, pred, zero_division=0),
        "recall_colapso": recall_score(y, pred, zero_division=0),
        "f1_colapso": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, prob),
        "falsas_alarmas_por_dia": falsas / max(n_dias, 1e-9),
        "tasa_positivos": float(np.mean(y)),
    }


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _recientes(df: pd.DataFrame) -> pd.DataFrame:
    """Las filas más recientes hasta FIT_MAX_ROWS (para modelos costosos: RF y MLP)."""
    return df.iloc[-config.FIT_MAX_ROWS:] if len(df) > config.FIT_MAX_ROWS else df


def _muestra_busqueda(train: pd.DataFrame) -> pd.DataFrame:
    """Submuestra ordenada en el tiempo para la búsqueda de hiperparámetros."""
    if len(train) > config.SEARCH_SAMPLE:
        train = train.sample(config.SEARCH_SAMPLE, random_state=config.SEED)
    return train.sort_values("timestamp")


def _latencia_ms(model, X: pd.DataFrame, clasif: bool) -> float:
    Xs = X.iloc[: min(1000, len(X))]
    t0 = time.perf_counter()
    for _ in range(3):
        model.predict_proba(Xs) if clasif else model.predict(Xs)
    return (time.perf_counter() - t0) / 3 * 1000 / len(Xs) * 1000   # ms por 1000 filas


def _importancias(model, nombre: str) -> pd.DataFrame | None:
    est = model
    if isinstance(est, TransformedTargetRegressor):
        est = est.regressor_
    if isinstance(est, Pipeline):
        est = est.steps[-1][1]
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(np.ravel(est.coef_))
    else:
        return None
    imp = imp / imp.sum() if imp.sum() else imp
    return (pd.DataFrame({"feature": FEATURES_NUM, "importancia": imp})
            .sort_values("importancia", ascending=False).reset_index(drop=True))


def _tscv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=3)


# --------------------------------------------------------------------------- #
# Candidatos y búsqueda de hiperparámetros
# --------------------------------------------------------------------------- #
def _imp() -> SimpleImputer:
    return SimpleImputer(strategy="median")


def _random_search(est, dist, X, y, scoring, n_iter=None):
    rs = RandomizedSearchCV(est, dist, n_iter=n_iter or config.N_ITER_RANDOM, cv=_tscv(), scoring=scoring,
                            random_state=config.SEED, n_jobs=-1, refit=False)
    rs.fit(X, y)
    return rs.best_params_, float(rs.best_score_)


def _optuna_lgbm(X, y, clasif: bool):
    def objetivo(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        if clasif:
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1, 20, log=True)
            m = lgb.LGBMClassifier(**params, random_state=config.SEED, verbose=-1, n_jobs=-1)
            return cross_val_score(m, X, y, cv=_tscv(), scoring="average_precision").mean()
        m = lgb.LGBMRegressor(**params, random_state=config.SEED, verbose=-1, n_jobs=-1)
        return cross_val_score(m, X, y, cv=_tscv(), scoring="neg_root_mean_squared_error").mean()

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    estudio.optimize(objetivo, n_trials=config.N_TRIALS_OPTUNA, show_progress_bar=False)
    best = dict(estudio.best_params, subsample_freq=1)
    return best, float(estudio.best_value)


def ajustar_regresores(train: pd.DataFrame) -> dict:
    """Devuelve {nombre: (modelo_entrenado, params, cv_score)} para la tarea de velocidad."""
    s = _muestra_busqueda(train)
    Xs, ys = s[FEATURES_NUM], s[TARGET_REG]
    X, y = train[FEATURES_NUM], train[TARGET_REG]
    out = {}

    out["perfil_historico"] = (PerfilHistoricoRegressor().fit(X, y), {"tipo": "linea_base"}, np.nan)

    ridge = Pipeline([("imp", _imp()), ("sc", StandardScaler()), ("m", Ridge())])
    bp, sc = _random_search(ridge, {"m__alpha": loguniform(1e-2, 1e3)}, Xs, ys, "neg_root_mean_squared_error")
    out["ridge"] = (ridge.set_params(**bp).fit(X, y), bp, sc)

    rf = Pipeline([("imp", _imp()), ("m", RandomForestRegressor(random_state=config.SEED, n_jobs=-1))])
    dist = {"m__n_estimators": [100, 150, 200], "m__max_depth": [8, 12, 16, 22],
            "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": ["sqrt", 0.3, 0.5]}
    bp, sc = _random_search(rf, dist, Xs, ys, "neg_root_mean_squared_error")
    r = _recientes(train)
    out["random_forest"] = (rf.set_params(**bp).fit(r[FEATURES_NUM], r[TARGET_REG]), bp, sc)

    bp, sc = _optuna_lgbm(Xs, ys, clasif=False)
    out["lightgbm"] = (lgb.LGBMRegressor(**bp, random_state=config.SEED, verbose=-1, n_jobs=-1).fit(X, y), bp, sc)

    mlp = TransformedTargetRegressor(
        regressor=Pipeline([("imp", _imp()), ("sc", StandardScaler()),
                            ("m", MLPRegressor(early_stopping=True, max_iter=300, random_state=config.SEED))]),
        transformer=StandardScaler())
    dist = {"regressor__m__hidden_layer_sizes": [(64,), (128, 64), (64, 32), (128, 64, 32)],
            "regressor__m__alpha": loguniform(1e-5, 1e-2),
            "regressor__m__learning_rate_init": loguniform(5e-4, 5e-3)}
    bp, sc = _random_search(mlp, dist, Xs, ys, "neg_root_mean_squared_error", n_iter=max(3, config.N_ITER_RANDOM // 2))
    out["mlp"] = (mlp.set_params(**bp).fit(r[FEATURES_NUM], r[TARGET_REG]), bp, sc)
    return out


def ajustar_clasificadores(train: pd.DataFrame) -> dict:
    s = _muestra_busqueda(train)
    Xs, ys = s[FEATURES_NUM], s[TARGET_CLF]
    X, y = train[FEATURES_NUM], train[TARGET_CLF]
    out = {}

    out["regla_umbral"] = (ReglaVelocidadClassifier().fit(X, y), {"tipo": "linea_base"}, np.nan)

    lr = Pipeline([("imp", _imp()), ("sc", StandardScaler()),
                   ("m", LogisticRegression(class_weight="balanced", max_iter=3000))])
    bp, sc = _random_search(lr, {"m__C": loguniform(1e-3, 1e2)}, Xs, ys, "average_precision")
    out["logistica"] = (lr.set_params(**bp).fit(X, y), bp, sc)

    rf = Pipeline([("imp", _imp()), ("m", RandomForestClassifier(class_weight="balanced_subsample",
                                                                  random_state=config.SEED, n_jobs=-1))])
    dist = {"m__n_estimators": [100, 150, 200], "m__max_depth": [6, 10, 14, 20],
            "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": ["sqrt", 0.3, 0.5]}
    bp, sc = _random_search(rf, dist, Xs, ys, "average_precision")
    r = _recientes(train)
    out["random_forest"] = (rf.set_params(**bp).fit(r[FEATURES_NUM], r[TARGET_CLF]), bp, sc)

    bp, sc = _optuna_lgbm(Xs, ys, clasif=True)
    out["lightgbm"] = (lgb.LGBMClassifier(**bp, random_state=config.SEED, verbose=-1, n_jobs=-1).fit(X, y), bp, sc)
    return out


# --------------------------------------------------------------------------- #
# Evaluación + MLflow
# --------------------------------------------------------------------------- #
def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def _registrar_run(tarea: str, nombre: str, modelo, params: dict, cv_score: float, met: dict,
                   imp: pd.DataFrame | None, extra_params: dict | None = None) -> str:
    with mlflow.start_run(run_name=f"{tarea}-{nombre}", nested=True) as run:
        mlflow.set_tags({"tarea": tarea, "modelo": nombre, "proyecto": "MobiPredict"})
        mlflow.log_params({k: (str(v) if isinstance(v, (tuple, list)) else v) for k, v in params.items()})
        if extra_params:
            mlflow.log_params(extra_params)
        if not np.isnan(cv_score):
            mlflow.log_metric("cv_score", cv_score)
        mlflow.log_metrics({k: float(v) for k, v in met.items() if v is not None and not np.isnan(v)})
        if imp is not None:
            ruta = config.MODELS_DIR / f"_imp_{tarea}_{nombre}.csv"
            imp.to_csv(ruta, index=False)
            mlflow.log_artifact(str(ruta), artifact_path="importancias")
            ruta.unlink()
        ruta_modelo = CAND_DIR / f"{tarea}_{nombre}.joblib"
        joblib.dump(modelo, ruta_modelo)
        mlflow.log_artifact(str(ruta_modelo), artifact_path="modelo")
        return run.info.run_id


def entrenar_y_registrar(particiones: dict[str, pd.DataFrame], spec: dict) -> pd.DataFrame:
    """Entrena todos los candidatos, los evalúa y registra cada uno como un run de MLflow.

    Devuelve la tabla comparativa (una fila por tarea x modelo).
    """
    config.ensure_dirs()
    CAND_DIR.mkdir(exist_ok=True)
    _setup_mlflow()
    train = particiones["train"].sort_values("timestamp")
    valid, test, drift = particiones["valid"], particiones["test"], particiones["drift"]
    dias = {p: particiones[p]["timestamp"].dt.normalize().nunique() for p in ["valid", "test", "drift"]}
    filas = []

    with mlflow.start_run(run_name=f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}") as padre:
        mlflow.set_tags({"tipo": "pipeline", "modo_rapido": str(config.FAST)})
        mlflow.log_params({"horizonte_min": spec["horizonte_min"], "n_features": len(FEATURES_NUM),
                           "filas_train": len(train), "filas_valid": len(valid), "filas_test": len(test),
                           "trials_optuna": config.N_TRIALS_OPTUNA, "iter_random_search": config.N_ITER_RANDOM})
        mlflow.log_dict(spec, "feature_spec.json")

        # ----- Regresión: velocidad a 1 hora
        t0 = time.time()
        regs = ajustar_regresores(train)
        log.info("Regresores ajustados en %.0f s", time.time() - t0)
        for nombre, (m, params, cv) in regs.items():
            met = {}
            for part, df in [("valid", valid), ("test", test), ("drift", drift)]:
                if df.empty:
                    continue
                for k, v in metricas_regresion(df[TARGET_REG], m.predict(df[FEATURES_NUM]), df["es_pico"]).items():
                    met[f"{part}_{k}"] = v
            met["latencia_ms_1000"] = _latencia_ms(m, test[FEATURES_NUM], clasif=False)
            run_id = _registrar_run("velocidad", nombre, m, params, cv, met, _importancias(m, nombre))
            filas.append({"tarea": "velocidad", "modelo": nombre, "run_id": run_id, "cv_score": cv,
                          "params": json.dumps(params, default=str), **met})

        # ----- Clasificación: colapso a 1 hora
        t0 = time.time()
        clfs = ajustar_clasificadores(train)
        log.info("Clasificadores ajustados en %.0f s", time.time() - t0)
        for nombre, (m, params, cv) in clfs.items():
            pv = m.predict_proba(valid[FEATURES_NUM])[:, 1]
            umbral = umbral_optimo(valid[TARGET_CLF], pv)
            met = {f"valid_{k}": v for k, v in
                   metricas_clasificacion(valid[TARGET_CLF], pv, umbral, dias["valid"]).items()}
            pt = m.predict_proba(test[FEATURES_NUM])[:, 1]
            met.update({f"test_{k}": v for k, v in
                        metricas_clasificacion(test[TARGET_CLF], pt, umbral, dias["test"]).items()})
            if not drift.empty:
                pdft = m.predict_proba(drift[FEATURES_NUM])[:, 1]
                met.update({f"drift_{k}": v for k, v in
                            metricas_clasificacion(drift[TARGET_CLF], pdft, umbral, dias["drift"]).items()})
            met["umbral"] = umbral
            met["latencia_ms_1000"] = _latencia_ms(m, test[FEATURES_NUM], clasif=True)
            run_id = _registrar_run("colapso", nombre, m, params, cv, met, _importancias(m, nombre),
                                    {"umbral_decision": round(umbral, 4)})
            filas.append({"tarea": "colapso", "modelo": nombre, "run_id": run_id, "cv_score": cv,
                          "params": json.dumps(params, default=str), **met})

        tabla = pd.DataFrame(filas)
        ruta = config.MODELS_DIR / "_resultados_entrenamiento.csv"
        tabla.to_csv(ruta, index=False)
        mlflow.log_artifact(str(ruta))
        tabla.attrs["parent_run_id"] = padre.info.run_id
    return tabla
