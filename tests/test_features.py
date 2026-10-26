"""Pruebas con un extracto REAL de MEData (Av. 80 y Autopista Sur, 10-11 feb 2020)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobipredict import config, features, fuentes

FIX = Path(__file__).parent / "fixtures" / "velocidad_gt_extracto.csv"


@pytest.fixture(scope="module")
def datos():
    sens, tramos = fuentes.estandarizar_sensores(FIX, guardar=False)
    horas = pd.DataFrame({"hora": range(24)})
    for c in ["share_viajes_privado", "share_viajes_publico", "share_viajes_total"]:
        horas[c] = 1 / 24
    store = {
        "tramos": tramos.assign(vel_referencia_kmh=sens.groupby("tramo_id")["velocidad_kmh"].max()
                                .reindex(tramos["tramo_id"]).to_numpy()),
        "perfil": pd.DataFrame(columns=["tramo_id", "tipo_dia", "hora", "perfil_velocidad"]),
        "eod_hora": horas,
        "eod_mz": pd.DataFrame({"id_macrozona": [16, 21, 14], "eod_viajes_generados_mz": [1.0, 2.0, 3.0],
                                "eod_viajes_atraidos_mz": [1.0, 2.0, 3.0]}),
        "clima": pd.DataFrame({"timestamp": sens["timestamp"].unique(), "lluvia_mm": 0.0}),
        "festivos": set(),
    }
    return sens, tramos, store


def test_parser_fecha_y_agregacion_horaria(datos):
    sens, tramos, _ = datos
    assert set(tramos["tramo_id"]) == {"AVENIDA_80_NS", "AUTOPISTA_SUR_NS"}
    # Lecturas cada 30 min -> 1 fila por hora; 10-feb tiene 6:00..20:00 = 15 horas
    dia = sens[(sens["tramo_id"] == "AVENIDA_80_NS") & (sens["timestamp"].dt.day == 10)]
    assert len(dia) == 15
    # 6:00 (22.1) y 6:30 (15.3) se promedian
    assert dia.iloc[0]["velocidad_kmh"] == pytest.approx((22.1 + 15.3) / 2)
    assert (sens["timestamp"].dt.dayofweek.isin([0, 1])).all()   # lunes y martes


def test_rezagos_no_cruzan_la_noche(datos):
    sens, _, store = datos
    f = features.construir_features(sens, store)
    primera_hora = f[(f["tramo_id"] == "AVENIDA_80_NS") & (f["timestamp"] == "2020-02-11 06:00")]
    assert primera_hora["velocidad_kmh_lag1"].isna().all()
    otra = f[(f["tramo_id"] == "AVENIDA_80_NS") & (f["timestamp"] == "2020-02-10 09:00")].iloc[0]
    previa = sens[(sens["tramo_id"] == "AVENIDA_80_NS") & (sens["timestamp"] == "2020-02-10 08:00")].iloc[0]
    assert otra["velocidad_kmh_lag1"] == pytest.approx(previa["velocidad_kmh"])


def test_inferencia_igual_a_lote(datos):
    sens, _, store = datos
    lote = features.construir_features(sens, store)
    t = sens[sens["tramo_id"] == "AUTOPISTA_SUR_NS"].iloc[5:9]
    obs = [{"timestamp": r.timestamp, "velocidad_kmh": r.velocidad_kmh, "lluvia_mm": 0.0} for r in t.itertuples()]
    fila_api = features.features_inferencia("AUTOPISTA_SUR_NS", obs, store)
    fila_lote = lote[(lote["tramo_id"] == "AUTOPISTA_SUR_NS") & (lote["timestamp"] == t["timestamp"].iloc[-1])]
    a, b = fila_api[features.FEATURES_NUM].to_numpy(float), fila_lote[features.FEATURES_NUM].to_numpy(float)
    np.testing.assert_allclose(a, b, rtol=1e-6, equal_nan=True)


def test_historia_insuficiente(datos):
    sens, _, store = datos
    obs = [{"timestamp": "2020-02-10 06:00", "velocidad_kmh": 20.0}]
    with pytest.raises(ValueError):
        features.features_inferencia("AVENIDA_80_NS", obs, store)


def test_targets_sin_fuga(datos):
    sens, _, store = datos
    f = features.agregar_targets(features.construir_features(sens, store))
    ultima = f[(f["tramo_id"] == "AVENIDA_80_NS") & (f["timestamp"] == "2020-02-10 20:00")]
    assert ultima[features.TARGET_REG].isna().all()          # el objetivo sería al día siguiente
    r = f[(f["tramo_id"] == "AVENIDA_80_NS") & (f["timestamp"] == "2020-02-10 07:00")].iloc[0]
    s8 = sens[(sens["tramo_id"] == "AVENIDA_80_NS") & (sens["timestamp"] == "2020-02-10 08:00")].iloc[0]
    assert r[features.TARGET_REG] == pytest.approx(s8["velocidad_kmh"])
    assert config.HORIZON_STEPS == 1
