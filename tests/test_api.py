"""Pruebas de la API. Requieren haber corrido el pipeline (models/ con artefactos)."""
import pytest
from fastapi.testclient import TestClient

from mobipredict import config

pytestmark = pytest.mark.skipif(not config.MODEL_SPEED_FILE.exists(), reason="ejecute run_pipeline.py primero")

from api.main import app  # noqa: E402
from api.schemas import EJEMPLO_OBS, EJEMPLO_TRAMO  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modelos_cargados"] is True


def test_tramos(client):
    r = client.get("/tramos")
    assert r.status_code == 200
    assert any(t["tramo_id"] == EJEMPLO_TRAMO for t in r.json())


def test_predict(client):
    r = client.post("/predict", json={"tramo_id": EJEMPLO_TRAMO, "observaciones": EJEMPLO_OBS})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 1 <= b["velocidad_predicha_kmh"] <= 130
    assert b["tiempo_viaje_predicho_min"] > 0
    assert 0 <= b["probabilidad_colapso"] <= 1


def test_predict_tramo_invalido(client):
    r = client.post("/predict", json={"tramo_id": "XX", "observaciones": EJEMPLO_OBS})
    assert r.status_code == 422


def test_predict_batch(client):
    ids = [t["tramo_id"] for t in client.get("/tramos").json()][:3]
    r = client.post("/predict/batch", json={"peticiones": [{"tramo_id": t, "observaciones": EJEMPLO_OBS} for t in ids]})
    assert r.status_code == 200
    assert len(r.json()["predicciones"]) == len(ids)
