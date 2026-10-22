"""API smoke tests. Skipped when no trained model is present (e.g. before `dvc pull`)."""
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "models" / "final_model.joblib").exists(), reason="no trained model")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from serve.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_predict_and_drought_lowers_or_keeps_forecast(client):
    base = {"district": "Kurunegala", "season": "Yala"}
    normal = client.post("/predict", json=base | {"rain_pct_of_normal": 100}).json()
    assert 500 < normal["yield_kg_ha"] < 9000
    crisis = client.post("/predict", json=base | {"fertilizer_crisis": 1}).json()
    assert "shortfall" in crisis


def test_unknown_district_404(client):
    assert client.post("/predict", json={"district": "Atlantis", "season": "Maha"}).status_code == 404
