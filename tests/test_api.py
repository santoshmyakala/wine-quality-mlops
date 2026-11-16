import joblib
import pytest
from fastapi.testclient import TestClient

from wine_quality.features import FEATURES, TARGET, build_pipeline
from wine_quality.prepare import load_raw

SAMPLE = {
    "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0, "residual sugar": 1.9,
    "chlorides": 0.076, "free sulfur dioxide": 11, "total sulfur dioxide": 34, "density": 0.9978,
    "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4, "is_red": True,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    df = load_raw().sample(500, random_state=0)
    pipe = build_pipeline({"model": "ridge", "ridge": {}}, seed=0).fit(df[FEATURES], df[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    import wine_quality.api as api

    monkeypatch.setattr(api, "MODEL_PATH", path)
    with TestClient(api.app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200


def test_predict_returns_one_score_per_sample(client):
    r = client.post("/predict", json={"samples": [SAMPLE, SAMPLE]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and all(0 < p < 10 for p in preds)


def test_predict_validates_input(client):
    bad = {**SAMPLE, "pH": -1}
    assert client.post("/predict", json={"samples": [bad]}).status_code == 422
    assert client.post("/predict", json={"samples": []}).status_code == 422


def test_metrics_endpoint_exposes_prediction_counter(client):
    client.post("/predict", json={"samples": [SAMPLE]})
    assert "wine_predictions_total" in client.get("/metrics/").text
