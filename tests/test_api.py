import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import api
import evalkit

FEATURES = ["a__mean5", "b__std5", "c__delta10"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API pointed at a tiny forest trained on made-up numbers (no real data needed)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    path = tmp_path / "tiny.joblib"
    joblib.dump({"model": model, "features": FEATURES, "threshold": 0.5, "k_persist": 3,
                 "trained_on": "tiny test model"}, path)
    monkeypatch.setenv("ALFA_MODEL", str(path))
    api.load_bundle.cache_clear()
    yield TestClient(api.app)
    api.load_bundle.cache_clear()


def make_rows(n, seed=1):
    rng = np.random.default_rng(seed)
    return [dict(zip(FEATURES, map(float, rng.normal(size=3)))) for _ in range(n)]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["features"] == FEATURES
    assert body["k_persist"] == 3


def test_predict_reply_shape(client):
    r = client.post("/predict", json={"rows": make_rows(8)})
    assert r.status_code == 200
    body = r.json()
    assert body["n_seconds"] == 8
    assert len(body["seconds"]) == 8
    assert all(0.0 <= s["proba"] <= 1.0 for s in body["seconds"])
    assert [s["second"] for s in body["seconds"]] == list(range(8))


def test_missing_feature_gives_422(client):
    rows = make_rows(3)
    del rows[1]["c__delta10"]
    r = client.post("/predict", json={"rows": rows})
    assert r.status_code == 422
    assert r.json()["detail"]["missing_features"] == ["c__delta10"]


def test_empty_rows_rejected(client):
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_threshold_out_of_range_rejected(client):
    r = client.post("/predict", json={"rows": make_rows(5), "threshold": 1.5})
    assert r.status_code == 422


def test_threshold_zero_alarms_every_second_after_the_first_two(client):
    r = client.post("/predict", json={"rows": make_rows(10), "threshold": 0.0})
    body = r.json()
    assert all(s["raw_alarm"] for s in body["seconds"])
    assert body["n_alarm_seconds"] == 8          # 10 seconds minus the first k-1 = 2
    assert body["first_alarm_second"] == 2


def test_persistent_alarm_hand_example():
    raw = [1, 1, 0, 1, 1, 1, 0]
    expected = [False, False, False, False, False, True, False]
    assert api.persistent_alarm(raw, 3).tolist() == expected


def test_persistent_alarm_matches_evalkit():
    rng = np.random.default_rng(5)
    raw = rng.integers(0, 2, size=60).astype(bool)
    d = pd.DataFrame({"flight": ["f1"] * 60})
    assert api.persistent_alarm(raw, 3).tolist() == evalkit.persist(d, raw, 3).tolist()
