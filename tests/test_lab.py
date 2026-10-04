"""Tests: datos, métrica de negocio y el flujo MLflow completo (tracking -> registry -> API)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import build_pipeline, evaluate, load_data, profit_curve


def test_data_split():
    X_train, X_test, y_train, y_test = load_data()
    assert len(X_train) + len(X_test) == 7043
    assert not X_train.isna().any().any()
    assert abs(y_train.mean() - y_test.mean()) < 0.01  # split estratificado


def test_profit_curve_math():
    y, proba = np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    b = {"clv_saved": 500, "offer_cost": 60, "success_rate": 0.5}
    curve = profit_curve(y, proba, b).set_index("threshold")
    assert curve.loc[0.75, "profit"] == pytest.approx(380)  # 2*0.5*500 - 2*60
    assert curve.loc[0.50, "profit"] == pytest.approx(320)  # 500 - 3*60


def test_model_beats_random():
    X_train, X_test, y_train, y_test = load_data()
    pipe = build_pipeline("logreg").fit(X_train, y_train)
    metrics, _ = evaluate(y_test, pipe.predict_proba(X_test)[:, 1])
    assert metrics["test_auc"] > 0.80


def test_mlflow_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.chdir(tmp_path)
    import promote
    import register
    from common import setup_mlflow
    from train import train_and_log

    setup_mlflow()
    train_and_log("logreg", None, *load_data())
    register.main([])
    assert promote.main() == 0

    from fastapi.testclient import TestClient

    import api

    client = TestClient(api.app)
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/predict", json={}).json()
    assert 0 <= body["churn_probability"] <= 1
