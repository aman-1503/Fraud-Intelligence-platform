"""Regressions from the production-readiness audit."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import app
from app.ml.artifacts import ArtifactError, sha256_file
from app.ml.scorer import FeatureExtractor, FraudScorer


def headers(role, subject="tester"):
    return {"Authorization": f"Bearer {create_access_token(subject, role)}"}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def body(transaction_id="TXN-1", amount=42.5, **extra):
    payload = {
        "transaction_id": transaction_id,
        "user_id": "U000001",
        "amount": amount,
        "merchant_category": "grocery",
        "country": "US",
        "currency": "USD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra)
    return payload


def test_unauthenticated_score_is_rejected(client):
    response = client.post("/api/v1/score", json=body())
    assert response.status_code == 401


def test_scorer_cannot_review_and_reviewer_cannot_score(client):
    scored = client.post("/api/v1/score", json=body("TXN-ROLE"), headers=headers("scorer"))
    assert scored.status_code == 200, scored.text
    review = client.get("/api/v1/review/cases", headers=headers("scorer"))
    assert review.status_code == 403
    denied = client.post("/api/v1/score", json=body("TXN-ROLE-2"), headers=headers("reviewer"))
    assert denied.status_code == 403


def test_same_id_different_amount_conflicts(client):
    auth = headers("scorer")
    stamp = datetime.now(timezone.utc).isoformat()
    first = client.post("/api/v1/score", json=body("TXN-IDEM", amount=80, timestamp=stamp), headers=auth)
    assert first.status_code == 200, first.text
    replay = client.post("/api/v1/score", json=body("TXN-IDEM", amount=80, timestamp=stamp), headers=auth)
    assert replay.status_code == 200
    assert replay.json()["fraud_score"] == first.json()["fraud_score"]
    conflict = client.post("/api/v1/score", json=body("TXN-IDEM", amount=90), headers=auth)
    assert conflict.status_code == 409
    assert "fraud_score" not in conflict.json()


def test_non_finite_and_oversized_input_rejected(client):
    auth = headers("scorer")
    infinite = client.post("/api/v1/score", json=body(amount=1e309), headers=auth)
    assert infinite.status_code == 422
    assert "nan" not in infinite.text.lower()
    missing = client.post("/api/v1/score", json={"amount": float("nan"), "transaction_id": "X"}, headers=auth)
    assert missing.status_code == 422
    huge = client.post("/api/v1/score", json=body(transaction_id="T" * 200_000), headers=auth)
    assert huge.status_code == 422
    history = client.get("/api/v1/user/U000001/transactions?limit=1000000", headers=auth)
    assert history.status_code == 422


def test_batch_reports_conflicts_without_a_fake_score(client):
    auth = headers("scorer")
    assert client.post("/api/v1/score", json=body("TXN-BATCH", amount=15), headers=auth).status_code == 200
    response = client.post("/api/v1/score/batch", json={"transactions": [
        body("TXN-BATCH", amount=99),
        body("TXN-BATCH-OK", amount=12),
    ]}, headers=auth)
    assert response.status_code == 200, response.text
    rows = {row["transaction_id"]: row for row in response.json()["results"]}
    assert rows["TXN-BATCH"]["error"] == "conflict"
    assert rows["TXN-BATCH"]["fraud_score"] is None
    assert rows["TXN-BATCH-OK"]["error"] is None
    assert "0.5" not in json.dumps(rows["TXN-BATCH"])


def test_metrics_are_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_http_requests_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")


def test_ready_reports_real_dependencies(client):
    response = client.get("/health/ready")
    assert response.status_code == 200, response.text
    assert response.json()["checks"] == {"model": True, "database": True, "cache": True}


def test_seeded_history_changes_the_score(client):
    auth = headers("admin", "velocity-tester")
    now = datetime.now(timezone.utc)
    first = client.post("/api/v1/score", json=body(
        "VEL-1", amount=40, user_id="VEL-USER", timestamp=now.isoformat()), headers=auth)
    second = client.post("/api/v1/score", json=body(
        "VEL-2", amount=40, user_id="VEL-USER",
        timestamp=(now + timedelta(minutes=1)).isoformat()), headers=auth)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["fraud_score"] != second.json()["fraud_score"]


def test_scrambled_feature_order_is_refused(tmp_path, monkeypatch):
    names = FeatureExtractor().feature_names
    matrix = np.random.default_rng(3).normal(size=(16, len(names)))
    labels = np.array([0, 1] * 8)
    model = xgb.XGBClassifier(n_estimators=1, max_depth=1, n_jobs=1)
    model.fit(matrix, labels)
    model.get_booster().feature_names = list(reversed(names))
    model.save_model(tmp_path / "xgboost_model.json")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "version": "scrambled",
        "feature_names": names,
        "model_sha256": sha256_file(Path(tmp_path / "xgboost_model.json")),
    }))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    with pytest.raises(ArtifactError, match="feature order"):
        import asyncio
        asyncio.run(FraudScorer().initialize())


@pytest.mark.skipif(not __import__("os").getenv("OPENAI_LIVE"), reason="set OPENAI_LIVE=1 with a real API key")
def test_live_openai_investigation(client):
    auth = headers("admin", "live-openai")
    scored = client.post("/api/v1/score", json=body("TXN-LIVE-OPENAI", amount=5000, country="NG",
                                                    merchant_category="crypto_exchange"), headers=auth)
    assert scored.status_code == 200, scored.text
    started = client.post("/api/v1/investigations/TXN-LIVE-OPENAI", headers=auth)
    assert started.status_code == 200, started.text
    assert started.json()["investigation_mode"] in {"openai", "deterministic"}
