"""Model directory contract. Pickle files are refused."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb


class ArtifactError(ValueError):
    """The model directory failed its integrity contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LinearScaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.scale = np.where(self.scale == 0, 1.0, self.scale)

    def transform(self, matrix):
        return (matrix - self.mean) / self.scale


def publish(directory: Path, model: xgb.XGBClassifier, scaler, metadata: dict) -> None:
    """Write a booster, an optional scaler.json, and a hashed metadata file."""
    directory.mkdir(parents=True, exist_ok=True)
    names = list(metadata["feature_names"])
    model.get_booster().feature_names = names
    model_path = directory / "xgboost_model.json"
    model.save_model(model_path)
    for leftover in directory.glob("*.joblib"):
        leftover.unlink()
    body = dict(metadata)
    body["feature_names"] = names
    body["model_sha256"] = sha256_file(model_path)
    body["created_at"] = datetime.now(timezone.utc).isoformat()
    body["calibration"] = "not exported; joblib calibrators are refused at load"
    scaler_path = directory / "scaler.json"
    if scaler is not None:
        payload = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        scaler_path.write_text(json.dumps(payload))
        body["scaler_sha256"] = sha256_file(scaler_path)
    elif scaler_path.exists():
        scaler_path.unlink()
    (directory / "metadata.json").write_text(json.dumps(body, indent=2, default=str))


def publish_demo_model(directory: Path) -> None:
    """Small verified artifact whose only split is one-hour velocity."""
    from app.ml.features import FeatureExtractor

    names = FeatureExtractor().feature_names
    velocity = names.index("txn_count_1h")
    rng = np.random.default_rng(7)
    matrix = rng.normal(0, 0.01, size=(40, len(names)))
    matrix[:20, velocity] = 0
    matrix[20:, velocity] = 1
    labels = np.array([0] * 20 + [1] * 20)
    model = xgb.XGBClassifier(n_estimators=1, max_depth=1, n_jobs=1, base_score=0.5)
    model.fit(matrix, labels)
    publish(Path(directory), model, None, {"version": "demo-velocity", "feature_names": names})


def load_verified(directory: Path, expected_features: list[str]):
    """Load a model only when the hash, feature order, and booster names agree."""
    directory = Path(directory)
    pickles = list(directory.glob("*.joblib"))
    if pickles:
        raise ArtifactError(f"refusing pickle artifacts: {[path.name for path in pickles]}")
    metadata_path = directory / "metadata.json"
    model_path = directory / "xgboost_model.json"
    if not metadata_path.is_file() or not model_path.is_file():
        raise ArtifactError("model directory is missing metadata.json or xgboost_model.json")
    metadata = json.loads(metadata_path.read_text())
    expected_hash = metadata.get("model_sha256")
    if not expected_hash or sha256_file(model_path) != expected_hash:
        raise ArtifactError("model file hash does not match metadata")
    names = list(metadata.get("feature_names") or [])
    if names != list(expected_features):
        raise ArtifactError("metadata feature order does not match the scorer feature contract")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    booster_names = list(model.get_booster().feature_names or [])
    if booster_names != names:
        raise ArtifactError("booster feature order does not match metadata")
    scaler = None
    if metadata.get("scaler_sha256"):
        scaler_path = directory / "scaler.json"
        if not scaler_path.is_file() or sha256_file(scaler_path) != metadata["scaler_sha256"]:
            raise ArtifactError("scaler.json hash does not match metadata")
        raw = json.loads(scaler_path.read_text())
        scaler = LinearScaler(raw["mean"], raw["scale"])
        if len(scaler.mean) != len(names):
            raise ArtifactError("scaler width does not match feature order")
    return model, scaler, names, metadata.get("version", "unknown")
