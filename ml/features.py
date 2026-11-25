"""Training entry point. The implementation lives in backend/app/ml/features.py."""
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parents[1] / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.ml.features import FeatureExtractor, UserProfile, extract_features_from_dataframe

__all__ = ["FeatureExtractor", "UserProfile", "extract_features_from_dataframe"]
