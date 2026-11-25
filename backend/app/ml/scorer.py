"""Immediate fraud scoring. A missing artifact does not invent a probability."""
import os
import time
from pathlib import Path

import numpy as np

from app.core.config import get_decision, get_risk_level, settings
from app.ml.artifacts import ArtifactError, load_verified
from app.ml.features import FeatureExtractor as BaseFeatureExtractor


class FeatureExtractor(BaseFeatureExtractor):
    FEATURE_NAMES = (BaseFeatureExtractor.TRANSACTION_FEATURES + BaseFeatureExtractor.MERCHANT_FEATURES
                     + BaseFeatureExtractor.VELOCITY_FEATURES + BaseFeatureExtractor.BEHAVIORAL_FEATURES
                     + BaseFeatureExtractor.GEOGRAPHIC_FEATURES + BaseFeatureExtractor.DEVICE_FEATURES)

    def extract(self, transaction, user_profile=None):
        profile = self.load_user_profile(transaction.get('user_id', ''), user_profile) if user_profile else None
        values = self.extract_features(_enrich_time(transaction), profile)
        return np.array([values[name] for name in self.FEATURE_NAMES], dtype=float)


class ModelNotReady(RuntimeError):
    """No verified artifact is loaded."""


def _enrich_time(transaction):
    enriched = dict(transaction)
    timestamp = enriched.get('timestamp')
    if timestamp:
        if isinstance(timestamp, str):
            from datetime import datetime
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        enriched['hour_of_day'] = timestamp.hour
        enriched['day_of_week'] = timestamp.weekday()
        enriched['is_weekend'] = timestamp.weekday() >= 5
    return enriched


def risk_factors(values, amount) -> list[str]:
    factors = []
    if amount >= 2000:
        factors.append('HIGH_AMOUNT')
    if values['is_high_risk_merchant']:
        factors.append('HIGH_RISK_MERCHANT')
    if values['is_high_risk_country']:
        factors.append('HIGH_RISK_COUNTRY')
    if values['is_foreign']:
        factors.append('FOREIGN_TRANSACTION')
    if values['is_new_device']:
        factors.append('NEW_DEVICE')
    if values['is_night']:
        factors.append('OFF_HOURS')
    return factors


class FraudScorer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.extractor = FeatureExtractor()
        self.model_version = 'unloaded'
        self.metrics = {'predictions_total': 0, 'avg_latency_ms': 0.0}
        self.load_error = None

    async def initialize(self):
        path = Path(os.getenv('MODEL_PATH', settings.MODEL_PATH))
        if not (path / 'xgboost_model.json').exists():
            self.model = None
            self.model_version = 'unloaded'
            self.load_error = 'model artifact not found'
            return self
        try:
            self.model, self.scaler, self.feature_names, self.model_version = load_verified(
                path, self.extractor.feature_names
            )
            self.load_error = None
        except (ArtifactError, OSError, ValueError) as exc:
            self.model = None
            self.model_version = 'unloaded'
            self.load_error = str(exc)
            raise
        return self

    async def warmup(self):
        if self.model is not None:
            await self.score({'amount': 1.0, 'merchant_category': 'grocery', 'country': 'US', 'user_id': 'warmup'})

    def is_ready(self):
        return self.model is not None

    async def score(self, transaction, user_profile=None):
        if self.model is None:
            raise ModelNotReady(self.load_error or 'model artifact not loaded')
        start = time.perf_counter()
        prepared = _enrich_time(transaction)
        profile = None
        if user_profile:
            if user_profile.get('primary_device_id') and prepared.get('device_id'):
                prepared['is_new_device'] = user_profile['primary_device_id'] != prepared['device_id']
            profile = self.extractor.load_user_profile(prepared.get('user_id', ''), user_profile)
        values = self.extractor.extract_features(prepared, profile)
        factors = risk_factors(values, float(transaction.get('amount', 0)))
        matrix = np.array([[values[name] for name in self.feature_names]], dtype=float)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        probability = float(self.model.predict_proba(matrix)[0, 1])
        confidence = float(max(probability, 1.0 - probability))
        risk = get_risk_level(probability)
        latency = (time.perf_counter() - start) * 1000
        count = self.metrics['predictions_total'] + 1
        self.metrics['avg_latency_ms'] += (latency - self.metrics['avg_latency_ms']) / count
        self.metrics['predictions_total'] = count
        return {
            'fraud_score': probability,
            'risk_level': risk,
            'decision': get_decision(probability, risk),
            'confidence': confidence,
            'risk_factors': factors,
            'latency_ms': latency,
            'model_version': self.model_version,
        }
