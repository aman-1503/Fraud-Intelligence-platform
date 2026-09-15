"""Immediate fraud scoring. Investigation is deliberately outside this module."""
import json
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
import joblib

from app.core.config import get_decision, get_risk_level, settings
from app.ml.features import FeatureExtractor as BaseFeatureExtractor


class FeatureExtractor(BaseFeatureExtractor):
    FEATURE_NAMES = (BaseFeatureExtractor.TRANSACTION_FEATURES + BaseFeatureExtractor.MERCHANT_FEATURES
                     + BaseFeatureExtractor.VELOCITY_FEATURES + BaseFeatureExtractor.BEHAVIORAL_FEATURES
                     + BaseFeatureExtractor.GEOGRAPHIC_FEATURES + BaseFeatureExtractor.DEVICE_FEATURES)

    def extract(self, transaction, user_profile=None):
        profile = self.load_user_profile(transaction.get('user_id', ''), user_profile) if user_profile else None
        values = self.extract_features(_enrich_time(transaction), profile)
        return np.array([values[name] for name in self.FEATURE_NAMES], dtype=float)


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


class FraudScorer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.feature_names = []
        self.extractor = FeatureExtractor()
        self.model_version = 'rules-only'
        self.metrics = {'predictions_total': 0, 'avg_latency_ms': 0.0}

    async def initialize(self):
        path = Path(os.getenv('MODEL_PATH', settings.MODEL_PATH))
        model_file = path / 'xgboost_model.json'
        metadata_file = path / 'metadata.json'
        if model_file.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            self.feature_names = metadata['feature_names']
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_file)
            if (path / 'scaler.joblib').exists():
                self.scaler = joblib.load(path / 'scaler.joblib')
            if (path / 'calibrator.joblib').exists():
                self.calibrator = joblib.load(path / 'calibrator.joblib')
            self.model_version = metadata.get('version', settings.MODEL_VERSION)
        return self

    def _create_dummy_model(self):
        """Explicit deterministic test mode, not a trained model."""
        self.model = None
        self.model_version = 'rules-only'

    async def warmup(self):
        if self.model is not None:
            await self.score({'amount': 1.0, 'merchant_category': 'grocery', 'country': 'US'})

    def is_ready(self):
        return True

    async def score(self, transaction, user_profile=None):
        start = time.perf_counter()
        values = self.extractor.extract_features(_enrich_time(transaction),
            self.extractor.load_user_profile(transaction.get('user_id', ''), user_profile) if user_profile else None)
        factors = []
        amount = float(transaction.get('amount', 0))
        if amount >= 2000: factors.append('HIGH_AMOUNT')
        if values['is_high_risk_merchant']: factors.append('HIGH_RISK_MERCHANT')
        if values['is_high_risk_country']: factors.append('HIGH_RISK_COUNTRY')
        if values['is_foreign']: factors.append('FOREIGN_TRANSACTION')
        if values['is_new_device']: factors.append('NEW_DEVICE')
        if values['is_night']: factors.append('OFF_HOURS')
        if self.model is None:
            # Transparent rules-only mode until a trained artifact is supplied.
            probability = min(0.95, 0.05 + 0.18 * len(factors) + (0.12 if amount >= 5000 else 0))
            confidence = 0.65
        else:
            unknown = set(self.feature_names) - set(values)
            if unknown:
                raise ValueError(f'Model requires unavailable features: {sorted(unknown)}')
            matrix = np.array([[values[name] for name in self.feature_names]], dtype=float)
            if self.scaler is not None: matrix = self.scaler.transform(matrix)
            predictor = self.calibrator or self.model
            probability = float(predictor.predict_proba(matrix)[0, 1])
            confidence = 0.9
        risk = get_risk_level(probability)
        latency = (time.perf_counter() - start) * 1000
        count = self.metrics['predictions_total'] + 1
        self.metrics['avg_latency_ms'] += (latency - self.metrics['avg_latency_ms']) / count
        self.metrics['predictions_total'] = count
        return {'fraud_score': probability, 'risk_level': risk, 'decision': get_decision(probability, risk),
                'confidence': confidence, 'risk_factors': factors, 'latency_ms': latency}
