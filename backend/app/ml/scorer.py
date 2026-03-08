"""
Fraud Scorer Module
===================
Real-time fraud scoring with XGBoost model.
Optimized for sub-150ms latency.
"""

import asyncio
import time
import numpy as np
import xgboost as xgb
import joblib
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
from collections import defaultdict
import threading

from app.core.config import settings, get_risk_level, get_decision
from app.core.logging import logger, tx_logger


class FeatureExtractor:
    """
    Fast feature extraction for real-time scoring.
    Mirrors the training pipeline feature engineering.
    """
    
    # Risk mappings (same as training)
    MERCHANT_RISK = {
        'grocery': 0.1, 'gas_station': 0.15, 'restaurant': 0.12,
        'online_retail': 0.3, 'electronics': 0.4, 'jewelry': 0.5,
        'travel': 0.25, 'entertainment': 0.15, 'atm_withdrawal': 0.2,
        'money_transfer': 0.6, 'crypto_exchange': 0.8, 'gambling': 0.55
    }
    
    HIGH_RISK_MERCHANTS = {'money_transfer', 'crypto_exchange', 'gambling', 'jewelry'}
    CASH_EQUIVALENT = {'atm_withdrawal', 'money_transfer', 'crypto_exchange'}
    
    COUNTRY_RISK = {
        'US': 0.1, 'UK': 0.12, 'CA': 0.11, 'DE': 0.12, 'FR': 0.13,
        'BR': 0.35, 'NG': 0.6, 'RU': 0.5, 'CN': 0.3, 'IN': 0.25
    }
    
    HIGH_RISK_COUNTRIES = {'NG', 'RU', 'BR'}
    
    FEATURE_NAMES = [
        'amount', 'amount_log', 'hour_of_day', 'day_of_week', 
        'is_weekend', 'is_night', 'is_high_risk_hour',
        'merchant_risk_score', 'is_high_risk_merchant', 
        'is_cash_equivalent', 'is_online',
        'txn_count_1h', 'txn_count_6h', 'txn_count_24h',
        'amount_sum_1h', 'amount_sum_24h',
        'unique_merchants_24h', 'unique_countries_24h',
        'time_since_last_txn_hours', 'velocity_score',
        'amount_zscore', 'amount_vs_avg_ratio',
        'is_above_95th_percentile', 'merchant_diversity_score',
        'spending_velocity_vs_normal',
        'is_foreign', 'is_high_risk_country', 'country_risk_score',
        'distance_from_home', 'is_impossible_travel'
    ]
    
    def extract(
        self, 
        transaction: Dict, 
        user_profile: Optional[Dict] = None,
        recent_transactions: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Extract features from a single transaction."""
        features = {}
        
        # Transaction features
        amount = float(transaction.get('amount', 0))
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)
        
        # Time features
        timestamp = transaction.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        hour = timestamp.hour
        features['hour_of_day'] = hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = float(timestamp.weekday() >= 5)
        features['is_night'] = float(hour >= 23 or hour <= 5)
        features['is_high_risk_hour'] = float(0 <= hour <= 6)
        
        # Merchant features
        merchant_cat = transaction.get('merchant_category', 'other')
        features['merchant_risk_score'] = self.MERCHANT_RISK.get(merchant_cat, 0.3)
        features['is_high_risk_merchant'] = float(merchant_cat in self.HIGH_RISK_MERCHANTS)
        features['is_cash_equivalent'] = float(merchant_cat in self.CASH_EQUIVALENT)
        features['is_online'] = float(merchant_cat in {'online_retail', 'crypto_exchange'})
        
        # Velocity features (from user history)
        velocity = self._compute_velocity(recent_transactions or [])
        features.update(velocity)
        
        # Behavioral features
        behavioral = self._compute_behavioral(amount, user_profile)
        features.update(behavioral)
        
        # Geographic features
        country = transaction.get('country', 'US')
        home_country = user_profile.get('home_country', 'US') if user_profile else 'US'
        
        features['is_foreign'] = float(country != home_country)
        features['is_high_risk_country'] = float(country in self.HIGH_RISK_COUNTRIES)
        features['country_risk_score'] = self.COUNTRY_RISK.get(country, 0.3)
        features['distance_from_home'] = 5000.0 if country != home_country else 0.0
        features['is_impossible_travel'] = 0.0  # Requires more sophisticated computation
        
        # Convert to array in correct order
        return np.array([features.get(name, 0.0) for name in self.FEATURE_NAMES])
    
    def _compute_velocity(self, recent_transactions: List[Dict]) -> Dict[str, float]:
        """Compute velocity features from recent transactions."""
        now = datetime.utcnow()
        
        txn_count_1h = 0
        txn_count_6h = 0
        txn_count_24h = 0
        amount_sum_1h = 0.0
        amount_sum_24h = 0.0
        merchants = set()
        countries = set()
        
        for txn in recent_transactions:
            txn_time = txn.get('timestamp')
            if isinstance(txn_time, str):
                txn_time = datetime.fromisoformat(txn_time.replace('Z', '+00:00'))
            if txn_time is None:
                continue
            
            hours_ago = (now - txn_time).total_seconds() / 3600
            
            if hours_ago <= 1:
                txn_count_1h += 1
                amount_sum_1h += txn.get('amount', 0)
            if hours_ago <= 6:
                txn_count_6h += 1
            if hours_ago <= 24:
                txn_count_24h += 1
                amount_sum_24h += txn.get('amount', 0)
                merchants.add(txn.get('merchant_category', ''))
                countries.add(txn.get('country', ''))
        
        velocity_score = (
            txn_count_1h * 3 +
            txn_count_6h * 0.5 +
            txn_count_24h * 0.1 +
            min(1, amount_sum_1h / 1000) * 2
        )
        
        return {
            'txn_count_1h': float(txn_count_1h),
            'txn_count_6h': float(txn_count_6h),
            'txn_count_24h': float(txn_count_24h),
            'amount_sum_1h': amount_sum_1h,
            'amount_sum_24h': amount_sum_24h,
            'unique_merchants_24h': float(len(merchants)),
            'unique_countries_24h': float(len(countries)),
            'time_since_last_txn_hours': 24.0,  # Default
            'velocity_score': min(10, velocity_score),
        }
    
    def _compute_behavioral(
        self, 
        amount: float, 
        user_profile: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute behavioral deviation features."""
        if not user_profile:
            return {
                'amount_zscore': 0.0,
                'amount_vs_avg_ratio': 1.0,
                'is_above_95th_percentile': 0.0,
                'merchant_diversity_score': 0.5,
                'spending_velocity_vs_normal': 1.0,
            }
        
        avg_amount = user_profile.get('avg_txn_amount', 100.0)
        std_amount = max(user_profile.get('std_txn_amount', 50.0), 1.0)
        
        zscore = (amount - avg_amount) / std_amount
        threshold_95 = avg_amount + 2 * std_amount
        
        return {
            'amount_zscore': zscore,
            'amount_vs_avg_ratio': amount / max(avg_amount, 1.0),
            'is_above_95th_percentile': float(amount > threshold_95),
            'merchant_diversity_score': user_profile.get('merchant_diversity', 0.5),
            'spending_velocity_vs_normal': 1.0,
        }


class FraudScorer:
    """
    Real-time fraud scoring engine.
    Thread-safe with model caching.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize scorer with model path."""
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.model_version = settings.MODEL_VERSION
        self.metadata: Dict = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = defaultdict(float)
        self._prediction_times: List[float] = []
    
    async def initialize(self):
        """Load model asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
    
    def _load_model(self):
        """Load model from disk (synchronous)."""
        try:
            # Load XGBoost model
            model_file = self.model_path / 'xgboost_model.json'
            if model_file.exists():
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_file)
                logger.info(f"Loaded XGBoost model from {model_file}")
            else:
                # Create a dummy model for demo
                logger.warning("Model file not found, creating dummy model")
                self._create_dummy_model()
            
            # Load calibrator
            calibrator_file = self.model_path / 'calibrator.joblib'
            if calibrator_file.exists():
                self.calibrator = joblib.load(calibrator_file)
                logger.info("Loaded probability calibrator")
            
            # Load scaler
            scaler_file = self.model_path / 'scaler.joblib'
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info("Loaded feature scaler")
            
            # Load metadata
            metadata_file = self.model_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    self.metadata = json.load(f)
                self.model_version = self.metadata.get('version', self.model_version)
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a simple dummy model for demonstration."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Create dummy data
        np.random.seed(42)
        n_features = len(FeatureExtractor.FEATURE_NAMES)
        X = np.random.randn(1000, n_features)
        y = (X[:, 0] + X[:, 7] > 0.5).astype(int)
        
        # Fit simple model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Use XGBoost with simple params
        self.model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        logger.info("Created dummy model for demonstration")
    
    async def warmup(self, n_samples: int = 100):
        """Warm up the model with dummy predictions."""
        logger.info(f"Warming up model with {n_samples} predictions...")
        
        for _ in range(n_samples):
            dummy_txn = {
                'amount': np.random.uniform(10, 1000),
                'merchant_category': 'grocery',
                'country': 'US',
                'timestamp': datetime.utcnow(),
            }
            await self.score(dummy_txn)
        
        logger.info("Model warmup complete")
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    async def score(
        self, 
        transaction: Dict,
        user_profile: Optional[Dict] = None,
        recent_transactions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Score a transaction for fraud risk.
        
        Args:
            transaction: Transaction data
            user_profile: User profile for behavioral analysis
            recent_transactions: Recent transaction history
            
        Returns:
            Score result with fraud probability and risk factors
        """
        start_time = time.perf_counter()
        
        try:
            # Extract features
            features = self.feature_extractor.extract(
                transaction, 
                user_profile, 
                recent_transactions
            )
            
            # Scale features
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Get prediction
            if self.calibrator is not None:
                fraud_prob = self.calibrator.predict_proba(features)[0, 1]
            else:
                fraud_prob = self.model.predict_proba(features)[0, 1]
            
            # Compute confidence (inverse of uncertainty)
            confidence = 1 - 2 * abs(fraud_prob - 0.5)
            
            # Get risk level and decision
            risk_level = get_risk_level(fraud_prob)
            decision = get_decision(fraud_prob, risk_level)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(transaction, user_profile, fraud_prob)
            
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(latency_ms)
            
            # Log transaction
            tx_logger.log_score(
                transaction_id=transaction.get('transaction_id', 'unknown'),
                user_id=transaction.get('user_id', 'unknown'),
                fraud_score=fraud_prob,
                decision=decision,
                latency_ms=latency_ms,
            )
            
            return {
                'fraud_score': float(fraud_prob),
                'risk_level': risk_level,
                'decision': decision,
                'confidence': float(confidence),
                'risk_factors': risk_factors,
                'latency_ms': latency_ms,
            }
            
        except Exception as e:
            logger.error(f"Scoring error: {e}", exc_info=True)
            return {
                'fraud_score': 0.5,
                'risk_level': 'UNKNOWN',
                'decision': 'REVIEW',
                'confidence': 0.0,
                'risk_factors': ['SCORING_ERROR'],
                'latency_ms': (time.perf_counter() - start_time) * 1000,
            }
    
    def _identify_risk_factors(
        self, 
        transaction: Dict,
        user_profile: Optional[Dict],
        fraud_score: float
    ) -> List[str]:
        """Identify contributing risk factors."""
        factors = []
        
        # High amount
        amount = transaction.get('amount', 0)
        if amount > 1000:
            factors.append('HIGH_AMOUNT')
        
        # High-risk merchant
        merchant = transaction.get('merchant_category', '')
        if merchant in FeatureExtractor.HIGH_RISK_MERCHANTS:
            factors.append('HIGH_RISK_MERCHANT')
        
        # High-risk country
        country = transaction.get('country', 'US')
        if country in FeatureExtractor.HIGH_RISK_COUNTRIES:
            factors.append('HIGH_RISK_COUNTRY')
        
        # Foreign transaction
        if user_profile:
            home_country = user_profile.get('home_country', 'US')
            if country != home_country:
                factors.append('FOREIGN_TRANSACTION')
        
        # New device
        if transaction.get('is_new_device', False):
            factors.append('NEW_DEVICE')
        
        # Off-hours transaction
        timestamp = transaction.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = timestamp.hour
            if 0 <= hour <= 6:
                factors.append('OFF_HOURS')
        
        # Amount deviation
        if user_profile:
            avg_amount = user_profile.get('avg_txn_amount', 100)
            if amount > avg_amount * 5:
                factors.append('UNUSUAL_AMOUNT')
        
        return factors
    
    def _update_metrics(self, latency_ms: float):
        """Update scoring metrics."""
        with self._lock:
            self.metrics['predictions_total'] += 1
            self._prediction_times.append(latency_ms)
            
            # Keep only last 1000 predictions for metrics
            if len(self._prediction_times) > 1000:
                self._prediction_times = self._prediction_times[-1000:]
            
            self.metrics['avg_latency_ms'] = np.mean(self._prediction_times)
            self.metrics['p50_latency_ms'] = np.percentile(self._prediction_times, 50)
            self.metrics['p95_latency_ms'] = np.percentile(self._prediction_times, 95)
            self.metrics['p99_latency_ms'] = np.percentile(self._prediction_times, 99)
