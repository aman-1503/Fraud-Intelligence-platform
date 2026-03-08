"""
Fraud Intelligence Platform - Test Suite
=========================================
Unit and integration tests for the fraud scoring API.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# Test imports
import sys
sys.path.insert(0, '.')

from app.core.config import settings, get_risk_level, get_decision
from app.ml.scorer import FraudScorer, FeatureExtractor


# ==========================================
# Configuration Tests
# ==========================================

class TestConfig:
    """Test configuration module."""
    
    def test_risk_level_low(self):
        assert get_risk_level(0.1) == "LOW"
        assert get_risk_level(0.29) == "LOW"
    
    def test_risk_level_medium(self):
        assert get_risk_level(0.31) == "MEDIUM"
        assert get_risk_level(0.59) == "MEDIUM"
    
    def test_risk_level_high(self):
        assert get_risk_level(0.61) == "HIGH"
        assert get_risk_level(0.84) == "HIGH"
    
    def test_risk_level_critical(self):
        assert get_risk_level(0.86) == "CRITICAL"
        assert get_risk_level(0.99) == "CRITICAL"
    
    def test_decision_approve(self):
        assert get_decision(0.1, "LOW") == "APPROVE"
        assert get_decision(0.4, "MEDIUM") == "APPROVE"
    
    def test_decision_review(self):
        assert get_decision(0.7, "HIGH") == "REVIEW"
    
    def test_decision_decline(self):
        assert get_decision(0.9, "CRITICAL") == "DECLINE"


# ==========================================
# Feature Extractor Tests
# ==========================================

class TestFeatureExtractor:
    """Test feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    @pytest.fixture
    def sample_transaction(self):
        return {
            'transaction_id': 'TXN-001',
            'user_id': 'U000001',
            'amount': 150.00,
            'merchant_category': 'electronics',
            'country': 'US',
            'device_id': 'device_001',
            'timestamp': datetime.utcnow(),
        }
    
    @pytest.fixture
    def sample_user_profile(self):
        return {
            'home_country': 'US',
            'account_age_days': 365,
            'avg_txn_amount': 100.0,
            'std_txn_amount': 50.0,
            'avg_monthly_txns': 20,
            'primary_device_id': 'device_001',
            'merchant_diversity': 0.6,
        }
    
    def test_extract_basic_features(self, extractor, sample_transaction):
        features = extractor.extract(sample_transaction)
        
        assert len(features) == len(FeatureExtractor.FEATURE_NAMES)
        assert features[0] == 150.00  # amount
        assert features[1] == np.log1p(150.00)  # amount_log
    
    def test_extract_with_user_profile(self, extractor, sample_transaction, sample_user_profile):
        features = extractor.extract(sample_transaction, sample_user_profile)
        
        # Check behavioral features are computed
        amount_zscore_idx = FeatureExtractor.FEATURE_NAMES.index('amount_zscore')
        assert features[amount_zscore_idx] == (150.0 - 100.0) / 50.0  # (amount - avg) / std
    
    def test_merchant_risk_scores(self, extractor):
        high_risk_txn = {
            'amount': 100.0,
            'merchant_category': 'crypto_exchange',
            'country': 'US',
            'timestamp': datetime.utcnow(),
        }
        
        low_risk_txn = {
            'amount': 100.0,
            'merchant_category': 'grocery',
            'country': 'US',
            'timestamp': datetime.utcnow(),
        }
        
        high_risk_features = extractor.extract(high_risk_txn)
        low_risk_features = extractor.extract(low_risk_txn)
        
        merchant_risk_idx = FeatureExtractor.FEATURE_NAMES.index('merchant_risk_score')
        assert high_risk_features[merchant_risk_idx] > low_risk_features[merchant_risk_idx]
    
    def test_foreign_transaction_detection(self, extractor, sample_user_profile):
        foreign_txn = {
            'amount': 100.0,
            'merchant_category': 'grocery',
            'country': 'NG',  # Nigeria
            'timestamp': datetime.utcnow(),
        }
        
        features = extractor.extract(foreign_txn, sample_user_profile)
        
        is_foreign_idx = FeatureExtractor.FEATURE_NAMES.index('is_foreign')
        is_high_risk_country_idx = FeatureExtractor.FEATURE_NAMES.index('is_high_risk_country')
        
        assert features[is_foreign_idx] == 1.0
        assert features[is_high_risk_country_idx] == 1.0
    
    def test_time_features(self, extractor):
        night_txn = {
            'amount': 100.0,
            'merchant_category': 'grocery',
            'country': 'US',
            'timestamp': datetime(2024, 1, 15, 3, 0, 0),  # 3 AM
        }
        
        features = extractor.extract(night_txn)
        
        is_night_idx = FeatureExtractor.FEATURE_NAMES.index('is_night')
        is_high_risk_hour_idx = FeatureExtractor.FEATURE_NAMES.index('is_high_risk_hour')
        
        assert features[is_night_idx] == 1.0
        assert features[is_high_risk_hour_idx] == 1.0


# ==========================================
# Fraud Scorer Tests
# ==========================================

class TestFraudScorer:
    """Test fraud scoring engine."""
    
    @pytest.fixture
    def scorer(self):
        scorer = FraudScorer()
        scorer._create_dummy_model()
        return scorer
    
    @pytest.fixture
    def sample_transaction(self):
        return {
            'transaction_id': 'TXN-001',
            'user_id': 'U000001',
            'amount': 150.00,
            'merchant_category': 'electronics',
            'country': 'US',
            'device_id': 'device_001',
            'timestamp': datetime.utcnow(),
        }
    
    @pytest.mark.asyncio
    async def test_score_transaction(self, scorer, sample_transaction):
        result = await scorer.score(sample_transaction)
        
        assert 'fraud_score' in result
        assert 'risk_level' in result
        assert 'decision' in result
        assert 'confidence' in result
        assert 'latency_ms' in result
        
        assert 0 <= result['fraud_score'] <= 1
        assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        assert result['decision'] in ['APPROVE', 'REVIEW', 'DECLINE']
    
    @pytest.mark.asyncio
    async def test_score_with_user_profile(self, scorer, sample_transaction):
        user_profile = {
            'home_country': 'US',
            'avg_txn_amount': 100.0,
            'std_txn_amount': 50.0,
        }
        
        result = await scorer.score(sample_transaction, user_profile)
        
        assert 'fraud_score' in result
        assert result['latency_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_high_risk_transaction(self, scorer):
        high_risk_txn = {
            'transaction_id': 'TXN-SUSPICIOUS',
            'user_id': 'U000001',
            'amount': 5000.00,
            'merchant_category': 'crypto_exchange',
            'country': 'NG',
            'is_new_device': True,
            'timestamp': datetime(2024, 1, 15, 3, 0, 0),
        }
        
        result = await scorer.score(high_risk_txn)
        
        # High-risk transactions should have risk factors
        assert len(result.get('risk_factors', [])) > 0
    
    @pytest.mark.asyncio
    async def test_risk_factor_identification(self, scorer):
        txn_with_factors = {
            'transaction_id': 'TXN-002',
            'user_id': 'U000001',
            'amount': 5000.00,  # High amount
            'merchant_category': 'money_transfer',  # High risk merchant
            'country': 'RU',  # High risk country
            'is_new_device': True,
            'timestamp': datetime(2024, 1, 15, 2, 0, 0),  # Off hours
        }
        
        user_profile = {
            'home_country': 'US',
            'avg_txn_amount': 100.0,
        }
        
        result = await scorer.score(txn_with_factors, user_profile)
        risk_factors = result.get('risk_factors', [])
        
        assert 'HIGH_AMOUNT' in risk_factors
        assert 'HIGH_RISK_MERCHANT' in risk_factors
        assert 'HIGH_RISK_COUNTRY' in risk_factors
        assert 'FOREIGN_TRANSACTION' in risk_factors
        assert 'NEW_DEVICE' in risk_factors
        assert 'OFF_HOURS' in risk_factors
    
    def test_is_ready(self, scorer):
        assert scorer.is_ready() == True
    
    @pytest.mark.asyncio
    async def test_latency_tracking(self, scorer, sample_transaction):
        # Run multiple predictions
        for _ in range(10):
            await scorer.score(sample_transaction)
        
        assert scorer.metrics['predictions_total'] >= 10
        assert scorer.metrics['avg_latency_ms'] > 0


# ==========================================
# API Endpoint Tests (Integration)
# ==========================================

class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        # Mock dependencies
        with patch('app.main.fraud_scorer') as mock_scorer:
            mock_scorer.score = AsyncMock(return_value={
                'fraud_score': 0.15,
                'risk_level': 'LOW',
                'decision': 'APPROVE',
                'confidence': 0.92,
                'risk_factors': [],
            })
            mock_scorer.model_version = 'v1.0.0'
            mock_scorer.is_ready = MagicMock(return_value=True)
            
            with patch('app.main.database') as mock_db:
                mock_db.get_user_profile = AsyncMock(return_value=None)
                mock_db.log_transaction = AsyncMock()
                mock_db.health_check = AsyncMock(return_value=True)
                
                with patch('app.main.redis_cache') as mock_cache:
                    mock_cache.get_score = AsyncMock(return_value=None)
                    mock_cache.set_score = AsyncMock()
                    mock_cache.get_user_profile = AsyncMock(return_value=None)
                    mock_cache.health_check = AsyncMock(return_value=True)
                    
                    yield TestClient(app)
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_score_endpoint(self, client):
        transaction = {
            "transaction_id": "TXN-TEST-001",
            "user_id": "U000001",
            "amount": 150.00,
            "merchant_category": "electronics",
            "country": "US",
        }
        
        response = client.post("/api/v1/score", json=transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_score" in data
        assert "risk_level" in data
        assert "decision" in data
    
    def test_score_validation_error(self, client):
        invalid_transaction = {
            "transaction_id": "TXN-TEST-001",
            # Missing required fields
        }
        
        response = client.post("/api/v1/score", json=invalid_transaction)
        assert response.status_code == 422  # Validation error


# ==========================================
# Performance Tests
# ==========================================

class TestPerformance:
    """Performance and latency tests."""
    
    @pytest.fixture
    def scorer(self):
        scorer = FraudScorer()
        scorer._create_dummy_model()
        return scorer
    
    @pytest.mark.asyncio
    async def test_single_prediction_latency(self, scorer):
        """Single prediction should be under 50ms."""
        import time
        
        txn = {
            'transaction_id': 'TXN-PERF-001',
            'user_id': 'U000001',
            'amount': 100.0,
            'merchant_category': 'grocery',
            'country': 'US',
            'timestamp': datetime.utcnow(),
        }
        
        start = time.perf_counter()
        await scorer.score(txn)
        latency_ms = (time.perf_counter() - start) * 1000
        
        assert latency_ms < 50, f"Latency {latency_ms}ms exceeds 50ms target"
    
    @pytest.mark.asyncio
    async def test_batch_throughput(self, scorer):
        """Should handle 100 predictions quickly."""
        import time
        
        transactions = [
            {
                'transaction_id': f'TXN-BATCH-{i}',
                'user_id': f'U{i:06d}',
                'amount': float(i * 10),
                'merchant_category': 'grocery',
                'country': 'US',
                'timestamp': datetime.utcnow(),
            }
            for i in range(100)
        ]
        
        start = time.perf_counter()
        for txn in transactions:
            await scorer.score(txn)
        total_time = time.perf_counter() - start
        
        throughput = 100 / total_time
        assert throughput > 500, f"Throughput {throughput:.0f}/s below 500/s target"


# ==========================================
# Run Tests
# ==========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
