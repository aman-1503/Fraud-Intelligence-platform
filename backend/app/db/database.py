"""
Database Module
===============
Async PostgreSQL database with connection pooling.
"""

import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

from app.core.config import settings
from app.core.logging import logger


class Database:
    """
    Async PostgreSQL database handler.
    Uses connection pooling for high throughput.
    """
    
    def __init__(self):
        """Initialize database connection settings."""
        self.pool = None
        self._connected = False
        
        # In-memory storage for demo (replace with actual DB in production)
        self._transactions: Dict[str, Dict] = {}
        self._users: Dict[str, Dict] = {}
        self._scores: Dict[str, Dict] = {}
    
    async def connect(self):
        """Establish database connection pool."""
        try:
            # For demo, we use in-memory storage
            # In production, use asyncpg:
            # import asyncpg
            # self.pool = await asyncpg.create_pool(
            #     settings.DATABASE_URL,
            #     min_size=5,
            #     max_size=settings.DATABASE_POOL_SIZE,
            # )
            
            self._connected = True
            logger.info("Database connected (in-memory mode)")
            
            # Initialize sample data
            await self._init_sample_data()
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
        self._connected = False
        logger.info("Database disconnected")
    
    async def health_check(self) -> bool:
        """Check database health."""
        return self._connected
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get database metrics."""
        return {
            "connected": self._connected,
            "transactions_count": len(self._transactions),
            "users_count": len(self._users),
        }
    
    async def _init_sample_data(self):
        """Initialize sample user data."""
        sample_users = [
            {
                "user_id": "U000001",
                "home_country": "US",
                "account_age_days": 365,
                "avg_txn_amount": 150.0,
                "std_txn_amount": 75.0,
                "avg_monthly_txns": 25,
                "primary_device_id": "device_001",
                "merchant_diversity": 0.6,
                "risk_score": 0.15,
            },
            {
                "user_id": "U000002",
                "home_country": "UK",
                "account_age_days": 180,
                "avg_txn_amount": 200.0,
                "std_txn_amount": 100.0,
                "avg_monthly_txns": 15,
                "primary_device_id": "device_002",
                "merchant_diversity": 0.4,
                "risk_score": 0.25,
            },
            {
                "user_id": "U000003",
                "home_country": "US",
                "account_age_days": 730,
                "avg_txn_amount": 500.0,
                "std_txn_amount": 200.0,
                "avg_monthly_txns": 40,
                "primary_device_id": "device_003",
                "merchant_diversity": 0.8,
                "risk_score": 0.10,
            },
        ]
        
        for user in sample_users:
            self._users[user["user_id"]] = user
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID."""
        return self._users.get(user_id)
    
    async def update_user_profile(self, user_id: str, profile: Dict):
        """Update user profile."""
        if user_id in self._users:
            self._users[user_id].update(profile)
        else:
            self._users[user_id] = profile
    
    async def get_transaction(self, transaction_id: str) -> Optional[Dict]:
        """Get transaction by ID."""
        return self._transactions.get(transaction_id)
    
    async def get_user_transactions(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[Dict]:
        """Get recent transactions for a user."""
        user_txns = [
            txn for txn in self._transactions.values()
            if txn.get('user_id') == user_id
        ]
        
        # Sort by timestamp descending
        user_txns.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return user_txns[:limit]
    
    async def log_transaction(
        self, 
        transaction: Dict, 
        score_result: Dict
    ):
        """Log transaction and score to database."""
        transaction_id = transaction.get('transaction_id')
        
        record = {
            **transaction,
            'fraud_score': score_result.get('fraud_score'),
            'risk_level': score_result.get('risk_level'),
            'decision': score_result.get('decision'),
            'risk_factors': score_result.get('risk_factors', []),
            'scored_at': datetime.utcnow().isoformat(),
        }
        
        self._transactions[transaction_id] = record
        self._scores[transaction_id] = score_result
    
    async def get_recent_transactions(
        self,
        limit: int = 100,
        fraud_only: bool = False
    ) -> List[Dict]:
        """Get recent transactions across all users."""
        txns = list(self._transactions.values())
        
        if fraud_only:
            txns = [t for t in txns if t.get('fraud_score', 0) > 0.5]
        
        txns.sort(
            key=lambda x: x.get('scored_at', ''),
            reverse=True
        )
        
        return txns[:limit]
    
    async def get_fraud_statistics(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get fraud statistics for the specified time period."""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        
        recent_txns = [
            t for t in self._transactions.values()
            if t.get('scored_at', '') != ''
        ]
        
        total = len(recent_txns)
        if total == 0:
            return {
                "total_transactions": 0,
                "fraud_count": 0,
                "fraud_rate": 0,
                "declined_count": 0,
                "review_count": 0,
            }
        
        fraud_count = sum(1 for t in recent_txns if t.get('fraud_score', 0) > 0.5)
        declined = sum(1 for t in recent_txns if t.get('decision') == 'DECLINE')
        review = sum(1 for t in recent_txns if t.get('decision') == 'REVIEW')
        
        return {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_count / total if total > 0 else 0,
            "declined_count": declined,
            "review_count": review,
        }


# SQL schema for production PostgreSQL
SQL_SCHEMA = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    home_country VARCHAR(3) NOT NULL DEFAULT 'US',
    account_age_days INTEGER NOT NULL DEFAULT 0,
    avg_txn_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    std_txn_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    avg_monthly_txns INTEGER NOT NULL DEFAULT 0,
    primary_device_id VARCHAR(64),
    merchant_diversity DECIMAL(5, 4) DEFAULT 0.5,
    risk_score DECIMAL(5, 4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    merchant_category VARCHAR(50) NOT NULL,
    merchant_category_code INTEGER,
    country VARCHAR(3) NOT NULL,
    is_foreign BOOLEAN DEFAULT FALSE,
    device_id VARCHAR(64),
    device_age_days INTEGER,
    is_new_device BOOLEAN DEFAULT FALSE,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_transactions_user_id (user_id),
    INDEX idx_transactions_timestamp (timestamp DESC)
);

-- Fraud scores table
CREATE TABLE IF NOT EXISTS fraud_scores (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id),
    fraud_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    risk_factors JSONB DEFAULT '[]',
    model_version VARCHAR(20) NOT NULL,
    latency_ms DECIMAL(10, 2) NOT NULL,
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_fraud_scores_transaction_id (transaction_id),
    INDEX idx_fraud_scores_scored_at (scored_at DESC),
    INDEX idx_fraud_scores_risk_level (risk_level)
);

-- User transaction history (for velocity features)
CREATE TABLE IF NOT EXISTS user_transaction_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    merchant_category VARCHAR(50) NOT NULL,
    country VARCHAR(3) NOT NULL,
    device_id VARCHAR(64),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Partition by time for efficient querying
    INDEX idx_user_history_user_timestamp (user_id, timestamp DESC)
);

-- Analytics aggregates
CREATE TABLE IF NOT EXISTS fraud_analytics_hourly (
    hour_bucket TIMESTAMP WITH TIME ZONE PRIMARY KEY,
    total_transactions INTEGER DEFAULT 0,
    fraud_count INTEGER DEFAULT 0,
    declined_count INTEGER DEFAULT 0,
    review_count INTEGER DEFAULT 0,
    total_amount DECIMAL(15, 2) DEFAULT 0,
    fraud_amount DECIMAL(15, 2) DEFAULT 0,
    avg_score DECIMAL(5, 4) DEFAULT 0,
    avg_latency_ms DECIMAL(10, 2) DEFAULT 0
);
"""
