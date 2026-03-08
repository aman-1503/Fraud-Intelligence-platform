"""
Application Configuration
=========================
Central configuration management using Pydantic settings.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Fraud Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Performance targets
    TARGET_LATENCY_MS: int = 150
    MAX_BATCH_SIZE: int = 100
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_platform"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 20
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    USER_PROFILE_TTL_SECONDS: int = 3600  # 1 hour
    
    # ML Model
    MODEL_PATH: str = "models/fraud_model_v1"
    MODEL_VERSION: str = "v1.0.0"
    
    # Fraud thresholds
    THRESHOLD_LOW: float = 0.3
    THRESHOLD_MEDIUM: float = 0.6
    THRESHOLD_HIGH: float = 0.85
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    JWT_SECRET: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_risk_level(fraud_score: float) -> str:
    """Convert fraud score to risk level."""
    if fraud_score < settings.THRESHOLD_LOW:
        return "LOW"
    elif fraud_score < settings.THRESHOLD_MEDIUM:
        return "MEDIUM"
    elif fraud_score < settings.THRESHOLD_HIGH:
        return "HIGH"
    else:
        return "CRITICAL"


def get_decision(fraud_score: float, risk_level: str) -> str:
    """Get recommended decision based on score and risk level."""
    if risk_level == "LOW":
        return "APPROVE"
    elif risk_level == "MEDIUM":
        return "APPROVE"  # With monitoring
    elif risk_level == "HIGH":
        return "REVIEW"
    else:
        return "DECLINE"
