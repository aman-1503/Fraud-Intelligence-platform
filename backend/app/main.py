"""
Fraud Intelligence Platform - FastAPI Backend
==============================================
Real-time fraud scoring API with sub-150ms latency.
Handles 3,000+ RPS with connection pooling and async I/O.
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.ml.scorer import FraudScorer
from app.db.database import Database
from app.db.redis_cache import RedisCache
from app.investigation.models import Case, InvestigationResult, Action
from app.investigation.workflow import InvestigationWorkflow
from app.archive.repository import ArchiveRepository
from app.live.bitcoin import LiveBitcoinFeed, BitcoinFeedResult


# Global instances
fraud_scorer: Optional[FraudScorer] = None
database: Optional[Database] = None
redis_cache: Optional[RedisCache] = None
live_bitcoin_feed = LiveBitcoinFeed()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global fraud_scorer, database, redis_cache
    
    logger.info("Starting Fraud Intelligence Platform...")
    
    # Initialize ML model
    logger.info("Loading fraud detection model...")
    fraud_scorer = FraudScorer()
    await fraud_scorer.initialize()
    logger.info(f"Model loaded: {fraud_scorer.model_version}")
    
    # Initialize database
    logger.info("Connecting to database...")
    database = Database()
    await database.connect()
    logger.info("Database connected")
    
    # Initialize Redis cache
    logger.info("Connecting to Redis...")
    redis_cache = RedisCache()
    await redis_cache.connect()
    logger.info("Redis connected")
    
    # Warmup model
    logger.info("Warming up model...")
    await fraud_scorer.warmup()
    logger.info("Model warmup complete")
    
    logger.info("=" * 50)
    logger.info("FRAUD INTELLIGENCE PLATFORM READY")
    logger.info(f"Model version: {fraud_scorer.model_version}")
    logger.info(f"Target latency: <{settings.TARGET_LATENCY_MS}ms")
    logger.info("=" * 50)
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if database:
        await database.disconnect()
    if redis_cache:
        await redis_cache.disconnect()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Fraud Intelligence Platform",
    description="Real-time fraud detection API with XGBoost ML pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add timing information to response headers."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Dependency injection
def get_scorer() -> FraudScorer:
    """Get fraud scorer instance."""
    if fraud_scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized")
    return fraud_scorer


def get_database() -> Database:
    """Get database instance."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return database


def get_cache() -> RedisCache:
    """Get Redis cache instance."""
    if redis_cache is None:
        raise HTTPException(status_code=503, detail="Cache not connected")
    return redis_cache


# Pydantic models
class TransactionRequest(BaseModel):
    """Incoming transaction for fraud scoring."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User/account identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    currency: str = Field(default="USD", description="Currency code")
    merchant_category: str = Field(..., description="Merchant category code")
    country: str = Field(default="US", description="Transaction country")
    device_id: Optional[str] = Field(None, description="Device fingerprint")
    timestamp: Optional[datetime] = Field(None, description="Transaction timestamp")
    
    # Optional enrichment fields
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-12345-ABCDE",
                "user_id": "U000123",
                "amount": 150.00,
                "currency": "USD",
                "merchant_category": "electronics",
                "country": "US",
                "device_id": "d8f7e6c5b4a3",
            }
        }


class FraudScore(BaseModel):
    """Fraud scoring response."""
    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability 0-1")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH, CRITICAL")
    decision: str = Field(..., description="Recommended action: APPROVE, REVIEW, DECLINE")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    
    # Timing
    latency_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Risk factors
    risk_factors: list[str] = Field(default_factory=list, description="Contributing risk factors")
    
    # Metadata
    model_version: str = Field(..., description="Model version used")
    scored_at: datetime = Field(..., description="Scoring timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-12345-ABCDE",
                "fraud_score": 0.15,
                "risk_level": "LOW",
                "decision": "APPROVE",
                "confidence": 0.92,
                "latency_ms": 45.2,
                "risk_factors": [],
                "model_version": "v1.0.0",
                "scored_at": "2024-01-15T10:30:00Z"
            }
        }


class BatchRequest(BaseModel):
    """Batch scoring request."""
    transactions: list[TransactionRequest] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    """Batch scoring response."""
    results: list[FraudScore]
    total_count: int
    avg_latency_ms: float
    total_latency_ms: float


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1)
    final_action: Action
    note: str = Field(min_length=1)


def get_archive() -> ArchiveRepository:
    archive = ArchiveRepository(settings.ARCHIVE_PATH)
    if not archive.available():
        raise HTTPException(status_code=503, detail="Real-world archive unavailable; run ml/train_real.py")
    return archive


@app.get("/api/v1/archive/metadata", tags=["Real-world archive"])
async def archive_metadata(archive: ArchiveRepository = Depends(get_archive)):
    """Measured holdout metrics and source provenance; historical data, not a live feed."""
    return archive.metadata()


@app.get("/api/v1/archive/rows", tags=["Real-world archive"])
async def archive_rows(after: int = -1, limit: int = 20, known_fraud_only: bool = False,
                       archive: ArchiveRepository = Depends(get_archive)):
    """Replay held-out rows in their original time order without invented enrichment."""
    try:
        return archive.rows(after, limit, known_fraud_only)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def get_live_bitcoin_feed() -> LiveBitcoinFeed:
    return live_bitcoin_feed


@app.get("/api/v1/live/bitcoin/recent", response_model=BitcoinFeedResult, tags=["Live public data"])
async def recent_bitcoin_transactions(feed: LiveBitcoinFeed = Depends(get_live_bitcoin_feed)):
    """Live unconfirmed Bitcoin activity; these transactions are not fraud-labeled."""
    try:
        return await feed.recent()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Public Bitcoin source unavailable: %s", exc)
        raise HTTPException(status_code=502, detail="Public Bitcoin feed unavailable")


# API Endpoints
@app.post("/api/v1/score", response_model=FraudScore, tags=["Scoring"])
async def score_transaction(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    """
    Score a single transaction for fraud risk.
    
    Returns fraud probability, risk level, and recommended decision.
    Target latency: <150ms P99
    """
    start_time = time.perf_counter()
    
    try:
        # Check cache first
        cached_score = await cache.get_score(request.transaction_id)
        if cached_score:
            cached_score['latency_ms'] = (time.perf_counter() - start_time) * 1000
            return FraudScore(**cached_score)
        
        # Get user profile from cache/db
        user_profile = await cache.get_user_profile(request.user_id)
        if not user_profile:
            user_profile = await db.get_user_profile(request.user_id)
            if user_profile:
                await cache.set_user_profile(request.user_id, user_profile)
        
        # Score transaction
        score_result = await scorer.score(request.model_dump(), user_profile)
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Build response
        response = FraudScore(
            transaction_id=request.transaction_id,
            fraud_score=score_result['fraud_score'],
            risk_level=score_result['risk_level'],
            decision=score_result['decision'],
            confidence=score_result['confidence'],
            latency_ms=latency_ms,
            risk_factors=score_result.get('risk_factors', []),
            model_version=scorer.model_version,
            scored_at=datetime.utcnow(),
        )
        
        # Cache result
        await cache.set_score(request.transaction_id, response.model_dump())
        
        # Log to database (async, non-blocking)
        background_tasks.add_task(
            db.log_transaction,
            request.model_dump(),
            response.model_dump()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/score/batch", response_model=BatchResponse, tags=["Scoring"])
async def score_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    """
    Score multiple transactions in batch.
    
    Maximum 100 transactions per batch.
    """
    start_time = time.perf_counter()
    results = []
    
    for txn in request.transactions:
        try:
            score = await score_transaction(txn, background_tasks, scorer, db, cache)
            results.append(score)
        except Exception as e:
            logger.error(f"Batch scoring error for {txn.transaction_id}: {e}")
            results.append(FraudScore(
                transaction_id=txn.transaction_id,
                fraud_score=0.5,
                risk_level="UNKNOWN",
                decision="REVIEW",
                confidence=0.0,
                latency_ms=0,
                risk_factors=["SCORING_ERROR"],
                model_version=scorer.model_version,
                scored_at=datetime.utcnow(),
            ))
    
    total_latency = (time.perf_counter() - start_time) * 1000
    avg_latency = total_latency / len(results) if results else 0
    
    return BatchResponse(
        results=results,
        total_count=len(results),
        avg_latency_ms=avg_latency,
        total_latency_ms=total_latency,
    )


@app.get("/api/v1/transaction/{transaction_id}", tags=["Transactions"])
async def get_transaction(
    transaction_id: str,
    db: Database = Depends(get_database),
):
    """Get transaction details and score history."""
    result = await db.get_transaction(transaction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result


async def _run_investigation(case_id: str, db: Database, cache: RedisCache):
    case = await db.get_case(case_id)
    transaction = await db.get_transaction(case['transaction_id'])
    score = await db.get_score(case['transaction_id'])
    try:
        state = await InvestigationWorkflow(db, cache).run(transaction, score)
        case.update(triage=state['triage'], evidence=state['evidence'], result=state['result'],
                    investigation_mode=state['investigation_mode'], provider_fallback=state['provider_fallback'],
                    review_reasons=state['review_reasons'],
                    status='REVIEW_REQUIRED' if state['result']['requires_human_review'] else 'COMPLETE')
    except Exception as exc:
        logger.exception("Investigation failed for %s", case_id)
        case.update(status='FAILED', error=str(exc))
    await db.save_case(case)


@app.post("/api/v1/investigations/{transaction_id}", response_model=Case, tags=["Investigations"])
async def start_investigation(transaction_id: str, background_tasks: BackgroundTasks,
                              db: Database = Depends(get_database), cache: RedisCache = Depends(get_cache)):
    """Start an investigation for an already scored transaction; scoring stays independent."""
    existing = await db.get_case_for_transaction(transaction_id)
    if existing:
        return Case.model_validate(existing)
    transaction, score = await db.get_transaction(transaction_id), await db.get_score(transaction_id)
    if not transaction or not score:
        raise HTTPException(status_code=404, detail="Scored transaction not found")
    case = Case(case_id=str(uuid.uuid4()), transaction_id=transaction_id, status='PENDING')
    await db.save_case(case.model_dump())
    background_tasks.add_task(_run_investigation, case.case_id, db, cache)
    return case


@app.get("/api/v1/investigations/{case_id}", response_model=Case, tags=["Investigations"])
async def get_investigation(case_id: str, db: Database = Depends(get_database)):
    case = await db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return Case.model_validate(case)


@app.get("/api/v1/review/cases", response_model=list[Case], tags=["Review"])
async def list_review_cases(db: Database = Depends(get_database)):
    return [Case.model_validate(case) for case in await db.list_review_cases()]


@app.post("/api/v1/review/cases/{case_id}", response_model=Case, tags=["Review"])
async def review_case(case_id: str, review: ReviewRequest, db: Database = Depends(get_database)):
    case = await db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if case['status'] != 'REVIEW_REQUIRED':
        raise HTTPException(status_code=409, detail="Case does not require review")
    case.update(status='REVIEWED', final_action=review.final_action, reviewer=review.reviewer,
                review_note=review.note)
    await db.save_case(case)
    return Case.model_validate(case)


@app.get("/api/v1/user/{user_id}/profile", tags=["Users"])
async def get_user_profile(
    user_id: str,
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    """Get user profile and risk summary."""
    # Try cache first
    profile = await cache.get_user_profile(user_id)
    if profile:
        return profile
    
    # Get from database
    profile = await db.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Cache for next time
    await cache.set_user_profile(user_id, profile)
    return profile


@app.get("/api/v1/user/{user_id}/transactions", tags=["Users"])
async def get_user_transactions(
    user_id: str,
    limit: int = 50,
    db: Database = Depends(get_database),
):
    """Get recent transactions for a user."""
    return await db.get_user_transactions(user_id, limit)


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready", tags=["Health"])
async def readiness_check(
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    """Readiness check - verifies all components are ready."""
    checks = {
        "model": scorer.is_ready(),
        "database": await db.health_check(),
        "cache": await cache.health_check(),
    }
    
    all_ready = all(checks.values())
    status = "ready" if all_ready else "not_ready"
    
    return {
        "status": status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check - verifies the service is running."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    """Get platform metrics for monitoring."""
    return {
        "model": {
            "version": scorer.model_version,
            "predictions_total": scorer.metrics.get('predictions_total', 0),
            "avg_latency_ms": scorer.metrics.get('avg_latency_ms', 0),
        },
        "database": await db.get_metrics(),
        "cache": await cache.get_metrics(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# Include routers
# app.include_router(health.router, prefix="/health", tags=["Health"])
# app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["Transactions"])
# app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        http="httptools",
        log_level="info",
    )
