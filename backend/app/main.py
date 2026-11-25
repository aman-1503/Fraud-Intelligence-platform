"""
Fraud Intelligence Platform - FastAPI Backend
==============================================
Real-time fraud scoring API. A scored decision is committed to Postgres
before the response is returned. Throughput numbers belong in a measured
benchmark, not in this docstring.
"""

import math
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, ValidationError

from app.archive.repository import ArchiveRepository
from app.core.config import settings
from app.core.idempotency import request_hash
from app.core.logging import logger, request_id_var, setup_logging, tx_logger
from app.core.security import assert_secret_configured, require_roles
from app.db.database import Database
from app.db.redis_cache import RedisCache
from app.investigation.models import Action, Case
from app.investigation.workflow import InvestigationWorkflow
from app.live.bitcoin import BitcoinFeedResult, BitcoinTransactionDetail, InvalidBitcoinTransactionId, LiveBitcoinFeed
from app.ml.scorer import FraudScorer, ModelNotReady

fraud_scorer: Optional[FraudScorer] = None
database: Optional[Database] = None
redis_cache: Optional[RedisCache] = None
live_bitcoin_feed = LiveBitcoinFeed()

HTTP_REQUESTS = Counter("fraud_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("fraud_http_request_duration_seconds", "HTTP latency", ["method", "path"])
PREDICTIONS = Counter("fraud_score_predictions_total", "Scored transactions", ["decision"])
MODEL_READY = Gauge("fraud_model_ready", "1 when a verified model is loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fraud_scorer, database, redis_cache
    setup_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    assert_secret_configured()
    logger.info("Starting Fraud Intelligence Platform")
    database = Database()
    await database.connect()
    redis_cache = RedisCache()
    await redis_cache.connect()
    fraud_scorer = FraudScorer()
    try:
        await fraud_scorer.initialize()
    except Exception:
        logger.exception("Model artifact failed verification")
        raise
    if fraud_scorer.is_ready():
        await fraud_scorer.warmup()
    MODEL_READY.set(1 if fraud_scorer.is_ready() else 0)
    logger.info("Model version: %s", fraud_scorer.model_version)
    yield
    if database:
        await database.disconnect()
    if redis_cache:
        await redis_cache.disconnect()


app = FastAPI(
    title="Fraud Intelligence Platform",
    description="Real-time fraud detection API with an XGBoost scoring pipeline",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    path = route.path if route is not None else "unmatched"
    HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    HTTP_LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"
    response.headers["X-Request-ID"] = request_id
    return response


def get_scorer() -> FraudScorer:
    if fraud_scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized")
    return fraud_scorer


def get_database() -> Database:
    if database is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return database


def get_cache() -> RedisCache:
    if redis_cache is None:
        raise HTTPException(status_code=503, detail="Cache not connected")
    return redis_cache


def guarded(*roles: str):
    async def dependency(identity: dict = Depends(require_roles(*roles)), cache: RedisCache = Depends(get_cache)):
        allowed = await cache.check_rate_limit(
            identity["sub"], settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS
        )
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return identity

    return dependency


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=50)
    amount: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    merchant_category: str = Field(..., min_length=1, max_length=50)
    country: str = Field(default="US", min_length=2, max_length=3)
    device_id: Optional[str] = Field(None, max_length=64)
    timestamp: Optional[datetime] = None
    ip_address: Optional[str] = Field(None, max_length=64)
    user_agent: Optional[str] = Field(None, max_length=256)
    session_id: Optional[str] = Field(None, max_length=64)


class FraudScore(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1)
    risk_level: str
    decision: str
    confidence: float = Field(..., ge=0, le=1)
    latency_ms: float
    risk_factors: list[str] = Field(default_factory=list)
    model_version: str
    scored_at: datetime


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., max_length=100)


class BatchItem(BaseModel):
    transaction_id: str
    fraud_score: Optional[float] = None
    risk_level: Optional[str] = None
    decision: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    risk_factors: list[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    scored_at: Optional[datetime] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    results: list[BatchItem]
    total_count: int
    avg_latency_ms: float
    total_latency_ms: float


class ReviewRequest(BaseModel):
    final_action: Action
    note: str = Field(min_length=1, max_length=2000)


class InvestigationProviderStatus(BaseModel):
    configured_provider: str
    llm_configured: bool
    model: str | None = None


def _error_code(exc: HTTPException) -> str:
    if exc.status_code == 409:
        return "conflict"
    if exc.status_code == 503:
        return "unavailable"
    if exc.status_code == 429:
        return "rate_limited"
    return "scoring_failed"


async def _score_one(request: TransactionRequest, scorer: FraudScorer, db: Database, cache: RedisCache) -> FraudScore:
    started = time.perf_counter()
    payload = request.model_dump()
    digest = request_hash(payload)
    try:
        existing = await db.get_score_record(request.transaction_id)
        if existing:
            if existing["request_hash"] != digest:
                raise HTTPException(status_code=409, detail="Transaction id already scored with a different payload")
            return FraudScore(**existing["score"])
        cached = await cache.get_score(request.transaction_id, digest)
        if cached:
            return FraudScore(**cached)
        if not scorer.is_ready():
            raise HTTPException(status_code=503, detail="Scoring model is not loaded")
        history = [
            row for row in await db.get_user_transactions(request.user_id, 100)
            if row["transaction_id"] != request.transaction_id
        ]
        profile = await db.get_user_profile(request.user_id)
        if profile:
            await cache.set_user_profile(request.user_id, profile)
        scoring_profile = None
        if profile or history:
            scoring_profile = dict(profile or {})
            scoring_profile["recent_transactions"] = history
        await cache.increment_velocity(request.user_id, "1h")
        score_result = await scorer.score(payload, scoring_profile)
        latency_ms = (time.perf_counter() - started) * 1000
        response = FraudScore(
            transaction_id=request.transaction_id,
            fraud_score=score_result["fraud_score"],
            risk_level=score_result["risk_level"],
            decision=score_result["decision"],
            confidence=score_result["confidence"],
            latency_ms=latency_ms,
            risk_factors=score_result.get("risk_factors", []),
            model_version=scorer.model_version,
            scored_at=datetime.now(timezone.utc),
        )
        stored = response.model_dump(mode="json")
        await db.log_transaction(payload, stored, digest)
        await cache.set_score(request.transaction_id, digest, stored)
        await cache.add_transaction(request.user_id, {
            "transaction_id": request.transaction_id,
            "amount": request.amount,
            "country": request.country,
            "device_id": request.device_id,
            "merchant_category": request.merchant_category,
            "timestamp": stored["scored_at"],
        })
        PREDICTIONS.labels(response.decision).inc()
        tx_logger.log_score(request.transaction_id, request.user_id, response.fraud_score, response.decision, latency_ms)
        return response
    except HTTPException:
        raise
    except ModelNotReady:
        raise HTTPException(status_code=503, detail="Scoring model is not loaded")
    except asyncpg.UniqueViolationError:
        existing = await db.get_score_record(request.transaction_id)
        if existing and existing["request_hash"] == digest:
            return FraudScore(**existing["score"])
        raise HTTPException(status_code=409, detail="Transaction id already scored with a different payload")
    except Exception:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail="Scoring failed")


@app.post("/api/v1/score", response_model=FraudScore, tags=["Scoring"])
async def score_transaction(
    request: TransactionRequest,
    identity: dict = Depends(guarded("scorer")),
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    return await _score_one(request, scorer, db, cache)


@app.post("/api/v1/score/batch", response_model=BatchResponse, tags=["Scoring"])
async def score_batch(
    request: BatchRequest,
    identity: dict = Depends(guarded("scorer")),
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    started = time.perf_counter()
    results = []
    latencies = []
    for txn in request.transactions:
        try:
            scored = await _score_one(txn, scorer, db, cache)
            results.append(BatchItem(**scored.model_dump()))
            latencies.append(scored.latency_ms)
        except HTTPException as exc:
            results.append(BatchItem(transaction_id=txn.transaction_id, error=_error_code(exc)))
    total_latency = (time.perf_counter() - started) * 1000
    return BatchResponse(
        results=results,
        total_count=len(results),
        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0,
        total_latency_ms=total_latency,
    )


@app.get("/api/v1/transaction/{transaction_id}", tags=["Transactions"])
async def get_transaction(
    transaction_id: str,
    identity: dict = Depends(guarded("scorer", "reviewer")),
    db: Database = Depends(get_database),
):
    result = await db.get_transaction(transaction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result


async def _run_investigation(case_id: str, db: Database, cache: RedisCache, request_id: str):
    token = request_id_var.set(request_id)
    try:
        case = await db.get_case(case_id)
        transaction = await db.get_transaction(case["transaction_id"])
        score = await db.get_score(case["transaction_id"])
        try:
            state = await InvestigationWorkflow(db, cache).run(transaction, score)
            case.update(
                triage=state["triage"], evidence=state["evidence"], result=state["result"],
                investigation_mode=state["investigation_mode"], provider_fallback=state["provider_fallback"],
                review_reasons=state["review_reasons"],
                status="REVIEW_REQUIRED" if state["result"]["requires_human_review"] else "COMPLETE",
            )
        except ValueError:
            logger.exception("Investigation rejected case %s", case_id)
            case.update(status="FAILED", error="citation_rejected")
        except Exception:
            logger.exception("Investigation failed for %s", case_id)
            case.update(status="FAILED", error="investigation_failed")
        await db.save_case(case)
    finally:
        request_id_var.reset(token)


@app.post("/api/v1/investigations/{transaction_id}", response_model=Case, tags=["Investigations"])
async def start_investigation(
    transaction_id: str,
    request: Request,
    identity: dict = Depends(guarded("reviewer")),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    existing = await db.get_case_for_transaction(transaction_id)
    if existing:
        return Case.model_validate(existing)
    transaction, score = await db.get_transaction(transaction_id), await db.get_score(transaction_id)
    if not transaction or not score:
        raise HTTPException(status_code=404, detail="Scored transaction not found")
    case = Case(case_id=str(uuid.uuid4()), transaction_id=transaction_id, status="PENDING")
    await db.save_case(case.model_dump())
    # Run before returning so a caller that reads the case cannot observe a missing write.
    await _run_investigation(case.case_id, db, cache, request.headers.get("x-request-id") or "-")
    return Case.model_validate(await db.get_case(case.case_id))


@app.get("/api/v1/investigations/provider-status", response_model=InvestigationProviderStatus, tags=["Investigations"])
async def investigation_provider_status():
    return InvestigationProviderStatus(
        configured_provider=settings.INVESTIGATION_PROVIDER,
        llm_configured=settings.INVESTIGATION_PROVIDER == "openai" and bool(settings.OPENAI_API_KEY),
        model=settings.OPENAI_MODEL if settings.INVESTIGATION_PROVIDER == "openai" else None,
    )


@app.get("/api/v1/investigations/{case_id}", response_model=Case, tags=["Investigations"])
async def get_investigation(
    case_id: str,
    identity: dict = Depends(guarded("reviewer")),
    db: Database = Depends(get_database),
):
    case = await db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return Case.model_validate(case)


@app.get("/api/v1/review/cases", response_model=list[Case], tags=["Review"])
async def list_review_cases(
    identity: dict = Depends(guarded("reviewer")),
    db: Database = Depends(get_database),
):
    return [Case.model_validate(case) for case in await db.list_review_cases()]


@app.post("/api/v1/review/cases/{case_id}", response_model=Case, tags=["Review"])
async def review_case(
    case_id: str,
    review: ReviewRequest,
    identity: dict = Depends(guarded("reviewer")),
    db: Database = Depends(get_database),
):
    case = await db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if case["status"] != "REVIEW_REQUIRED":
        raise HTTPException(status_code=409, detail="Case does not require review")
    case.update(status="REVIEWED", final_action=review.final_action, reviewer=identity["sub"], review_note=review.note)
    await db.save_case(case)
    return Case.model_validate(case)


@app.get("/api/v1/user/{user_id}/profile", tags=["Users"])
async def get_user_profile(
    user_id: str,
    identity: dict = Depends(guarded("scorer", "reviewer")),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    profile = await cache.get_user_profile(user_id)
    if profile:
        return profile
    profile = await db.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    await cache.set_user_profile(user_id, profile)
    return profile


@app.get("/api/v1/user/{user_id}/transactions", tags=["Users"])
async def get_user_transactions(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    identity: dict = Depends(guarded("scorer", "reviewer")),
    db: Database = Depends(get_database),
):
    return await db.get_user_transactions(user_id, limit)


def get_archive() -> ArchiveRepository:
    archive = ArchiveRepository(settings.ARCHIVE_PATH)
    if not archive.available():
        raise HTTPException(status_code=503, detail="Real-world archive unavailable; run ml/train_real.py")
    return archive


@app.get("/api/v1/archive/metadata", tags=["Real-world archive"])
async def archive_metadata(archive: ArchiveRepository = Depends(get_archive)):
    return archive.metadata()


@app.get("/api/v1/archive/rows", tags=["Real-world archive"])
async def archive_rows(after: int = -1, limit: int = 20, known_fraud_only: bool = False,
                       archive: ArchiveRepository = Depends(get_archive)):
    try:
        return archive.rows(after, limit, known_fraud_only)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def get_live_bitcoin_feed() -> LiveBitcoinFeed:
    return live_bitcoin_feed


@app.get("/api/v1/live/bitcoin/recent", response_model=BitcoinFeedResult, tags=["Live public data"])
async def recent_bitcoin_transactions(feed: LiveBitcoinFeed = Depends(get_live_bitcoin_feed)):
    try:
        return await feed.recent()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Public Bitcoin source unavailable: %s", exc)
        raise HTTPException(status_code=502, detail="Public Bitcoin feed unavailable")


@app.get("/api/v1/live/bitcoin/tx/{txid}", response_model=BitcoinTransactionDetail, tags=["Live public data"])
async def bitcoin_transaction_detail(txid: str, feed: LiveBitcoinFeed = Depends(get_live_bitcoin_feed)):
    try:
        return await feed.detail(txid)
    except InvalidBitcoinTransactionId:
        raise HTTPException(status_code=422, detail="Invalid Bitcoin transaction ID")
    except (httpx.HTTPError, ValidationError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Public Bitcoin detail unavailable: %s", exc)
        raise HTTPException(status_code=502, detail="Public Bitcoin transaction detail unavailable")


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/ready", tags=["Health"])
async def readiness_check(
    scorer: FraudScorer = Depends(get_scorer),
    db: Database = Depends(get_database),
    cache: RedisCache = Depends(get_cache),
):
    async def probe(awaitable):
        try:
            return bool(await awaitable)
        except Exception:
            return False

    checks = {
        "model": scorer.is_ready(),
        "database": await probe(db.health_check()),
        "cache": await probe(cache.health_check()),
    }
    MODEL_READY.set(1 if checks["model"] else 0)
    body = {"status": "ready" if all(checks.values()) else "not_ready", "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat()}
    return JSONResponse(status_code=200 if all(checks.values()) else 503, content=body)


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    setup_logging()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, loop="uvloop", http="httptools", log_level="info")
