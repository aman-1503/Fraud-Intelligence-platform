"""Postgres, Redis, and a verified model for the API tests."""
import asyncio
import os
from pathlib import Path

os.environ.setdefault("JWT_SECRET", "test-secret-value-not-a-placeholder-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_platform")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("RATE_LIMIT_REQUESTS", "100000")
os.environ.setdefault("MODEL_PATH", "/tmp/fraud-test-model")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "console")


def pytest_configure(config):
    from app.ml.artifacts import publish_demo_model
    publish_demo_model(Path(os.environ["MODEL_PATH"]))


def _dsn() -> str:
    return os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1)


import pytest


@pytest.fixture(autouse=True)
def clean_state():
    import asyncpg
    import redis

    from app.investigation.provider import reset_openai_breaker

    reset_openai_breaker()

    async def wipe():
        conn = await asyncpg.connect(_dsn())
        if await conn.fetchval("SELECT to_regclass('public.transactions')"):
            await conn.execute(
                "TRUNCATE investigation_cases, fraud_scores, transactions RESTART IDENTITY CASCADE"
            )
        await conn.close()

    asyncio.run(wipe())
    client = redis.Redis.from_url(os.environ["REDIS_URL"])
    client.flushdb()
    client.close()
