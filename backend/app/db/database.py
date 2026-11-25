"""Postgres access through an asyncpg pool.

Transactions, scores, profiles, and investigation cases all live here.
There is no in-memory fallback: connect() raises if Postgres is unreachable.
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from app.core.config import settings
from app.core.logging import logger


class IdempotencyConflict(Exception):
    """A transaction_id was reused with a different scoring payload."""


def _dsn() -> str:
    return settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)


def _amount(value) -> float:
    return float(value)


def _iso(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        await asyncio.to_thread(_upgrade)
        self.pool = await asyncpg.create_pool(
            _dsn(),
            min_size=1,
            max_size=settings.DATABASE_POOL_SIZE,
            command_timeout=10,
        )
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT 1")
        logger.info("Database connected")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
        logger.info("Database disconnected")

    async def health_check(self) -> bool:
        if self.pool is None:
            return False
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT 1") == 1

    async def get_metrics(self) -> Dict[str, Any]:
        async with self.pool.acquire() as conn:
            transactions = await conn.fetchval("SELECT COUNT(*) FROM transactions")
            users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE home_country IS NOT NULL")
        return {"connected": self.pool is not None, "transactions_count": transactions, "users_count": users}

    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id, home_country, account_age_days, avg_txn_amount, std_txn_amount, "
                "avg_monthly_txns, primary_device_id, merchant_diversity, risk_score "
                "FROM users WHERE user_id=$1 AND home_country IS NOT NULL",
                user_id,
            )
        if not row:
            return None
        profile = dict(row)
        profile["avg_txn_amount"] = _amount(profile["avg_txn_amount"])
        profile["std_txn_amount"] = _amount(profile["std_txn_amount"])
        profile["merchant_diversity"] = _amount(profile["merchant_diversity"] or 0)
        profile["risk_score"] = _amount(profile["risk_score"] or 0)
        return profile

    async def ensure_user(self, user_id: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users(user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                user_id,
            )

    async def get_transaction(self, transaction_id: str) -> Optional[Dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT t.payload AS transaction, s.payload AS score "
                "FROM transactions t LEFT JOIN fraud_scores s USING (transaction_id) "
                "WHERE t.transaction_id=$1",
                transaction_id,
            )
        if not row:
            return None
        transaction = json.loads(row["transaction"])
        if row["score"]:
            transaction.update(json.loads(row["score"]))
        return transaction

    async def get_score(self, transaction_id: str) -> Optional[Dict]:
        record = await self.get_score_record(transaction_id)
        return record["score"] if record else None

    async def get_score_record(self, transaction_id: str) -> Optional[Dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT request_hash, payload FROM fraud_scores WHERE transaction_id=$1",
                transaction_id,
            )
        if not row:
            return None
        return {"request_hash": row["request_hash"], "score": json.loads(row["payload"])}

    async def get_user_transactions(self, user_id: str, limit: int = 50) -> List[Dict]:
        limit = max(1, min(int(limit), 100))
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT transaction_id, user_id, amount, merchant_category, country, device_id, timestamp "
                "FROM transactions WHERE user_id=$1 ORDER BY timestamp DESC LIMIT $2",
                user_id, limit,
            )
        return [
            {
                "transaction_id": row["transaction_id"],
                "user_id": row["user_id"],
                "amount": _amount(row["amount"]),
                "merchant_category": row["merchant_category"],
                "country": row["country"],
                "device_id": row["device_id"],
                "timestamp": _iso(row["timestamp"]),
            }
            for row in rows
        ]

    async def log_transaction(self, transaction: Dict, score_result: Dict, request_hash: str = "0" * 64):
        """Commit the transaction and its score before the caller returns a decision."""
        transaction_id = transaction["transaction_id"]
        timestamp = transaction.get("timestamp") or datetime.now(timezone.utc)
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO users(user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                    transaction["user_id"],
                )
                await conn.execute(
                    "INSERT INTO transactions(transaction_id, user_id, amount, currency, merchant_category, "
                    "country, device_id, timestamp, payload) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb)",
                    transaction_id, transaction["user_id"], transaction["amount"],
                    transaction.get("currency") or "USD", transaction["merchant_category"],
                    transaction.get("country") or "US", transaction.get("device_id"), timestamp,
                    json.dumps(transaction, default=str),
                )
                await conn.execute(
                    "INSERT INTO fraud_scores(transaction_id, request_hash, fraud_score, risk_level, decision, "
                    "confidence, model_version, latency_ms, payload) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb)",
                    transaction_id, request_hash, score_result["fraud_score"], score_result["risk_level"],
                    score_result.get("decision") or "REVIEW", score_result.get("confidence") or 0,
                    score_result.get("model_version") or "unspecified",
                    score_result.get("latency_ms") or 0, json.dumps(score_result, default=str),
                )

    async def save_case(self, case: Dict):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO investigation_cases(case_id, transaction_id, status, payload) "
                "VALUES ($1,$2,$3,$4::jsonb) ON CONFLICT (case_id) DO UPDATE "
                "SET status=EXCLUDED.status, payload=EXCLUDED.payload",
                case["case_id"], case["transaction_id"], case["status"], json.dumps(case, default=str),
            )

    async def get_case(self, case_id: str) -> Optional[Dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT payload FROM investigation_cases WHERE case_id=$1", case_id)
        return json.loads(row["payload"]) if row else None

    async def get_case_for_transaction(self, transaction_id: str) -> Optional[Dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM investigation_cases WHERE transaction_id=$1", transaction_id
            )
        return json.loads(row["payload"]) if row else None

    async def list_review_cases(self) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT payload FROM investigation_cases WHERE status='REVIEW_REQUIRED' ORDER BY case_id"
            )
        return [json.loads(row["payload"]) for row in rows]


def _upgrade():
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    backend = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    command.upgrade(cfg, "head")
