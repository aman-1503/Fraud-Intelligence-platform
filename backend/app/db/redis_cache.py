"""Redis cache. redis.asyncio is the maintained client; aioredis 2 is abandoned.

Keys expire in Redis. This process does not keep a second copy of the data.
"""
import json
from typing import Any, Dict, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger


class RedisCache:
    def __init__(self):
        self.client = None

    async def connect(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await self.client.ping()
        logger.info("Redis connected")

    async def disconnect(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        logger.info("Redis disconnected")

    async def health_check(self) -> bool:
        if self.client is None:
            return False
        return (await self.client.ping()) is True

    async def get_metrics(self) -> Dict[str, Any]:
        size = await self.client.dbsize() if self.client is not None else 0
        return {"connected": self.client is not None, "cache_size": size}

    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        raw = await self.client.get(f"user_profile:{user_id}")
        return json.loads(raw) if raw else None

    async def set_user_profile(self, user_id: str, profile: Dict):
        await self.client.set(
            f"user_profile:{user_id}",
            json.dumps(profile, default=str),
            ex=settings.USER_PROFILE_TTL_SECONDS,
        )

    async def invalidate_user_profile(self, user_id: str):
        await self.client.delete(f"user_profile:{user_id}")

    async def get_score(self, transaction_id: str, request_hash: str) -> Optional[Dict]:
        raw = await self.client.get(f"score:{transaction_id}:{request_hash}")
        return json.loads(raw) if raw else None

    async def set_score(self, transaction_id: str, request_hash: str, score: Dict):
        await self.client.set(
            f"score:{transaction_id}:{request_hash}",
            json.dumps(score, default=str),
            ex=settings.CACHE_TTL_SECONDS,
        )

    async def get_recent_transactions(self, user_id: str) -> Optional[list]:
        raw = await self.client.lrange(f"recent_txns:{user_id}", 0, -1)
        if not raw:
            return None
        return [json.loads(item) for item in raw]

    async def add_transaction(self, user_id: str, transaction: Dict):
        key = f"recent_txns:{user_id}"
        await self.client.rpush(key, json.dumps(transaction, default=str))
        await self.client.ltrim(key, -100, -1)
        await self.client.expire(key, 3600)

    async def increment_velocity(self, user_id: str, window: str = "1h") -> int:
        key = f"velocity:{user_id}:{window}"
        count = await self.client.incr(key)
        if count == 1:
            await self.client.expire(key, 3600 if window == "1h" else 86400)
        return int(count)

    async def get_velocity(self, user_id: str, window: str = "1h") -> int:
        value = await self.client.get(f"velocity:{user_id}:{window}")
        return int(value or 0)

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        cache_key = f"ratelimit:{key}"
        count = await self.client.incr(cache_key)
        if count == 1:
            await self.client.expire(cache_key, window_seconds)
        return int(count) <= limit
