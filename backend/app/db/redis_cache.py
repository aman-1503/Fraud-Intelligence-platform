"""
Redis Cache Module
==================
High-performance caching for user profiles and transaction scores.
"""

import json
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.config import settings
from app.core.logging import logger


class RedisCache:
    """
    Redis cache for low-latency data access.
    Caches user profiles and recent scores.
    """
    
    def __init__(self):
        """Initialize Redis connection settings."""
        self.client = None
        self._connected = False
        
        # In-memory cache for demo
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl: Dict[str, float] = {}
    
    async def connect(self):
        """Establish Redis connection."""
        try:
            # For demo, we use in-memory caching
            # In production, use aioredis:
            # import aioredis
            # self.client = await aioredis.from_url(
            #     settings.REDIS_URL,
            #     encoding="utf-8",
            #     decode_responses=True,
            # )
            
            self._connected = True
            logger.info("Redis connected (in-memory mode)")
            
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
        self._connected = False
        logger.info("Redis disconnected")
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        return self._connected
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        return {
            "connected": self._connected,
            "cache_size": len(self._cache),
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cached item is expired."""
        if key not in self._cache_ttl:
            return True
        return datetime.utcnow().timestamp() > self._cache_ttl[key]
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if key in self._cache and not self._is_expired(key):
            return json.dumps(self._cache[key])
        return None
    
    async def set(
        self, 
        key: str, 
        value: str, 
        ttl_seconds: int = None
    ):
        """Set value in cache with TTL."""
        ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        self._cache[key] = json.loads(value) if isinstance(value, str) else value
        self._cache_ttl[key] = datetime.utcnow().timestamp() + ttl
    
    async def delete(self, key: str):
        """Delete value from cache."""
        self._cache.pop(key, None)
        self._cache_ttl.pop(key, None)
    
    # User profile caching
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get cached user profile."""
        key = f"user_profile:{user_id}"
        data = await self.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set_user_profile(self, user_id: str, profile: Dict):
        """Cache user profile."""
        key = f"user_profile:{user_id}"
        await self.set(
            key, 
            json.dumps(profile), 
            ttl_seconds=settings.USER_PROFILE_TTL_SECONDS
        )
    
    async def invalidate_user_profile(self, user_id: str):
        """Invalidate cached user profile."""
        key = f"user_profile:{user_id}"
        await self.delete(key)
    
    # Score caching
    async def get_score(self, transaction_id: str) -> Optional[Dict]:
        """Get cached transaction score."""
        key = f"score:{transaction_id}"
        data = await self.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set_score(self, transaction_id: str, score: Dict):
        """Cache transaction score."""
        key = f"score:{transaction_id}"
        await self.set(
            key, 
            json.dumps(score, default=str), 
            ttl_seconds=settings.CACHE_TTL_SECONDS
        )
    
    # Recent transactions caching (for velocity)
    async def get_recent_transactions(
        self, 
        user_id: str
    ) -> Optional[list]:
        """Get cached recent transactions for user."""
        key = f"recent_txns:{user_id}"
        data = await self.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def add_transaction(self, user_id: str, transaction: Dict):
        """Add transaction to user's recent history."""
        key = f"recent_txns:{user_id}"
        
        # Get existing transactions
        recent = await self.get_recent_transactions(user_id) or []
        
        # Add new transaction
        recent.append(transaction)
        
        # Keep only last 100
        if len(recent) > 100:
            recent = recent[-100:]
        
        # Cache with longer TTL
        await self.set(key, json.dumps(recent, default=str), ttl_seconds=3600)
    
    # Velocity counters (for rate limiting and velocity checks)
    async def increment_velocity(
        self, 
        user_id: str, 
        window: str = "1h"
    ) -> int:
        """Increment velocity counter for user."""
        key = f"velocity:{user_id}:{window}"
        
        if key not in self._cache or self._is_expired(key):
            self._cache[key] = 0
            ttl = 3600 if window == "1h" else 86400  # 1 hour or 24 hours
            self._cache_ttl[key] = datetime.utcnow().timestamp() + ttl
        
        self._cache[key] += 1
        return self._cache[key]
    
    async def get_velocity(self, user_id: str, window: str = "1h") -> int:
        """Get velocity count for user."""
        key = f"velocity:{user_id}:{window}"
        if key in self._cache and not self._is_expired(key):
            return self._cache[key]
        return 0
    
    # Rate limiting
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int
    ) -> bool:
        """
        Check if request is within rate limit.
        Returns True if allowed, False if rate limited.
        """
        cache_key = f"ratelimit:{key}"
        
        if cache_key not in self._cache or self._is_expired(cache_key):
            self._cache[cache_key] = 1
            self._cache_ttl[cache_key] = datetime.utcnow().timestamp() + window_seconds
            return True
        
        if self._cache[cache_key] >= limit:
            return False
        
        self._cache[cache_key] += 1
        return True
