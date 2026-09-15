"""Narrow, rate-conscious proxy for public Bitcoin mempool transactions."""
import asyncio
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

import httpx
from pydantic import BaseModel, Field

SOURCE_URL = 'https://mempool.space/api/mempool/recent'


class BitcoinMempoolTransaction(BaseModel):
    txid: str = Field(pattern=r'^[0-9a-f]{64}$')
    fee: int = Field(ge=0, description='Fee in satoshis')
    vsize: int = Field(gt=0, description='Virtual size in vbytes')
    value: int = Field(ge=0, description='Value reported by mempool.space, in satoshis')


class BitcoinFeedResult(BaseModel):
    source: str = SOURCE_URL
    network: str = 'bitcoin_mainnet'
    status: str = 'unconfirmed'
    observed_at: datetime
    transactions: list[BitcoinMempoolTransaction]


async def _fetch_public_feed():
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(SOURCE_URL)
        response.raise_for_status()
        return response.json()


class LiveBitcoinFeed:
    def __init__(self, fetcher: Callable[[], Awaitable[object]] = _fetch_public_feed, ttl_seconds: int = 10):
        self.fetcher = fetcher
        self.ttl_seconds = ttl_seconds
        self._cached = None
        self._expires = 0.0
        self._lock = asyncio.Lock()

    async def recent(self) -> BitcoinFeedResult:
        async with self._lock:
            if self._cached and time.monotonic() < self._expires:
                return self._cached
            raw = await self.fetcher()
            if not isinstance(raw, list) or not raw:
                raise ValueError('Live source returned no transactions')
            transactions = [BitcoinMempoolTransaction.model_validate(item) for item in raw[:10]]
            result = BitcoinFeedResult(observed_at=datetime.now(timezone.utc), transactions=transactions)
            self._cached = result
            self._expires = time.monotonic() + self.ttl_seconds
            return result
