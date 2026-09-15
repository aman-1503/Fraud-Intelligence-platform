"""Narrow, rate-conscious proxy for public Bitcoin mempool transactions."""
import asyncio
import re
import time
from datetime import datetime, timezone
from math import ceil
from typing import Awaitable, Callable

import httpx
from pydantic import BaseModel, Field

SOURCE_URL = 'https://mempool.space/api/mempool/recent'
DETAIL_URL = 'https://mempool.space/api/tx'


class InvalidBitcoinTransactionId(ValueError):
    pass


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


class BitcoinTransactionDetail(BitcoinMempoolTransaction):
    confirmed: bool


async def _fetch_public_feed():
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(SOURCE_URL)
        response.raise_for_status()
        return response.json()


async def _fetch_public_detail(txid: str):
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(f'{DETAIL_URL}/{txid}')
        response.raise_for_status()
        return response.json()


class LiveBitcoinFeed:
    def __init__(self, fetcher: Callable[[], Awaitable[object]] = _fetch_public_feed, ttl_seconds: int = 10,
                 detail_fetcher: Callable[[str], Awaitable[object]] = _fetch_public_detail):
        self.fetcher = fetcher
        self.detail_fetcher = detail_fetcher
        self.ttl_seconds = ttl_seconds
        self._cached = None
        self._expires = 0.0
        self._lock = asyncio.Lock()
        self._details = {}

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

    async def detail(self, txid: str) -> BitcoinTransactionDetail:
        # Validate before calling the public endpoint; the browser requests details only on demand.
        if not re.fullmatch(r'[0-9a-f]{64}', txid):
            raise InvalidBitcoinTransactionId('Invalid Bitcoin transaction ID')
        async with self._lock:
            cached = self._details.get(txid)
            if cached and time.monotonic() < cached[1]:
                return cached[0]
            raw = await self.detail_fetcher(txid)
            if not isinstance(raw, dict) or raw.get('txid') != txid:
                raise ValueError('Public source returned a different transaction')
            vout = raw.get('vout')
            if not isinstance(vout, list) or not vout:
                raise ValueError('Public transaction outputs unavailable')
            value = sum(int(output['value']) for output in vout)
            status = raw.get('status') or {}
            if not isinstance(status, dict):
                raise ValueError('Public transaction status unavailable')
            result = BitcoinTransactionDetail.model_validate({
                'txid': txid, 'fee': raw.get('fee'), 'vsize': ceil(raw['weight'] / 4),
                'value': value, 'confirmed': status.get('confirmed'),
            })
            self._details[txid] = (result, time.monotonic() + 30)
            return result
