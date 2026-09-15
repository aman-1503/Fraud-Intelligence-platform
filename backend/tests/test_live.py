"""Live feed validation and API tests use mocked upstream data, never invented fallback data."""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.live.bitcoin import LiveBitcoinFeed
from app.main import app, get_live_bitcoin_feed


def row(txid='a' * 64):
    return {'txid': txid, 'fee': 423, 'vsize': 140, 'value': 141528451}


@pytest.mark.asyncio
async def test_live_feed_validates_and_caches():
    calls = 0
    async def fetch():
        nonlocal calls
        calls += 1
        return [row()]
    feed = LiveBitcoinFeed(fetcher=fetch, ttl_seconds=10)
    first = await feed.recent()
    second = await feed.recent()
    assert calls == 1
    assert first.observed_at == second.observed_at
    assert first.transactions[0].txid == 'a' * 64
    assert first.status == 'unconfirmed'


@pytest.mark.asyncio
async def test_live_feed_rejects_malformed_upstream():
    async def fetch():
        return [row('not-a-real-txid')]
    with pytest.raises(ValueError):
        await LiveBitcoinFeed(fetcher=fetch).recent()


def test_live_endpoint_and_upstream_failure():
    async def fetch():
        return [row()]
    app.dependency_overrides[get_live_bitcoin_feed] = lambda: LiveBitcoinFeed(fetcher=fetch)
    try:
        with TestClient(app) as client:
            response = client.get('/api/v1/live/bitcoin/recent')
        assert response.status_code == 200
        assert response.json()['transactions'][0]['value'] == 141528451
    finally:
        app.dependency_overrides.clear()

    async def unavailable():
        raise httpx.ConnectError('upstream unavailable')
    app.dependency_overrides[get_live_bitcoin_feed] = lambda: LiveBitcoinFeed(fetcher=unavailable)
    try:
        with TestClient(app) as client:
            response = client.get('/api/v1/live/bitcoin/recent')
        assert response.status_code == 502
        assert response.json()['detail'] == 'Public Bitcoin feed unavailable'
    finally:
        app.dependency_overrides.clear()
