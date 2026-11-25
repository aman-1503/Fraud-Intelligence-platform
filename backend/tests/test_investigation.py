"""Focused workflow and API checks using real in-memory repository adapters."""
import pytest
import json
import numpy as np
import xgboost as xgb
import httpx
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from app.db.database import Database
from app.db.redis_cache import RedisCache
from app.investigation.workflow import InvestigationWorkflow
from app.investigation.provider import UnavailableProvider, OpenAIProvider
from app.main import app
from app.ml.scorer import FraudScorer, FeatureExtractor
from app.core.config import settings


@pytest.fixture
async def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'CASE_DB_PATH', str(tmp_path / 'cases.sqlite3'))
    db, cache = Database(), RedisCache()
    await db.connect()
    await cache.connect()
    yield db, cache
    await db.disconnect()
    await cache.disconnect()


def transaction(transaction_id='T-1', user_id='U000001', amount=5000, country='NG', device='new-device'):
    return {'transaction_id': transaction_id, 'user_id': user_id, 'amount': amount,
            'country': country, 'device_id': device, 'merchant_category': 'crypto_exchange',
            'timestamp': datetime.now(timezone.utc)}


def score(probability=0.9, risk='CRITICAL'):
    return {'fraud_score': probability, 'risk_level': risk,
            'risk_factors': ['HIGH_AMOUNT', 'HIGH_RISK_COUNTRY'] if risk in ('HIGH', 'CRITICAL') else []}


@pytest.mark.asyncio
async def test_low_risk_transaction(stores):
    db, cache = stores
    state = await InvestigationWorkflow(db, cache, UnavailableProvider()).run(transaction(amount=20), score(0.1, 'LOW'))
    assert state['triage']['requires_investigation'] is False
    assert state['result']['recommended_action'] == 'APPROVE'
    assert state['result']['evidence_ids'] == ['score:T-1']
    assert state['provider_fallback'] is False


@pytest.mark.asyncio
async def test_clearly_suspicious_transaction(stores):
    db, cache = stores
    old = transaction('OLD-1', amount=100, country='US', device='device_001')
    old['timestamp'] = datetime.now(timezone.utc) - timedelta(minutes=10)
    await db.log_transaction(old, score(0.1, 'LOW'))
    state = await InvestigationWorkflow(db, cache).run(transaction(), score())
    assert state['result']['recommended_action'] == 'BLOCK'
    assert state['result']['requires_human_review'] is True
    assert 'BLOCK_RECOMMENDATION' in state['review_reasons']
    ids = {item['evidence_id'] for item in state['evidence']['items']}
    assert set(state['result']['evidence_ids']).issubset(ids)
    assert 'history:U000001:recent' in ids


@pytest.mark.asyncio
async def test_insufficient_evidence(stores):
    db, cache = stores
    state = await InvestigationWorkflow(db, cache).run(transaction(user_id='UNKNOWN'), score())
    assert state['evidence']['insufficient_evidence'] is True
    assert state['result']['recommended_action'] == 'ESCALATE_FOR_REVIEW'
    assert state['result']['requires_human_review'] is True
    assert 'INSUFFICIENT_EVIDENCE' in state['review_reasons']
    gaps = {gap['signal']: gap for gap in state['evidence']['gaps']}
    assert set(gaps) == {'RECENT_HISTORY', 'AMOUNT_BASELINE', 'DEVICE_BASELINE', 'COUNTRY_BASELINE'}
    assert all(gap['next_check'] for gap in gaps.values())


@pytest.mark.asyncio
async def test_evidence_gap_brief_only_lists_missing_signals(stores):
    db, cache = stores
    await db.log_transaction(transaction('OLD-GAP', user_id='U-NO-PROFILE', amount=50,
                                         country='US', device='old-device'), score(0.1, 'LOW'))
    state = await InvestigationWorkflow(db, cache).run(transaction(user_id='U-NO-PROFILE'), score())
    gaps = {gap['signal'] for gap in state['evidence']['gaps']}
    assert gaps == {'AMOUNT_BASELINE'}
    assert state['evidence']['insufficient_evidence'] is False


@pytest.mark.asyncio
async def test_evidence_history_without_timestamps(stores):
    db, cache = stores
    first = transaction('OLD-NO-TIME-1')
    second = transaction('OLD-NO-TIME-2')
    first.pop('timestamp')
    second.pop('timestamp')
    await db.log_transaction(first, score(0.1, 'LOW'))
    await db.log_transaction(second, score(0.1, 'LOW'))
    state = await InvestigationWorkflow(db, cache).run(transaction(), score())
    assert state['evidence']['insufficient_evidence'] is False
    assert 'history:U000001:recent' in {item['evidence_id'] for item in state['evidence']['items']}


class LowConfidenceProvider:
    async def recommend(self, transaction, score, triage, evidence):
        return {'recommended_action': 'ESCALATE_FOR_REVIEW', 'confidence': 0.2,
                'risk_summary': 'Uncertain', 'evidence_ids': [evidence.items[0].evidence_id],
                'requires_human_review': False}


@pytest.mark.asyncio
async def test_low_confidence_escalation(stores):
    db, cache = stores
    state = await InvestigationWorkflow(db, cache, LowConfidenceProvider()).run(transaction(), score())
    assert state['result']['requires_human_review'] is True
    assert 'LOW_CONFIDENCE' in state['review_reasons']


class MalformedProvider:
    async def recommend(self, transaction, score, triage, evidence):
        return {'recommended_action': 'SEND_MONEY', 'confidence': 3,
                'risk_summary': '', 'evidence_ids': ['fabricated'], 'requires_human_review': False}


@pytest.mark.asyncio
async def test_malformed_agent_output(stores):
    db, cache = stores
    with pytest.raises(Exception):
        await InvestigationWorkflow(db, cache, MalformedProvider()).run(transaction(), score())


@pytest.mark.asyncio
async def test_llm_unavailable_deterministic_fallback(stores):
    db, cache = stores
    old = transaction('OLD-FALLBACK', amount=100, country='US', device='old-device')
    await db.log_transaction(old, score(0.1, 'LOW'))
    state = await InvestigationWorkflow(db, cache, UnavailableProvider()).run(transaction(), score())
    assert state['provider_fallback'] is True
    assert state['result']['recommended_action'] == 'BLOCK'
    assert 'MODEL_FALLBACK' in state['review_reasons']


@pytest.mark.asyncio
async def test_openai_provider_uses_retrieved_evidence_and_structured_output(stores):
    db, cache = stores
    await db.log_transaction(transaction('OLD-LLM', amount=30, country='US', device='old-device'), score(0.1, 'LOW'))
    seen = {}

    def response(request):
        seen['body'] = json.loads(request.content)
        seen['authorization'] = request.headers.get('authorization')
        aliases = [item['evidence_id'] for item in json.loads(seen['body']['input'])['evidence']
                   if item['kind'] in ('country_change', 'device_activity')]
        output = {'recommended_action': 'REQUEST_VERIFICATION', 'confidence': 0.82,
                  'risk_summary': 'New country and device versus stored history; verify customer.',
                  'evidence_ids': aliases,
                  'requires_human_review': False}
        return httpx.Response(200, json={'status': 'completed', 'output': [
            {'content': [{'type': 'output_text', 'text': json.dumps(output)}]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        provider = OpenAIProvider(api_key='test-key', model='gpt-4.1-mini', client=client)
        state = await InvestigationWorkflow(db, cache, provider).run(transaction(), score(0.7, 'HIGH'))
    assert state['investigation_mode'] == 'openai'
    assert state['provider_fallback'] is False
    assert state['result']['recommended_action'] == 'REQUEST_VERIFICATION'
    assert set(state['result']['evidence_ids']).issubset({e['evidence_id'] for e in state['evidence']['items']})
    assert seen['authorization'] == 'Bearer test-key'
    assert seen['body']['store'] is False
    assert seen['body']['text']['format']['type'] == 'json_schema'
    assert 'U000001' not in seen['body']['input']


@pytest.mark.asyncio
async def test_openai_provider_rejects_fabricated_evidence(stores):
    db, cache = stores
    await db.log_transaction(transaction('OLD-BAD'), score(0.1, 'LOW'))
    def response(request):
        output = {'recommended_action': 'BLOCK', 'confidence': 0.95,
                  'risk_summary': 'Fabricated claim', 'evidence_ids': ['made-up:device'],
                  'requires_human_review': False}
        return httpx.Response(200, json={'status': 'completed', 'output': [
            {'content': [{'type': 'output_text', 'text': json.dumps(output)}]}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ValueError, match='not retrieved'):
            await InvestigationWorkflow(db, cache, OpenAIProvider(api_key='test-key', client=client)).run(transaction(), score())


@pytest.mark.asyncio
async def test_openai_unavailable_uses_deterministic_fallback(stores):
    db, cache = stores
    await db.log_transaction(transaction('OLD-OFFLINE'), score(0.1, 'LOW'))
    def response(request):
        return httpx.Response(503, json={'error': {'message': 'unavailable'}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        state = await InvestigationWorkflow(db, cache, OpenAIProvider(api_key='test-key', client=client)).run(transaction(), score())
    assert state['investigation_mode'] == 'deterministic'
    assert state['provider_fallback'] is True
    assert state['result']['requires_human_review'] is True


def test_human_override_and_saved_status():
    from app.core.security import create_access_token
    headers = {"Authorization": f"Bearer {create_access_token('test-reviewer', 'admin')}"}
    with TestClient(app) as client:
        txn = transaction('API-CASE-1')
        txn['timestamp'] = txn['timestamp'].isoformat()
        scored = client.post('/api/v1/score', json=txn, headers=headers)
        assert scored.status_code == 200, scored.text
        started = client.post('/api/v1/investigations/API-CASE-1', headers=headers)
        assert started.status_code == 200, started.text
        case_id = started.json()['case_id']
        case = client.get(f'/api/v1/investigations/{case_id}', headers=headers).json()
        assert case['status'] == 'REVIEW_REQUIRED', case
        assert case['result']['evidence_ids']
        assert case['review_reasons']
        listed = client.get('/api/v1/review/cases', headers=headers).json()
        assert any(item['case_id'] == case_id for item in listed)
        reviewed = client.post(f'/api/v1/review/cases/{case_id}', json={
            'final_action': 'REQUEST_VERIFICATION',
            'note': 'Customer verification required before blocking'}, headers=headers)
        assert reviewed.status_code == 200, reviewed.text
        saved = client.get(f'/api/v1/investigations/{case_id}', headers=headers).json()
        assert saved['status'] == 'REVIEWED'
        assert saved['final_action'] == 'REQUEST_VERIFICATION'
        assert saved['reviewer'] == 'test-reviewer'

    import asyncio
    async def reopened():
        db = Database()
        await db.connect()
        result = await db.get_case(case_id)
        await db.disconnect()
        return result
    assert asyncio.run(reopened())['final_action'] == 'REQUEST_VERIFICATION'


def test_provider_status_does_not_expose_credential(monkeypatch):
    monkeypatch.setattr(settings, 'INVESTIGATION_PROVIDER', 'openai')
    monkeypatch.setattr(settings, 'OPENAI_API_KEY', 'test-secret-value')
    with TestClient(app) as client:
        response = client.get('/api/v1/investigations/provider-status')
    assert response.status_code == 200
    assert response.json()['llm_configured'] is True
    assert response.json()['model'] == settings.OPENAI_MODEL
    assert 'test-secret-value' not in response.text


class CitingOneItem:
    async def recommend(self, transaction, score, triage, evidence):
        return {'recommended_action': 'APPROVE', 'confidence': 0.91,
                'risk_summary': 'Only the score was cited.',
                'evidence_ids': [evidence.items[0].evidence_id], 'requires_human_review': False}


@pytest.mark.asyncio
async def test_uncited_anomalous_evidence_forces_human_review(stores):
    db, cache = stores
    state = await InvestigationWorkflow(db, cache, CitingOneItem()).run(transaction(), score(0.9, 'CRITICAL'))
    anomalous = [item['evidence_id'] for item in state['evidence']['items'] if item['anomalous']]
    assert len(anomalous) >= 3
    assert state['result']['evidence_ids'] == ['score:T-1']
    assert state['result']['requires_human_review'] is True
    assert 'UNCITED_ANOMALOUS_EVIDENCE' in state['review_reasons']


class ApproveEverything:
    async def recommend(self, transaction, score, triage, evidence):
        return {'recommended_action': 'APPROVE', 'confidence': 0.91,
                'risk_summary': 'Approve despite the rules action.',
                'evidence_ids': [item.evidence_id for item in evidence.items],
                'requires_human_review': False}


@pytest.mark.asyncio
async def test_escalate_then_approve_is_disagreement(stores):
    db, cache = stores
    state = await InvestigationWorkflow(db, cache, ApproveEverything()).run(transaction(), score(0.7, 'HIGH'))
    assert state['triage']['rules_action'] == 'ESCALATE_FOR_REVIEW'
    assert state['result']['recommended_action'] == 'APPROVE'
    assert 'RULE_DISAGREEMENT' in state['review_reasons']
    assert state['result']['requires_human_review'] is True


@pytest.mark.asyncio
async def test_investigation_does_not_invent_evidence(stores):
    db, cache = stores
    workflow = InvestigationWorkflow(db, cache, UnavailableProvider())
    with pytest.raises(ValueError, match='Evidence was not retrieved'):
        await workflow.investigation({
            'transaction': transaction(),
            'score': score(),
            'triage': {'requires_investigation': True, 'rules_action': 'ESCALATE_FOR_REVIEW', 'reason': 'test'},
        })


@pytest.mark.asyncio
async def test_exported_xgboost_artifact_used_for_immediate_score(tmp_path, monkeypatch):
    from app.ml.artifacts import publish
    names = FeatureExtractor().feature_names
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, len(names)))
    y = np.array([0, 1] * 12)
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1)
    model.fit(X, y)
    publish(tmp_path, model, None, {'feature_names': names, 'version': 'test-artifact'})
    monkeypatch.setenv('MODEL_PATH', str(tmp_path))
    scorer = await FraudScorer().initialize()
    result = await scorer.score(transaction())
    assert scorer.model is not None
    assert scorer.model_version == 'test-artifact'
    assert 0 <= result['fraud_score'] <= 1
    assert result['confidence'] == pytest.approx(max(result['fraud_score'], 1 - result['fraud_score']))
