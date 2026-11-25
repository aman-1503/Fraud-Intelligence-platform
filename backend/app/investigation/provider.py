"""Small provider interface. Mock mode needs no credentials or remote model."""
import asyncio
import json
import random
import time
from typing import Protocol

import httpx

from app.core.config import settings
from app.investigation.models import EvidenceResult, TriageResult


class InvestigationProvider(Protocol):
    async def recommend(self, transaction: dict, score: dict, triage: TriageResult,
                        evidence: EvidenceResult) -> dict: ...


class DeterministicProvider:
    async def recommend(self, transaction, score, triage, evidence):
        ids = [item.evidence_id for item in evidence.items]
        if evidence.insufficient_evidence:
            action, confidence = 'ESCALATE_FOR_REVIEW', 0.35
            summary = 'Insufficient corroborating history or profile; manual investigation needed.'
        elif score['fraud_score'] >= 0.85 and sum(item.anomalous for item in evidence.items) >= 2:
            action, confidence = 'BLOCK', 0.88
            summary = 'Critical score with at least two corroborating account, device, location, or velocity anomalies.'
        elif score['fraud_score'] >= 0.6:
            action, confidence = 'REQUEST_VERIFICATION', 0.74
            summary = 'Elevated score with available account evidence; verify the customer.'
        elif score['fraud_score'] >= 0.3:
            action, confidence = 'REQUEST_VERIFICATION', 0.7
            summary = 'Moderate score; request verification before final approval.'
        else:
            action, confidence = 'APPROVE', 0.9
            summary = 'Low fraud score; no investigation trigger.'
        return {'recommended_action': action, 'confidence': confidence,
                'risk_summary': summary, 'evidence_ids': ids, 'requires_human_review': False}


class UnavailableProvider:
    async def recommend(self, transaction, score, triage, evidence):
        raise ConnectionError('Investigation model unavailable')


class CircuitBreaker:
    """Opens after repeated transport failures so a dead provider fails fast."""

    def __init__(self, threshold=3, reset_seconds=30):
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.opened_at = 0.0

    def allow(self):
        if self.failures < self.threshold:
            return True
        if time.monotonic() - self.opened_at >= self.reset_seconds:
            self.failures = 0
            return True
        return False

    def success(self):
        self.failures = 0

    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


openai_breaker = CircuitBreaker()


def reset_openai_breaker():
    openai_breaker.failures = 0
    openai_breaker.opened_at = 0.0


class OpenAIProvider:
    """One bounded Responses API call; evidence retrieval remains in Python."""

    API_URL = 'https://api.openai.com/v1/responses'
    OUTPUT_SCHEMA = {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'recommended_action': {'type': 'string', 'enum': ['APPROVE', 'BLOCK', 'REQUEST_VERIFICATION', 'ESCALATE_FOR_REVIEW']},
            'confidence': {'type': 'number'},
            'risk_summary': {'type': 'string'},
            'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
            'requires_human_review': {'type': 'boolean'},
        },
        'required': ['recommended_action', 'confidence', 'risk_summary', 'evidence_ids', 'requires_human_review'],
    }

    def __init__(self, api_key=None, model=None, client=None):
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.client = client

    async def _post(self, payload, headers):
        if self.client is None:
            async with httpx.AsyncClient(timeout=settings.OPENAI_TIMEOUT_SECONDS) as client:
                return await client.post(self.API_URL, json=payload, headers=headers)
        return await self.client.post(self.API_URL, json=payload, headers=headers)

    async def recommend(self, transaction, score, triage, evidence):
        if not self.api_key:
            raise ConnectionError('Investigation model credential unavailable')
        if not openai_breaker.allow():
            raise ConnectionError('Investigation model circuit open')
        # Send only fields needed for this decision, never the complete transaction/profile.
        evidence_aliases = {f'E{i + 1}': item.evidence_id for i, item in enumerate(evidence.items)}
        case_input = {
            'transaction': {key: transaction.get(key) for key in ('amount', 'country', 'merchant_category')},
            'fraud_score': score['fraud_score'], 'risk_level': score['risk_level'],
            'risk_factors': score.get('risk_factors', []), 'rules_action': triage.rules_action,
            'insufficient_evidence': evidence.insufficient_evidence,
            'missing_signals': [gap.signal for gap in evidence.gaps],
            'evidence': [{'evidence_id': alias, 'kind': item.kind, 'detail': item.detail,
                          'anomalous': item.anomalous}
                         for alias, item in zip(evidence_aliases, evidence.items)],
        }
        payload = {
            'model': self.model, 'store': False,
            'instructions': ('You are a fraud case investigator. Use only the supplied facts. '
                             'Cite only listed evidence_ids. Never invent account history or claim a payment was blocked. '
                             'If evidence is weak, choose ESCALATE_FOR_REVIEW and lower confidence. '
                             'Give a concise explanation; your confidence is an uncertainty estimate, not a calibrated fraud probability. '
                             'The final human-review policy is enforced by the application.'),
            'input': json.dumps(case_input, default=str),
            'text': {'format': {'type': 'json_schema', 'name': 'fraud_investigation',
                                'strict': True, 'schema': self.OUTPUT_SCHEMA}},
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = None
        for attempt in range(3):
            try:
                response = await self._post(payload, headers)
            except httpx.RequestError as exc:
                openai_breaker.failure()
                raise ConnectionError('Investigation model request unavailable') from exc
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    continue
                openai_breaker.failure()
                raise ConnectionError('Investigation model request unavailable')
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                openai_breaker.failure()
                raise ConnectionError('Investigation model request unavailable') from exc
            openai_breaker.success()
            break
        body = response.json()
        if body.get('status') != 'completed':
            raise ConnectionError('Investigation model did not complete')
        chunks = [part.get('text') for output in body.get('output', [])
                  for part in output.get('content', []) if part.get('type') == 'output_text']
        if not chunks:
            raise ValueError('Investigation model returned no structured output')
        result = json.loads(''.join(chunks))
        # A fabricated alias stays unresolved and fails the graph's citation check.
        if isinstance(result, dict) and isinstance(result.get('evidence_ids'), list):
            result['evidence_ids'] = [evidence_aliases.get(alias, alias) for alias in result['evidence_ids']]
        return result


def configured_provider():
    if settings.INVESTIGATION_PROVIDER == 'deterministic':
        return DeterministicProvider()
    if settings.INVESTIGATION_PROVIDER == 'openai':
        return OpenAIProvider()
    raise ValueError('INVESTIGATION_PROVIDER must be deterministic or openai')
