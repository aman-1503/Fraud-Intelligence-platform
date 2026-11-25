"""Three-node LangGraph investigation, run outside the scoring request path."""
from typing import TypedDict
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from app.core.config import settings
from app.investigation.models import EvidenceGap, EvidenceItem, EvidenceResult, InvestigationResult, TriageResult
from app.investigation.provider import DeterministicProvider, OpenAIProvider, configured_provider


class State(TypedDict, total=False):
    transaction: dict
    score: dict
    triage: dict
    evidence: dict
    result: dict
    provider_fallback: bool
    investigation_mode: str
    review_reasons: list[str]


def _time(value):
    if isinstance(value, datetime): return value.replace(tzinfo=value.tzinfo or timezone.utc)
    if isinstance(value, str):
        try: return datetime.fromisoformat(value.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
        except ValueError: return None
    return None


class InvestigationWorkflow:
    def __init__(self, db, cache, provider=None):
        self.db, self.cache = db, cache
        self.provider = provider if provider is not None else configured_provider()
        graph = StateGraph(State)
        graph.add_node('triage_node', self.triage)
        graph.add_node('evidence_node', self.evidence)
        graph.add_node('investigation_node', self.investigation)
        graph.set_entry_point('triage_node')
        # Evidence always runs. Skipping it used to let investigation() invent a score item.
        graph.add_edge('triage_node', 'evidence_node')
        graph.add_edge('evidence_node', 'investigation_node')
        graph.add_edge('investigation_node', END)
        self.graph = graph.compile()

    async def triage(self, state: State):
        score = state['score']
        probability = float(score['fraud_score'])
        factors = score.get('risk_factors', [])
        flagged = probability >= settings.INVESTIGATION_SCORE_THRESHOLD or score['risk_level'] in ('HIGH', 'CRITICAL') or len(factors) >= 2
        action = 'BLOCK' if probability >= 0.85 else 'ESCALATE_FOR_REVIEW' if flagged else 'APPROVE'
        result = TriageResult(requires_investigation=flagged, rules_action=action,
                              reason='score or risk-factor trigger' if flagged else 'below investigation threshold')
        return {'triage': result.model_dump()}

    async def evidence(self, state: State):
        txn = state['transaction']
        user_id, txn_id = txn['user_id'], txn['transaction_id']
        items = [EvidenceItem(evidence_id=f'score:{txn_id}', kind='score',
                              detail=f"Recorded fraud score {state['score']['fraud_score']:.3f}", transaction_ids=[txn_id])]
        profile = await self.db.get_user_profile(user_id)
        history = await self.db.get_user_transactions(user_id, 50)
        by_id = {t['transaction_id']: t for t in history if t.get('transaction_id') and t['transaction_id'] != txn_id}
        history = list(by_id.values())
        if profile:
            avg, std = profile.get('avg_txn_amount'), profile.get('std_txn_amount')
            if avg is not None and std is not None and std > 0:
                z = (float(txn['amount']) - float(avg)) / float(std)
                items.append(EvidenceItem(evidence_id=f'profile:{user_id}:amount', kind='amount_anomaly',
                                          detail=f'Amount z-score against stored profile: {z:.2f}', anomalous=z >= 3))
            home = profile.get('home_country')
            if home:
                items.append(EvidenceItem(evidence_id=f'profile:{user_id}:country', kind='country_change',
                                          detail=f"Stored home country {home}; transaction country {txn.get('country', 'unknown')}",
                                          anomalous=home != txn.get('country')))
            primary = profile.get('primary_device_id')
            if primary and txn.get('device_id'):
                items.append(EvidenceItem(evidence_id=f'profile:{user_id}:device', kind='device_activity',
                                          detail=f"Stored primary device matches transaction: {primary == txn['device_id']}",
                                          anomalous=primary != txn['device_id']))
        if history:
            ids = list(by_id)
            items.append(EvidenceItem(evidence_id=f'history:{user_id}:recent', kind='recent_transactions',
                                      detail=f'{len(history)} stored earlier transactions', transaction_ids=ids))
            now = _time(txn.get('timestamp')) or datetime.now(timezone.utc)
            within_hour = [t for t in history if (ts := _time(t.get('timestamp'))) and 0 <= (now - ts).total_seconds() <= 3600]
            items.append(EvidenceItem(evidence_id=f'history:{user_id}:velocity', kind='transaction_velocity',
                                      detail=f'{len(within_hour)} stored earlier transactions in one hour', anomalous=len(within_hour) >= 3,
                                      transaction_ids=[t['transaction_id'] for t in within_hour]))
            countries = {t.get('country') for t in history if t.get('country')}
            devices = {t.get('device_id') for t in history if t.get('device_id')}
            if countries:
                items.append(EvidenceItem(evidence_id=f'history:{user_id}:countries', kind='country_change',
                                          detail=f"Prior countries: {sorted(countries)}; current: {txn.get('country')}",
                                          anomalous=txn.get('country') not in countries, transaction_ids=ids))
            if devices and txn.get('device_id'):
                items.append(EvidenceItem(evidence_id=f'history:{user_id}:devices', kind='device_activity',
                                          detail=f"Current device in prior devices: {txn['device_id'] in devices}",
                                          anomalous=txn['device_id'] not in devices, transaction_ids=ids))
        kinds = {item.kind for item in items}
        gap_definitions = (
            ('RECENT_HISTORY', 'recent_transactions', 'No earlier transactions are stored for this account.',
             'Check permitted payment history before deciding whether this is a new pattern.'),
            ('AMOUNT_BASELINE', 'amount_anomaly', 'No usable amount baseline is stored.',
             'Check a verified historical amount range; do not infer an anomaly from amount alone.'),
            ('DEVICE_BASELINE', 'device_activity', 'No stored primary or prior device is available.',
             'Check whether this device is familiar to the customer through a permitted source.'),
            ('COUNTRY_BASELINE', 'country_change', 'No stored home or prior country is available.',
             'Check permitted prior locations or verify the customer before treating this as a country change.'),
        )
        gaps = [EvidenceGap(signal=signal, reason=reason, next_check=next_check)
                for signal, kind, reason, next_check in gap_definitions if kind not in kinds]
        insufficient = len(items) == 1
        result = EvidenceResult(items=items, insufficient_evidence=insufficient,
                                reason='No stored profile or prior transactions' if insufficient else '', gaps=gaps)
        return {'evidence': result.model_dump()}

    async def investigation(self, state: State):
        triage = TriageResult.model_validate(state['triage'])
        if not state.get('evidence'):
            raise ValueError('Evidence was not retrieved')
        evidence = EvidenceResult.model_validate(state['evidence'])
        fallback = False
        mode = 'deterministic'
        if not triage.requires_investigation:
            # No agent/provider call for transactions below the deterministic trigger.
            raw = {'recommended_action': 'APPROVE', 'confidence': 0.9,
                   'risk_summary': 'Below deterministic investigation threshold.',
                   'evidence_ids': [evidence.items[0].evidence_id], 'requires_human_review': False}
        elif evidence.insufficient_evidence:
            # A score alone cannot support a model-generated account narrative.
            raw = await DeterministicProvider().recommend(state['transaction'], state['score'], triage, evidence)
        else:
            try:
                raw = await self.provider.recommend(state['transaction'], state['score'], triage, evidence)
                if isinstance(self.provider, OpenAIProvider):
                    mode = 'openai'
            except ConnectionError:
                raw = await DeterministicProvider().recommend(state['transaction'], state['score'], triage, evidence)
                fallback = True
        result = InvestigationResult.model_validate(raw)
        retrieved = {item.evidence_id for item in evidence.items}
        if not set(result.evidence_ids).issubset(retrieved):
            raise ValueError('Agent cited evidence that was not retrieved')
        anomalous = {item.evidence_id for item in evidence.items if item.anomalous}
        uncited_anomalous = not anomalous.issubset(set(result.evidence_ids))
        disagreement = triage.rules_action != result.recommended_action
        review_reasons = []
        if result.confidence < settings.HUMAN_REVIEW_CONFIDENCE_THRESHOLD:
            review_reasons.append('LOW_CONFIDENCE')
        if evidence.insufficient_evidence:
            review_reasons.append('INSUFFICIENT_EVIDENCE')
        if disagreement:
            review_reasons.append('RULE_DISAGREEMENT')
        if uncited_anomalous:
            review_reasons.append('UNCITED_ANOMALOUS_EVIDENCE')
        if result.recommended_action == 'BLOCK':
            review_reasons.append('BLOCK_RECOMMENDATION')
        if fallback:
            review_reasons.append('MODEL_FALLBACK')
        result.requires_human_review = bool(review_reasons)
        return {'evidence': evidence.model_dump(), 'result': result.model_dump(),
                'provider_fallback': fallback, 'investigation_mode': mode,
                'review_reasons': review_reasons}

    async def run(self, transaction, score):
        return await self.graph.ainvoke({'transaction': transaction, 'score': score})
