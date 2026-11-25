from typing import Literal
from pydantic import BaseModel, Field, model_validator

Action = Literal['APPROVE', 'BLOCK', 'REQUEST_VERIFICATION', 'ESCALATE_FOR_REVIEW']
ReviewReason = Literal['LOW_CONFIDENCE', 'INSUFFICIENT_EVIDENCE', 'RULE_DISAGREEMENT',
                       'BLOCK_RECOMMENDATION', 'MODEL_FALLBACK', 'UNCITED_ANOMALOUS_EVIDENCE']


class TriageResult(BaseModel):
    requires_investigation: bool
    rules_action: Action
    reason: str


class EvidenceItem(BaseModel):
    evidence_id: str
    kind: str
    detail: str
    anomalous: bool = False
    transaction_ids: list[str] = Field(default_factory=list)


class EvidenceGap(BaseModel):
    signal: Literal['RECENT_HISTORY', 'AMOUNT_BASELINE', 'DEVICE_BASELINE', 'COUNTRY_BASELINE']
    reason: str = Field(min_length=1)
    next_check: str = Field(min_length=1)


class EvidenceResult(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    insufficient_evidence: bool = False
    reason: str = ''
    gaps: list[EvidenceGap] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    recommended_action: Action
    confidence: float = Field(ge=0, le=1)
    risk_summary: str = Field(min_length=1)
    evidence_ids: list[str]
    requires_human_review: bool

    @model_validator(mode='after')
    def require_citation(self):
        if not self.evidence_ids:
            raise ValueError('Recommendation must cite retrieved evidence')
        return self


class Case(BaseModel):
    case_id: str
    transaction_id: str
    status: Literal['PENDING', 'COMPLETE', 'REVIEW_REQUIRED', 'REVIEWED', 'FAILED']
    triage: TriageResult | None = None
    evidence: EvidenceResult | None = None
    result: InvestigationResult | None = None
    final_action: Action | None = None
    reviewer: str | None = None
    review_note: str | None = None
    error: str | None = None
    investigation_mode: Literal['deterministic', 'openai'] | None = None
    provider_fallback: bool = False
    review_reasons: list[ReviewReason] = Field(default_factory=list)
