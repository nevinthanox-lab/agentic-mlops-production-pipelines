"""
PromptShield - Agent Output Schemas
=====================================
This module defines STRICT, FAIL-CLOSED Pydantic schemas for every agent
in the CrewAI quarantine pipeline (Triage -> Sanitizer -> Judge -> Escalation).

Design principle: FAIL-CLOSED.
    If an agent's raw LLM output cannot be parsed into one of these schemas,
    the orchestrator MUST treat that as a validation failure and escalate
    the request to the highest risk tier rather than silently passing it
    through. These schemas therefore use `extra="forbid"` and strict field
    types so that malformed or hallucinated JSON from a local LLM is
    rejected immediately instead of silently coerced.

Place this file at: C:\\projects\\PromptShield\\src\\agents\\schemas.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared Enums
# ---------------------------------------------------------------------------

class RiskTier(str, Enum):
    """Risk tiers used consistently across the classifier and all agents."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AttackCategory(str, Enum):
    """Must match the label classes produced by the DistilBERT / GBM classifier."""
    BENIGN = "benign"
    DIRECT_INJECTION = "direct_injection"
    INSTRUCTION_OVERRIDE = "instruction_override"
    OBFUSCATION = "obfuscation"
    ROLEPLAY = "roleplay"


class DispositionDecision(str, Enum):
    """The action the Triage Analyst / Judge recommend for a given prompt."""
    ALLOW = "allow"
    SANITIZE = "sanitize"
    BLOCK = "block"
    ESCALATE = "escalate"


class AgentStatus(str, Enum):
    """Whether an agent's own execution succeeded or failed validation."""
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    LLM_ERROR = "llm_error"


# ---------------------------------------------------------------------------
# Agent 1: Triage Analyst
# ---------------------------------------------------------------------------

class TriageDisposition(BaseModel):
    """
    Strict output contract for the Triage Analyst agent.

    The Triage Analyst receives the raw prompt + the classifier's
    {label, confidence, risk_tier} and must produce an initial disposition
    decision along with a short justification for audit purposes.
    """
    model_config = {"extra": "forbid"}

    prompt_id: str = Field(..., min_length=1, description="Unique ID for the prompt under review")
    attack_category: AttackCategory
    classifier_risk_tier: RiskTier
    disposition: DispositionDecision
    justification: str = Field(..., min_length=10, max_length=1000)
    triage_confidence: float = Field(..., ge=0.0, le=1.0)
    requires_sanitization: bool
    requires_judge_review: bool
    agent_status: AgentStatus = AgentStatus.SUCCESS
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("justification")
    @classmethod
    def justification_must_be_meaningful(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("justification is too short to be meaningful; fail-closed rejection")
        return stripped

    @model_validator(mode="after")
    def enforce_disposition_consistency(self) -> "TriageDisposition":
        """
        Fail-closed consistency rule:
        A BLOCK or ESCALATE disposition can never coexist with a
        classifier_risk_tier of LOW, since that combination implies the
        agent is disagreeing wildly with the model without strong grounds.
        We do not silently fix this -- we raise, forcing the orchestrator
        to treat the whole agent output as invalid and escalate manually.
        """
        if self.classifier_risk_tier == RiskTier.LOW and self.disposition in (
            DispositionDecision.BLOCK,
            DispositionDecision.ESCALATE,
        ):
            raise ValueError(
                "Inconsistent triage output: LOW risk_tier cannot map to "
                "BLOCK/ESCALATE without explicit judge review. Fail-closed."
            )
        return self


# ---------------------------------------------------------------------------
# Agent 2: Sanitizer (only invoked for MEDIUM risk)
# ---------------------------------------------------------------------------

class SanitizerOutput(BaseModel):
    """
    Strict output contract for the Sanitizer agent.

    The Sanitizer only runs on medium-risk prompts. It attempts to strip
    adversarial content (role-play framing, override instructions,
    obfuscated tokens) while preserving the user's legitimate intent.
    """
    model_config = {"extra": "forbid"}

    prompt_id: str = Field(..., min_length=1)
    original_prompt: str = Field(..., min_length=1)
    sanitized_prompt: str = Field(..., min_length=1)
    tokens_removed: List[str] = Field(default_factory=list)
    sanitization_confidence: float = Field(..., ge=0.0, le=1.0)
    residual_risk_detected: bool = Field(
        ..., description="True if the sanitizer itself still sees risk after cleaning"
    )
    agent_status: AgentStatus = AgentStatus.SUCCESS
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def sanitized_must_differ_or_flag(self) -> "SanitizerOutput":
        """
        Fail-closed rule: if sanitized_prompt is identical to original_prompt,
        the sanitizer MUST explicitly flag residual_risk_detected=True.
        A silent no-op sanitization pass is not allowed to look "clean".
        """
        if self.sanitized_prompt.strip() == self.original_prompt.strip():
            if not self.residual_risk_detected:
                raise ValueError(
                    "Sanitizer made no changes but did not flag residual risk. "
                    "Fail-closed rejection to force explicit escalation."
                )
        return self


# ---------------------------------------------------------------------------
# Agent 3: LLM-Judge (local Ollama model, cross-validates Triage)
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    """
    Strict output contract for the LLM-Judge agent.

    The Judge independently re-assesses the prompt using a local Ollama
    model and states whether it AGREES or DISAGREES with the Triage
    Analyst's disposition. This is the cross-validation step designed to
    catch correlated blind spots between the classifier and Triage agent.
    """
    model_config = {"extra": "forbid"}

    prompt_id: str = Field(..., min_length=1)
    triage_disposition_reviewed: DispositionDecision
    judge_disposition: DispositionDecision
    agrees_with_triage: bool
    judge_reasoning: str = Field(..., min_length=10, max_length=1000)
    judge_confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str = Field(..., description="Ollama model tag, e.g. 'llama3.2:latest'")
    agent_status: AgentStatus = AgentStatus.SUCCESS
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def agreement_flag_must_match_dispositions(self) -> "JudgeVerdict":
        """
        Fail-closed rule: agrees_with_triage must be mathematically
        consistent with the two disposition fields. An LLM might say
        agrees_with_triage=True while actually outputting a different
        disposition value -- we never trust the LLM's own boolean claim,
        we recompute it and reject if it lied.
        """
        computed_agreement = self.triage_disposition_reviewed == self.judge_disposition
        if computed_agreement != self.agrees_with_triage:
            raise ValueError(
                "Judge's agrees_with_triage flag contradicts the actual "
                "disposition comparison. Fail-closed rejection."
            )
        return self


# ---------------------------------------------------------------------------
# Agent 4: Escalation Notifier
# ---------------------------------------------------------------------------

class EscalationReport(BaseModel):
    """
    Strict output contract for the Escalation Notifier agent.

    Fires whenever: risk_tier == HIGH, OR Triage/Judge disagree, OR any
    upstream agent returned agent_status != SUCCESS. Packages a full
    incident report for Slack + MongoDB persistence.
    """
    model_config = {"extra": "forbid"}

    incident_id: str = Field(..., min_length=1)
    prompt_id: str = Field(..., min_length=1)
    original_prompt: str = Field(..., min_length=1)
    attack_category: AttackCategory
    classifier_risk_tier: RiskTier
    triage_disposition: Optional[DispositionDecision] = None
    judge_disposition: Optional[DispositionDecision] = None
    agents_disagreed: bool
    escalation_reason: str = Field(..., min_length=10, max_length=2000)
    severity: RiskTier
    recommended_action: DispositionDecision
    notify_slack: bool = True
    agent_status: AgentStatus = AgentStatus.SUCCESS
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("incident_id")
    @classmethod
    def incident_id_must_be_prefixed(cls, v: str) -> str:
        if not v.startswith("INC-"):
            raise ValueError("incident_id must be prefixed with 'INC-' for traceability")
        return v


# ---------------------------------------------------------------------------
# Orchestrator-level wrapper: full pipeline trace for a single request
# ---------------------------------------------------------------------------

class PipelineTrace(BaseModel):
    """
    The complete, end-to-end record of one request moving through the
    4-agent crew. This is what gets written to MongoDB as the
    variable-schema incident document, and rolled up into Postgres.
    """
    model_config = {"extra": "forbid"}

    prompt_id: str = Field(..., min_length=1)
    triage: Optional[TriageDisposition] = None
    sanitizer: Optional[SanitizerOutput] = None
    judge: Optional[JudgeVerdict] = None
    escalation: Optional[EscalationReport] = None
    final_disposition: DispositionDecision
    pipeline_status: AgentStatus = AgentStatus.SUCCESS
    total_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
