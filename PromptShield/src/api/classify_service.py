"""
PromptShield - Phase 3 + Phase 4 + Chroma Pre-Filter Integration
====================================================================
FastAPI service wrapping the fine-tuned DistilBERT classifier, the
CrewAI quarantine pipeline, AND the Chroma known-attack similarity
pre-filter. Exposes POST /classify which returns:
  { label, confidence, risk_tier, feature_breakdown, prompt_id,
    quarantine_triggered, chroma_prefilter }

CHROMA INTEGRATION (new in this version):
    Every request is ALSO checked against the existing 601-embedding
    Chroma known-attack store (data/chroma_db, collection
    'promptshield_known_attacks') via chroma_filter.query_nearest_known_attack().
    This is a CHEAP, near-zero-cost signal (one embedding + one ANN
    lookup) that runs alongside the DistilBERT forward pass.

    Fail-open design for this SPECIFIC signal: if a prompt is a
    near-duplicate of a known attack (distance below threshold) AND the
    DistilBERT classifier's own risk_tier is "low", we conservatively
    UPGRADE the risk_tier to "medium" -- a near-duplicate of a known
    attack pattern is strong independent evidence the classifier may be
    under-confident on this specific input. We never downgrade risk_tier
    based on Chroma (i.e. Chroma can only make things MORE cautious, never
    less), consistent with the fail-closed philosophy used everywhere
    else in this pipeline. If Chroma itself fails (store unreachable),
    we proceed using only the classifier's output -- this pre-filter is a
    cost/recall optimization, not a security guardrail, so failing open
    here (unlike the CrewAI agents) is the correct and intentional design.

Run with:
  uvicorn src.api.classify_service:app --host 0.0.0.0 --port 8000 --reload
"""

import math
import os
import re
import uuid
from collections import Counter
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.agents.crew_orchestrator import run_quarantine_pipeline
from src.api.chroma_filter import query_nearest_known_attack

logger.add("logs/api_service.log", rotation="5 MB")

MODEL_DIR = "models/distilbert_promptshield"
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Risk tier thresholds (confidence of "is an attack" -> risk tier)
# These gate downstream CrewAI routing in Phase 4.
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "low": 0.0,        # benign or very low attack confidence
    "medium": 0.50,    # moderate attack confidence -> sanitizer review
    "high": 0.80,      # high attack confidence -> escalation
}

# Risk tier ordering used to implement the "Chroma can only upgrade,
# never downgrade" rule below.
RISK_TIER_RANK = {"low": 0, "medium": 1, "high": 2}

# ---------------------------------------------------------------------------
# Hand-engineered feature functions (identical logic to training pipeline,
# duplicated here so the API has zero dependency on the training scripts)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = [
    "<|", "|>", "[INST]", "[/INST]", "###", "system:", "assistant:", "user:",
    "<system>", "</system>", "{{", "}}", "<<", ">>",
]

IMPERATIVE_VERBS = [
    "ignore", "disregard", "forget", "override", "bypass", "reveal", "print",
    "act", "pretend", "become", "respond", "answer", "comply", "obey",
    "execute", "output", "generate", "tell", "show", "give", "do",
]

ROLEPLAY_LEXICON = [
    "pretend", "roleplay", "persona", "character", "dan mode", "jailbreak",
    "developer mode", "act as", "you are now", "simulate", "stay in character",
    "immersive", "fictional", "story mode",
]


def count_special_tokens(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(tok.lower()) for tok in SPECIAL_TOKENS)


def imperative_verb_density(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    imperative_count = sum(1 for w in words if w in IMPERATIVE_VERBS)
    return imperative_count / len(words)


def roleplay_lexicon_score(text: str) -> float:
    lower = text.lower()
    hits = sum(lower.count(phrase) for phrase in ROLEPLAY_LEXICON)
    return hits / max(len(text) / 100, 1)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def compute_feature_breakdown(text: str) -> dict:
    """Returns the 5 hand-engineered features for transparency in the API response.
    Note: entropy and length z-scores are computed relative to this single prompt
    (no corpus context at inference time), so they are reported as raw values here
    rather than corpus-relative z-scores."""
    return {
        "special_token_count": count_special_tokens(text),
        "imperative_verb_density": round(imperative_verb_density(text), 4),
        "roleplay_lexicon_score": round(roleplay_lexicon_score(text), 4),
        "char_entropy": round(shannon_entropy(text), 4),
        "prompt_length_chars": len(text),
    }


def determine_risk_tier(attack_confidence: float) -> str:
    if attack_confidence >= RISK_THRESHOLDS["high"]:
        return "high"
    if attack_confidence >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="The prompt text to classify")


class ChromaPreFilterInfo(BaseModel):
    nearest_category: Optional[str] = None
    similarity_distance: Optional[float] = None
    is_near_duplicate: bool = False
    risk_tier_upgraded: bool = Field(
        False, description="True if this Chroma match caused risk_tier to be "
        "conservatively upgraded from the classifier's own determination."
    )


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    risk_tier: str
    feature_breakdown: dict
    prompt_id: str = Field(
        ..., description="Stable ID for this request, used to correlate with the "
        "background quarantine pipeline's PipelineTrace document in MongoDB."
    )
    quarantine_triggered: bool = Field(
        ..., description="True if risk_tier was not 'low' and the 4-agent CrewAI "
        "quarantine pipeline was scheduled to run in the background for this prompt_id."
    )
    chroma_prefilter: ChromaPreFilterInfo = Field(
        ..., description="Result of the cheap Chroma known-attack similarity pre-filter."
    )


# ---------------------------------------------------------------------------
# FastAPI app + model loading (once, at startup)
# ---------------------------------------------------------------------------
app = FastAPI(title="PromptShield Classifier", version="1.2.0")

_model = None
_tokenizer = None
_label_classes = None
_device = None


@app.on_event("startup")
def load_model():
    global _model, _tokenizer, _label_classes, _device

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"Model directory '{MODEL_DIR}' not found. Run src/train_distilbert.py first."
        )

    logger.info(f"Loading DistilBERT model from {MODEL_DIR} ...")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()

    label_classes_path = os.path.join(MODEL_DIR, "label_classes.txt")
    with open(label_classes_path, "r") as f:
        _label_classes = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"Model loaded on device: {_device}")
    logger.info(f"Label classes: {_label_classes}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest, background_tasks: BackgroundTasks):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Generate a stable prompt_id NOW, before any classification happens, so
    # it can be returned to the caller AND passed into the background
    # quarantine pipeline for end-to-end correlation in MongoDB.
    prompt_id = f"PROMPT-{uuid.uuid4().hex[:12]}"

    logger.info(f"Classifying prompt_id={prompt_id} of length {len(text)} chars")

    # -----------------------------------------------------------------
    # DistilBERT classification (the primary, authoritative signal)
    # -----------------------------------------------------------------
    inputs = _tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    predicted_idx = int(probs.argmax())
    predicted_label = _label_classes[predicted_idx]
    confidence = float(probs[predicted_idx])

    benign_idx = _label_classes.index("benign") if "benign" in _label_classes else None
    if benign_idx is not None:
        attack_confidence = 1.0 - float(probs[benign_idx])
    else:
        attack_confidence = confidence if predicted_label != "benign" else 0.0

    risk_tier = determine_risk_tier(attack_confidence)
    feature_breakdown = compute_feature_breakdown(text)

    logger.info(
        f"Result for prompt_id={prompt_id}: label={predicted_label} "
        f"confidence={confidence:.4f} risk_tier={risk_tier} "
        f"attack_confidence={attack_confidence:.4f}"
    )

    # -----------------------------------------------------------------
    # Chroma known-attack similarity pre-filter (cheap secondary signal)
    # -----------------------------------------------------------------
    chroma_result = query_nearest_known_attack(text)
    risk_tier_upgraded = False

    if chroma_result.query_succeeded and chroma_result.is_near_duplicate:
        # Fail-closed-STYLE upgrade rule: Chroma can only push risk_tier UP,
        # never down. A near-duplicate of a known attack pattern overrides
        # classifier under-confidence, but a "no match" from Chroma never
        # overrides classifier over-confidence in the other direction.
        chroma_implied_tier = "medium"
        if RISK_TIER_RANK[chroma_implied_tier] > RISK_TIER_RANK[risk_tier]:
            logger.info(
                f"prompt_id={prompt_id}: Chroma near-duplicate of known "
                f"'{chroma_result.nearest_category}' attack (distance="
                f"{chroma_result.similarity_distance}) -- upgrading risk_tier "
                f"from '{risk_tier}' to '{chroma_implied_tier}'."
            )
            risk_tier = chroma_implied_tier
            risk_tier_upgraded = True

    chroma_prefilter_info = ChromaPreFilterInfo(
        nearest_category=chroma_result.nearest_category,
        similarity_distance=chroma_result.similarity_distance,
        is_near_duplicate=chroma_result.is_near_duplicate,
        risk_tier_upgraded=risk_tier_upgraded,
    )

    # -----------------------------------------------------------------
    # Phase 4 hook: anything above "low" risk (AFTER the Chroma upgrade
    # rule above has been applied) gets routed into the 4-agent CrewAI
    # quarantine pipeline, as a BackgroundTask so the HTTP response below
    # returns immediately.
    # -----------------------------------------------------------------
    quarantine_triggered = risk_tier != "low"

    if quarantine_triggered:
        logger.info(
            f"risk_tier='{risk_tier}' for prompt_id={prompt_id} -> "
            "scheduling background CrewAI quarantine pipeline run."
        )
        background_tasks.add_task(
            run_quarantine_pipeline,
            prompt_text=text,
            classifier_label=predicted_label,
            classifier_confidence=confidence,
            classifier_risk_tier=risk_tier,
            prompt_id=prompt_id,
        )
    else:
        logger.info(f"risk_tier='low' for prompt_id={prompt_id} -> quarantine pipeline skipped.")

    return ClassifyResponse(
        label=predicted_label,
        confidence=round(confidence, 4),
        risk_tier=risk_tier,
        feature_breakdown=feature_breakdown,
        prompt_id=prompt_id,
        quarantine_triggered=quarantine_triggered,
        chroma_prefilter=chroma_prefilter_info,
    )
