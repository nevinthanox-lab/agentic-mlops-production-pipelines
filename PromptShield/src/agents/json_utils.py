"""
PromptShield - Shared JSON Extraction & Repair Utility
=========================================================
Every agent in the PromptShield crew (Triage, Sanitizer, Judge, Escalation)
asks a local Ollama model to return a strict JSON object matching a
Pydantic schema. Small local models (llama3.2 etc.) frequently produce
JSON that is *almost* valid but has one of these common defects:

    - Wrapped in ```json ... ``` markdown fences
    - Trailing commas before a closing brace/bracket
    - Extra prose before or after the JSON object
    - Single quotes instead of double quotes
    - Missing quotes around keys
    - Truncated output (cut off mid-object)

Rather than hand-rolling regex fixes for each case (error-prone, and JSON
is a hierarchical grammar that regex fundamentally cannot parse safely),
this module uses the `json_repair` library, which uses a proper
parser-based repair strategy. This is the current recommended approach
for post-processing LLM JSON output.

Install dependency:
    pip install json-repair

Place this file at: C:\\projects\\PromptShield\\src\\agents\\json_utils.py
"""

from __future__ import annotations

import json
import re
from typing import Optional, Type, TypeVar

from json_repair import repair_json
from loguru import logger
from pydantic import BaseModel, ValidationError

# Generic type variable bound to any Pydantic model, so callers get proper
# type hints back (e.g. extract_and_validate(text, TriageDisposition) -> TriageDisposition)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _strip_markdown_fences(raw_text: str) -> str:
    """
    Removes ```json / ``` code fences that local models often wrap their
    JSON output in, even when explicitly told not to.
    """
    cleaned = re.sub(r"```(?:json|JSON)?", "", raw_text)
    return cleaned.strip()


def extract_json_dict(raw_text: str) -> Optional[dict]:
    """
    Attempts to extract a valid Python dict from raw, potentially messy
    LLM output text. Uses a two-stage strategy:

        1. Strip markdown fences, then try strict json.loads() first
           (fast path -- most well-behaved models hit this).
        2. If that fails, fall back to json_repair.repair_json(), which
           performs proper parser-based repair of common LLM JSON defects.

    Args:
        raw_text: The raw string output from a CrewAI agent / Ollama model.

    Returns:
        A parsed dict if extraction succeeded, otherwise None. Never
        raises -- all exceptions are caught and logged, since this function
        sits on the fail-closed path and callers must handle None explicitly.
    """
    cleaned = _strip_markdown_fences(raw_text)

    # --- Fast path: strict parsing ---
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        logger.warning(f"Strict JSON parse succeeded but result is not a dict: {type(result)}")
    except json.JSONDecodeError:
        pass  # fall through to repair path

    # --- Repair path: json_repair handles trailing commas, single quotes,
    #     missing quotes, stray prose, truncated objects, etc. ---
    try:
        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            logger.info("JSON successfully recovered via json_repair fallback.")
            return repaired
        logger.warning(f"json_repair returned a non-dict object: {type(repaired)}")
        return None
    except Exception as exc:  # noqa: BLE001 - json_repair can raise various internal errors
        logger.error(f"json_repair failed to recover a valid object: {exc}")
        return None


def extract_and_validate(
    raw_text: str,
    schema: Type[SchemaT],
) -> "tuple[Optional[SchemaT], Optional[str]]":
    """
    Combines JSON extraction/repair with Pydantic schema validation in a
    single call. This is the function agent files should use directly.

    Args:
        raw_text: The raw string output from a CrewAI agent / Ollama model.
        schema: The Pydantic model class to validate the extracted dict against
            (e.g. TriageDisposition, SanitizerOutput, JudgeVerdict, EscalationReport).

    Returns:
        A tuple of (validated_instance, error_message):
            - (instance, None) on success
            - (None, "description of what went wrong") on failure

        Callers MUST check for None and apply their own fail-closed fallback
        logic -- this function never raises and never guesses defaults.
    """
    parsed_dict = extract_json_dict(raw_text)

    if parsed_dict is None:
        error_msg = f"Could not extract any valid JSON object from raw output: {raw_text[:300]}"
        logger.warning(error_msg)
        return None, error_msg

    try:
        instance = schema(**parsed_dict)
        return instance, None
    except ValidationError as exc:
        error_msg = f"Pydantic validation failed for schema {schema.__name__}: {exc}"
        logger.warning(error_msg)
        return None, error_msg
