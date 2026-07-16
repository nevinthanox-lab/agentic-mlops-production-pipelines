"""
PromptShield - Local LLM Configuration for CrewAI
====================================================
This module wires CrewAI's Agent/Task/Crew objects to your LOCAL Ollama
instance, running at http://localhost:11434 with the 'llama3.2:latest'
model already pulled (confirmed via `ollama list`).

No external API keys are used anywhere in this pipeline. Everything runs
fully offline against your local Ollama server.

Place this file at: C:\\projects\\PromptShield\\src\\agents\\llm_config.py
"""

from __future__ import annotations

import os

from crewai import LLM
from loguru import logger


# ---------------------------------------------------------------------------
# Ollama connection settings
# ---------------------------------------------------------------------------

# Base URL for the local Ollama server. This must match what you confirmed
# with `curl http://localhost:11434/api/tags`.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model tag exactly as shown by `ollama list` (NAME column).
OLLAMA_MODEL_TAG: str = os.getenv("OLLAMA_MODEL_TAG", "llama3.2:latest")

# CrewAI (via LiteLLM under the hood) expects the "ollama/<model_tag>" format.
CREWAI_MODEL_STRING: str = f"ollama/{OLLAMA_MODEL_TAG}"


def get_local_llm(temperature: float = 0.1) -> LLM:
    """
    Returns a configured CrewAI LLM object pointed at the local Ollama
    server. Used by every agent in the PromptShield crew so that ZERO
    requests ever leave the machine.

    Args:
        temperature: Sampling temperature. Kept LOW (0.1 default) for
            Triage/Judge/Escalation agents because we want deterministic,
            reproducible security decisions -- not creative variation.

    Returns:
        crewai.LLM: A ready-to-use LLM instance for Agent(llm=...).
    """
    logger.info(
        f"Configuring local LLM -> model='{CREWAI_MODEL_STRING}' "
        f"base_url='{OLLAMA_BASE_URL}' temperature={temperature}"
    )

    llm = LLM(
        model=CREWAI_MODEL_STRING,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        # Ollama does not require an API key, but LiteLLM's client sometimes
        # expects the field to exist. Passing a dummy placeholder value
        # avoids spurious "missing API key" errors while making explicit
        # that no real credential is involved.
        api_key="ollama-local-no-key-required",
    )
    return llm


def verify_ollama_connection() -> bool:
    """
    Performs a lightweight connectivity check against the local Ollama
    server before the crew starts running. This should be called once at
    FastAPI startup (or CrewAI orchestrator startup) so that a dead Ollama
    server fails FAST and LOUD rather than causing silent agent timeouts
    deep inside a CrewAI task.

    Returns:
        bool: True if Ollama responded successfully, False otherwise.
    """
    import requests

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        tags = response.json().get("models", [])
        model_names = [m.get("name", "") for m in tags]

        if OLLAMA_MODEL_TAG not in model_names:
            logger.warning(
                f"Ollama is reachable but model '{OLLAMA_MODEL_TAG}' was not "
                f"found in `ollama list` output: {model_names}. "
                f"Run: ollama pull {OLLAMA_MODEL_TAG.split(':')[0]}"
            )
            return False

        logger.info(f"Ollama connection verified. Available models: {model_names}")
        return True

    except requests.exceptions.RequestException as exc:
        logger.error(
            f"Failed to reach Ollama at {OLLAMA_BASE_URL}. "
            f"Is 'ollama serve' running? Error: {exc}"
        )
        return False
