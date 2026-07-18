"""
STEP 6.1-6.3 - RAG Retrieval + Grounded Generation
Retrieves relevant runbook chunks from pgvector via LlamaIndex, then calls
Groq (Llama 3.3 70B, OpenAI-compatible endpoint) under a system prompt that
enforces:
  (a) remediation grounded ONLY in retrieved runbook content, never free-lanced
  (b) read-only diagnostic commands by default
  (c) any destructive-potential command only inside a clearly labeled
      "REQUIRES HUMAN APPROVAL" block, never directly executable
Also implements the grounding-confidence check: if retrieval returns nothing
above a similarity threshold, output explicitly says to escalate to a human
SRE instead of letting the LLM hallucinate a fix.

STEP 9.2.2.1 - Added retry/backoff wrapper around the Groq call to handle
HTTP 429 (rate limit / TPD exceeded) errors. Groq returns a
"Please try again in Xs" message inside the error - we parse that and sleep
exactly that long (plus a small buffer) before retrying, up to MAX_RETRIES
times, instead of the whole /remediate call failing with a 502.
"""

import os
import re
import time
import torch  # noqa: F401 - import before sentence-transformers to avoid Windows DLL init-order conflict
import json
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from openai import OpenAI, RateLimitError

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "sentinelops")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sentinelops_dev_pw")
DB_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "sentinelops")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# STEP 6.3 - grounding-confidence threshold (cosine similarity 0-1 scale
# after LlamaIndex's default normalization). Below this -> escalate, no LLM call.
GROUNDING_CONFIDENCE_THRESHOLD = 0.40
TOP_K = 3

# STEP 9.2.2.1 - retry/backoff settings for Groq 429s
MAX_RETRIES = 6
DEFAULT_BACKOFF_SECONDS = 15
MAX_WAIT_CAP_SECONDS = 300  # never wait longer than 5 min on a single retry

SYSTEM_PROMPT = """You are an SRE remediation assistant. You MUST follow these rules exactly:

1. GROUNDED ONLY: Base your remediation steps ONLY on the retrieved runbook content provided below.
   Do not invent steps, commands, or root causes that are not present in the retrieved runbooks.
2. READ-ONLY BY DEFAULT: List diagnostic/investigation steps as plain, directly-actionable read-only steps.
3. DESTRUCTIVE COMMANDS GATED: Any step with destructive potential (restart, scale-down, delete, kill,
   rollback, failover) MUST be placed inside a clearly labeled section titled exactly:
   "REQUIRES HUMAN APPROVAL"
   Never present a destructive action as a directly-executable step outside that block.
4. Output valid JSON matching this exact schema:
{
  "matched_pattern": "<name of the closest matching runbook pattern>",
  "root_cause_hypothesis": "<1-2 sentence hypothesis, grounded in the runbook>",
  "diagnostic_steps": ["<read-only step 1>", "<read-only step 2>", "..."],
  "requires_human_approval": ["<destructive step 1>", "..."]
}
"""


def get_query_engine_components():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    vector_store = PGVectorStore.from_params(
        database=DB_NAME, host=DB_HOST, password=DB_PASSWORD,
        port=DB_PORT, user=DB_USER, table_name="sre_runbooks", embed_dim=384,
    )
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    return retriever


def _parse_retry_after_seconds(error: Exception, fallback: float) -> float:
    """Groq's 429 body includes text like 'Please try again in 3m44.64s'
    or '12.5s'. Pull that number out so we wait the exact right amount
    instead of guessing."""
    msg = str(error)

    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2))
        return minutes * 60 + seconds

    return fallback


def _call_groq_with_retry(client: OpenAI, **kwargs):
    """STEP 9.2.2.1 - wraps client.chat.completions.create with retry/backoff
    on RateLimitError (HTTP 429). Reads Groq's own 'try again in Xs' hint
    when present; otherwise falls back to exponential backoff."""
    backoff = DEFAULT_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise

            wait_s = _parse_retry_after_seconds(e, fallback=backoff)
            wait_s = min(wait_s, MAX_WAIT_CAP_SECONDS) + 2  # small safety buffer

            print(f"[WARN] Groq 429 rate limit (attempt {attempt}/{MAX_RETRIES}). "
                  f"Waiting {wait_s:.1f}s before retry...")
            time.sleep(wait_s)
            backoff *= 2

    raise RuntimeError("Groq retry loop exited unexpectedly")


def generate_remediation(anomaly_description: str, retriever) -> dict:
    """
    STEP 6.1-6.3 core function. Takes a natural-language anomaly description
    (built from a detection payload), retrieves grounding context, checks
    grounding confidence, and if confident enough calls Groq for a
    guardrailed, grounded remediation.
    """
    nodes = retriever.retrieve(anomaly_description)

    if not nodes or nodes[0].score < GROUNDING_CONFIDENCE_THRESHOLD:
        return {
            "grounding_confidence": "low",
            "top_similarity_score": round(float(nodes[0].score), 4) if nodes else 0.0,
            "result": "NO CONFIDENT RUNBOOK MATCH - escalate to human SRE. "
                      "Do not auto-generate remediation for this anomaly pattern.",
            "retrieved_sources": [],
        }

    context_chunks = []
    sources = []
    for n in nodes:
        context_chunks.append(f"[Source: {n.node.metadata.get('file_name')}]\n{n.node.text}")
        sources.append({"file": n.node.metadata.get("file_name"), "score": round(float(n.score), 4)})

    context_text = "\n\n---\n\n".join(context_chunks)

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    response = _call_groq_with_retry(
        client,
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ANOMALY CONTEXT:\n{anomaly_description}\n\n"
                                         f"RETRIEVED RUNBOOK CONTENT:\n{context_text}\n\n"
                                         f"Generate the grounded remediation JSON now."}
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        parsed = {"error": "LLM returned malformed JSON", "raw": response.choices[0].message.content}

    return {
        "grounding_confidence": "high",
        "top_similarity_score": round(float(nodes[0].score), 4),
        "result": parsed,
        "retrieved_sources": sources,
    }


if __name__ == "__main__":
    retriever = get_query_engine_components()

    test_cases = [
        "service checkout-service: p95 latency +340% over a 5-min window while CPU stayed flat "
        "- contributing metrics: latency_p95_ms, queue_depth",
        "service auth-service: memory_pct climbing steadily over the last 90 minutes, no drop between cycles",
        "service search-service: quantum flux capacitor readings unstable, warp core fluctuating",  # deliberately nonsense - tests grounding-confidence escalation
    ]

    for tc in test_cases:
        print("=" * 80)
        print(f"QUERY: {tc}\n")
        result = generate_remediation(tc, retriever)
        print(json.dumps(result, indent=2))
