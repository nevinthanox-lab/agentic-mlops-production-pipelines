# Agentic MLOps Production Pipelines

**A production-grade ecosystem bridging classical ML development loops with GenAI agentic orchestration — three end-to-end systems, each pairing a trained/calibrated model with an LLM-agent layer that acts on its output under strict human-in-the-loop guardrails.**

[![Python](https://img.shields.io/badge/Python-3.11%2F3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservices-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![n8n](https://img.shields.io/badge/n8n-Workflow_Automation-EA4B71?style=flat&logo=n8n&logoColor=white)](https://n8n.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_FSM-1C3C3C?style=flat)](https://www.langchain.com/langgraph)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF6B6B?style=flat)](https://www.crewai.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-6E56CF?style=flat)](https://www.llamaindex.ai/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!--
  🎬 Optional: replace the line below with an actual recorded GIF once available.
  Suggested path: docs/assets/repo_overview.gif
  <p align="center"><img src="docs/assets/repo_overview.gif" alt="Repo overview demo" width="850"></p>
-->

---

## Quick Summary

- 🧩 **Three independent, production-shaped systems** — each with its own trained model, its own agentic layer, and its own n8n orchestration, unified only by a shared architectural philosophy.
- 🛑 **Fail-closed, human-gated by default** — every system stops and asks a human before anything irreversible happens; none of them auto-execute high-stakes actions.
- 📐 **Every metric in every project README is real** — pulled from actual evaluation runs (MLflow, W&B, or `metrics.json`), never invented placeholders.
- 🔌 **A consistent integration pattern** — FastAPI as the strict-schema boundary, n8n as the orchestration/alerting layer, Slack as the human touchpoint — reused across all three, proven three separate ways.
- 🗄️ **Full state traceability** — Supabase, MongoDB+Postgres, or pgvector, every system persists what it did and why, not just what it output.

---

## Repository Structure

```
agentic-mlops-production-pipelines/
├── README.md                          ← you are here
├── LICENSE
├── .gitignore
├── churnguard-retention-engine/        Project 1 — see its own README
├── PromptShield/                       Project 2 — see its own README
└── sentinelops-anomaly-rag/            Project 3 — see its own README
```

Each subfolder is a **fully self-contained project** — its own `requirements.txt`, `docker-compose.yml`, `.env.example`, and README with full setup instructions. This root README is a map, not a substitute for them.

---

## Table of Contents

- [The Shared Philosophy](#the-shared-philosophy)
- [Project 1 — ChurnGuard](#-project-1--churnguard-predictive-churn--agentic-retention-orchestrator)
- [Project 2 — PromptShield](#-project-2--promptshield)
- [Project 3 — SentinelOps](#-project-3--sentinelops-telemetry-anomaly-detection-with-rag-grounded-remediation)
- [Cross-Project Tech Stack](#cross-project-tech-stack)
- [Getting Started](#getting-started)
- [Design Principles Behind All Three](#design-principles-behind-all-three)

---

## The Shared Philosophy

| Without this pattern | With this pattern (all 3 projects) |
|---|---|
| A model's raw prediction gets acted on directly | A model's prediction is explained (SHAP), classified (risk tier), or ensembled (multi-vote) before anything downstream trusts it |
| An LLM agent free-runs to completion | Every agent graph pauses at a defined checkpoint for human approval before anything consequential happens |
| Malformed LLM output silently "works anyway" | Strict Pydantic schemas reject anything malformed — parsing failure escalates, it never defaults to "safe" |
| Nobody can reconstruct why a system did what it did | Every decision path is logged — to Supabase, MongoDB+Postgres, or a replay-tested report — with real timestamps |
| Orchestration is custom glue code per project | n8n provides a visually auditable, swappable orchestration layer across all three |

---

## 📉 Project 1 — ChurnGuard: Predictive Churn & Agentic Retention Orchestrator

**[→ Full README](churnguard-retention-engine/README.md)**

An XGBoost churn-risk classifier hands off SHAP-explained high-risk accounts to a Groq-powered LangGraph agent that drafts personalized retention offers — gated behind human approval in Slack, with every state transition logged to Supabase.

- **Model:** XGBoost (`binary:logistic`), Optuna-tuned, **ROC-AUC 0.91**
- **Business-tuned threshold (0.41):** 87.4% recall / 27.7% precision — deliberately trading precision to catch 97 of 111 actual churners (only 14 false negatives) on the theory that missing a high-ARR churner is far more expensive than a false alarm
- **Explainability:** SHAP `TreeExplainer`, top-3 churn drivers per account passed directly into the agent payload — the same explanation a data scientist would use, not a black box
- **Agent:** 4-node LangGraph FSM (`AnalyzeRiskContext → DraftRetentionOffer → PolicyGuardrailCheck → HITLApprovalGate`) on Groq Llama-3.3-70B
- **Traceability:** every node transition logged to a Supabase `churnguard_audit` table with full state snapshots

<details>
<summary><strong>Tech stack</strong></summary>

XGBoost · SHAP · Optuna · MLflow · FastAPI · LangGraph · Groq (Llama-3.3-70B) · n8n · Slack · Supabase (Postgres)

</details>

---

## 🛡️ Project 2 — PromptShield

**[→ Full README](PromptShield/README.md)**

An adversarial defense pipeline that classifies, quarantines, and escalates prompt-injection attempts against production LLM systems — before a malicious prompt ever reaches the model.

- **Model bake-off:** DistilBERT fine-tune vs. Gradient Boosting vs. Logistic Regression, benchmarked head-to-head in Weights & Biases — **DistilBERT wins: 94.67% test accuracy, 0.9188 weighted F1**
- **Pre-filter:** every prompt checked against a Chroma vector store of 601 known attack embeddings before the classifier even runs
- **4-agent CrewAI quarantine crew:** Triage → Sanitizer (medium-risk only) → independent LLM-Judge (adversarial red-team framing) → Escalation — all running on local Ollama (llama3.2), zero external API calls
- **Fail-closed everywhere:** every agent output is Pydantic-validated; a parsing failure always escalates, it never defaults to "allow" — and the Escalation agent has a deterministic Python fallback so it never shares a single point of failure with the agents it's defending against
- **Zero silent drops:** every request — escalated or not — is written to MongoDB (full trace) and rolled up into Postgres (aggregate metrics), with real-time Slack alerts via n8n

<details>
<summary><strong>Tech stack</strong></summary>

DistilBERT · scikit-learn (GBM, LogReg) · Weights & Biases · Chroma · CrewAI · Ollama (llama3.2) · FastAPI · MongoDB · PostgreSQL · n8n · Slack

</details>

---

## 🔍 Project 3 — SentinelOps: Telemetry Anomaly Detection with RAG-Grounded Remediation

**[→ Full README](sentinelops-anomaly-rag/README.md)**

A 3-model anomaly-detection ensemble scores infrastructure telemetry across 6 simulated microservices; flagged anomalies are matched against a 20-runbook knowledge base via pgvector + LlamaIndex RAG, generating grounded remediation guidance gated behind human approval.

- **Ensemble:** Isolation Forest + One-Class SVM + TensorFlow Autoencoder, evaluated on **12,096 samples across 6 services** (3.51% real anomaly rate)
- **Best individual model:** Autoencoder — **F1 0.2803, ROC-AUC 0.7377** — reconstruction RMSE cleanly separates anomalies (0.77) from normal windows (0.60)
- **RAG grounding:** every flagged anomaly is matched against 20 synthetic SRE runbooks via pgvector; a `high`/`low`/`none` confidence label decides whether to answer or escalate — no hallucinated fixes
- **Guardrails:** destructive commands are gated behind a "REQUIRES HUMAN APPROVAL" block by construction
- **Verified end-to-end:** full replay test over the entire telemetry window — 97 anomalies detected and remediated with **zero API errors**, 97/97 at `high` grounding confidence

<details>
<summary><strong>Tech stack</strong></summary>

scikit-learn (IsolationForest, OneClassSVM) · TensorFlow/Keras (Autoencoder) · DuckDB · MLflow · PostgreSQL + pgvector · LlamaIndex · sentence-transformers · FastAPI · n8n · Slack

</details>

---

## Cross-Project Tech Stack

| Layer | ChurnGuard | PromptShield | SentinelOps |
|---|---|---|---|
| Core model | XGBoost | DistilBERT / GBM / LogReg | IsolationForest / OneClassSVM / Autoencoder |
| Explainability / grounding | SHAP | Chroma vector pre-filter | pgvector + LlamaIndex RAG |
| Agent framework | LangGraph | CrewAI | LlamaIndex query engine |
| LLM | Groq (Llama-3.3-70B) | Local Ollama (llama3.2) | LLM API (OpenAI-compatible) |
| API layer | FastAPI | FastAPI | FastAPI |
| Orchestration | n8n | n8n | n8n |
| Human approval channel | Slack | Slack | Slack |
| Persistence | Supabase (Postgres) | MongoDB + PostgreSQL | PostgreSQL + pgvector |
| Experiment tracking | MLflow + Optuna | Weights & Biases | MLflow |

---

## Getting Started

Each project is independent — pick the one you want and follow its own README in full:

```
cd churnguard-retention-engine   && cat README.md   # Getting Started section
cd PromptShield                  && cat README.md   # Getting Started section
cd sentinelops-anomaly-rag       && cat README.md   # Getting Started section
```

All three share the same basic shape:
1. `python -m venv venv` + `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and fill in credentials
3. `docker compose up -d` for the project's datastore(s)
4. Run the model/data pipeline scripts
5. Launch the FastAPI service
6. Import the project's n8n workflow JSON and connect Slack
7. Fire a test event and watch it flow end-to-end

---

## Design Principles Behind All Three

- **The model never acts alone.** Every prediction passes through an explanation, classification, or ensemble-agreement step before anything downstream trusts it.
- **The agent never acts alone either.** Every agentic layer pauses at a defined checkpoint — Slack approval, escalation gate, or confidence threshold — before anything consequential happens.
- **Malformed output fails closed, not open.** Across all three systems, a parsing or validation failure escalates to a human; it never silently defaults to "safe" or "allow."
- **Orchestration is visible, not buried in code.** n8n workflows are exported as JSON and versioned alongside the code, so the automation is auditable the same way the code is.
- **Every number in every README is real.** No project in this repo reports a metric it didn't actually measure on an actual run.

---

## License

MIT — see [LICENSE](LICENSE).
