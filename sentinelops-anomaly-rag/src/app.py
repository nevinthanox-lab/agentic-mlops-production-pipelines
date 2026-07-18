"""
sentinelops/src/app.py
FastAPI service exposing /detect and /remediate.
STEP 7.2 + STEP 7.3
Wired directly to the real artifacts produced by src/train_models.py
(models/scaler.joblib, models/isolation_forest.joblib, models/one_class_svm.joblib,
models/autoencoder.keras, models/thresholds.json) and to src/rag_pipeline.py
(get_query_engine_components, generate_remediation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import keras

from schemas import (
    DetectRequest, DetectResponse, DetectResult, ModelVote,
    RemediateRequest, RemediateResponse,
)
from rag_pipeline import get_query_engine_components, generate_remediation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinelops")

app = FastAPI(
    title="SentinelOps API",
    description="Telemetry anomaly detection + RAG-grounded remediation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
WINDOW_ROWS = 12  # must match src/features.py

# same order as FEATURE_COLS in src/train_models.py
FEATURE_COLS = [
    "cpu_pct", "memory_pct", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "error_rate_pct", "queue_depth", "thread_count",
    "cpu_zscore", "mem_zscore", "latency_zscore", "error_zscore", "queue_zscore", "thread_zscore",
    "cpu_rate_of_change", "latency_rate_of_change", "memory_rate_of_change",
    "latency_per_cpu_ratio", "queue_per_thread_ratio",
]

# lazy-loaded singletons
_scaler = _iso_model = _svm_model = _ae_model = _thresholds = None
_retriever = None


def get_models():
    global _scaler, _iso_model, _svm_model, _ae_model, _thresholds
    if _scaler is None:
        logger.info("Loading models from %s ...", MODEL_DIR)
        _scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        _iso_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
        _svm_model = joblib.load(MODEL_DIR / "one_class_svm.joblib")
        _ae_model = keras.models.load_model(MODEL_DIR / "autoencoder.keras")
        _thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text())
    return _scaler, _iso_model, _svm_model, _ae_model, _thresholds


def get_retriever():
    global _retriever
    if _retriever is None:
        logger.info("Initializing RAG retriever...")
        _retriever = get_query_engine_components()
    return _retriever


def build_features_from_records(records: list[dict]) -> pd.DataFrame:
    """Same rolling-window SQL as src/features.py, but over the posted batch.
    Writes the batch to a temp Parquet file first and reads it back with
    read_parquet() - avoids a DuckDB/pandas string-dtype registration bug."""
    import tempfile
    raw_df = pd.DataFrame(records)

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "batch.parquet"
        raw_df.to_parquet(tmp_path, index=False)
        tmp_path_sql = tmp_path.as_posix()

        con = duckdb.connect()
        query = f"""
    WITH base AS (
        SELECT * FROM read_parquet('{tmp_path_sql}')
    ),
    windowed AS (
        SELECT
            *,
            AVG(cpu_pct) OVER w AS cpu_roll_mean,
            STDDEV(cpu_pct) OVER w AS cpu_roll_std,
            AVG(memory_pct) OVER w AS mem_roll_mean,
            STDDEV(memory_pct) OVER w AS mem_roll_std,
            AVG(latency_p95_ms) OVER w AS lat_p95_roll_mean,
            STDDEV(latency_p95_ms) OVER w AS lat_p95_roll_std,
            AVG(error_rate_pct) OVER w AS err_roll_mean,
            STDDEV(error_rate_pct) OVER w AS err_roll_std,
            AVG(queue_depth) OVER w AS queue_roll_mean,
            STDDEV(queue_depth) OVER w AS queue_roll_std,
            AVG(thread_count) OVER w AS thread_roll_mean,
            STDDEV(thread_count) OVER w AS thread_roll_std,
            LAG(cpu_pct, 1) OVER (PARTITION BY service_name ORDER BY timestamp) AS cpu_prev,
            LAG(latency_p95_ms, 1) OVER (PARTITION BY service_name ORDER BY timestamp) AS lat_prev,
            LAG(memory_pct, 1) OVER (PARTITION BY service_name ORDER BY timestamp) AS mem_prev
        FROM base
        WINDOW w AS (
            PARTITION BY service_name ORDER BY timestamp
            ROWS BETWEEN {WINDOW_ROWS} PRECEDING AND CURRENT ROW
        )
    )
    SELECT
        *,
        (cpu_pct - cpu_roll_mean) / NULLIF(cpu_roll_std, 0) AS cpu_zscore,
        (memory_pct - mem_roll_mean) / NULLIF(mem_roll_std, 0) AS mem_zscore,
        (latency_p95_ms - lat_p95_roll_mean) / NULLIF(lat_p95_roll_std, 0) AS latency_zscore,
        (error_rate_pct - err_roll_mean) / NULLIF(err_roll_std, 0) AS error_zscore,
        (queue_depth - queue_roll_mean) / NULLIF(queue_roll_std, 0) AS queue_zscore,
        (thread_count - thread_roll_mean) / NULLIF(thread_roll_std, 0) AS thread_zscore,
        (cpu_pct - cpu_prev) AS cpu_rate_of_change,
        (latency_p95_ms - lat_prev) AS latency_rate_of_change,
        (memory_pct - mem_prev) AS memory_rate_of_change,
        latency_p95_ms / NULLIF(cpu_pct, 0) AS latency_per_cpu_ratio,
        queue_depth / NULLIF(thread_count, 0) AS queue_per_thread_ratio
    FROM windowed
    ORDER BY service_name, timestamp
    """
        df = con.execute(query).df()

    df = df.fillna(0)
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- STEP 7.2
@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    scaler, iso_model, svm_model, ae_model, thresholds = get_models()

    records = [r.model_dump() for r in payload.records]
    feat_df = build_features_from_records(records)

    X = feat_df[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)

    # STEP 3.2/3.3/3.4 scoring - same sign convention as train_models.py
    iso_scores = -iso_model.score_samples(X_scaled)
    svm_scores = -svm_model.decision_function(X_scaled)
    recon = ae_model.predict(X_scaled, verbose=0)
    ae_rmse = np.sqrt(np.mean((X_scaled - recon) ** 2, axis=1))

    iso_thresh = thresholds["isolation_forest"]
    svm_thresh = thresholds["one_class_svm"]
    ae_thresh = thresholds["autoencoder"]

    results: list[DetectResult] = []
    anomalies_found = 0

    for i, row in feat_df.iterrows():
        iso_vote = bool(iso_scores[i] >= iso_thresh)
        svm_vote = bool(svm_scores[i] >= svm_thresh)
        ae_vote = bool(ae_rmse[i] >= ae_thresh)

        # STEP 4.3 ensemble rule: IsolationForest-score + Autoencoder-RMSE agreement
        is_anomaly = iso_vote and ae_vote
        if is_anomaly:
            anomalies_found += 1

        votes = [
            ModelVote(model_name="isolation_forest", score=float(iso_scores[i]), threshold=iso_thresh, is_anomaly=iso_vote),
            ModelVote(model_name="one_class_svm", score=float(svm_scores[i]), threshold=svm_thresh, is_anomaly=svm_vote),
            ModelVote(model_name="autoencoder", score=float(ae_rmse[i]), threshold=ae_thresh, is_anomaly=ae_vote),
        ]

        results.append(DetectResult(
            service_name=row["service_name"],
            timestamp=row["timestamp"],
            is_anomaly=is_anomaly,
            votes=votes,
        ))

    return DetectResponse(results=results, anomalies_found=anomalies_found)


# ---------------------------------------------------------------- STEP 7.3
@app.post("/remediate", response_model=RemediateResponse)
def remediate(payload: RemediateRequest):
    retriever = get_retriever()
    try:
        result = generate_remediation(payload.anomaly_description, retriever)
    except Exception as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=502, detail=f"RAG pipeline error: {exc}")

    return RemediateResponse(**result)
