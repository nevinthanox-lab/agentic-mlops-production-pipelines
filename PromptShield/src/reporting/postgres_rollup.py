"""
PromptShield - MongoDB -> Postgres Reporting Rollup
======================================================
This script is the Layer 4 "aggregate rollup" job. It:

    1. Reads every PipelineTrace document from MongoDB's `pipeline_traces`
       collection that has not yet been synced to Postgres.
    2. Inserts (or upserts) each one as a row in Postgres `incident_events`
       (the raw, per-request audit log -- mirrors MongoDB but in a
       relational store suitable for SQL dashboards / Supabase / BI tools).
    3. Recomputes `daily_metrics_rollup` -- the pre-aggregated table used
       for fast dashboarding (counts, category breakdowns, response-time
       SLA compliance) -- for every date touched by this sync run.

DESIGN NOTE ON SLA THRESHOLD:
    SLA_THRESHOLD_MS defines what counts as an "SLA breach" for the
    end-to-end quarantine pipeline latency (classifier + all agent calls).
    Default is 60,000ms (60 seconds). Override via the POSTGRES_SLA_
    THRESHOLD_MS environment variable if your operational SLA differs.

USAGE:
    Run manually after a batch of requests, or on a schedule (cron /
    Windows Task Scheduler / a simple `while True: sleep` loop) to keep
    Postgres continuously in sync with MongoDB:

        python -m src.reporting.postgres_rollup

Place this file at: C:\\projects\\PromptShield\\src\\reporting\\postgres_rollup.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from loguru import logger
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Connection settings (match docker-compose.yml exactly)
# ---------------------------------------------------------------------------

MONGO_URI: str = os.getenv(
    "MONGO_URI",
    "mongodb://root:your_secure_password@localhost:27017/?authSource=admin",
)
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "promptshield")
MONGO_COLLECTION_NAME: str = os.getenv("MONGO_COLLECTION_NAME", "pipeline_traces")

# Postgres runs on host port 5433 per docker-compose.yml port mapping
# ("5433:5432"), so we connect to 5433 from outside the container.
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5433"))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "promptshield")
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "your_secure_password")

# SLA threshold for total pipeline latency, in milliseconds. Override with
# an environment variable if your operational SLA target differs.
SLA_THRESHOLD_MS: float = float(os.getenv("POSTGRES_SLA_THRESHOLD_MS", "60000"))


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def _get_mongo_collection():
    """Connects to MongoDB and returns the pipeline_traces collection."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    logger.info(f"Connected to MongoDB at {MONGO_URI}")
    return client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


def _get_postgres_connection():
    """Connects to Postgres and returns a live connection."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    logger.info(f"Connected to Postgres at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    return conn


# ---------------------------------------------------------------------------
# Step 1: Sync raw incident_events from MongoDB
# ---------------------------------------------------------------------------

def sync_incident_events(mongo_collection, pg_conn) -> "set[str]":
    """
    Reads every PipelineTrace document from MongoDB and upserts it into
    Postgres `incident_events`. Uses ON CONFLICT (prompt_id) DO NOTHING so
    re-running this script is always safe and idempotent -- no duplicate
    rows are ever created, and existing rows are never silently overwritten
    (a trace, once recorded, is treated as an immutable audit fact).

    Returns:
        set[str]: The set of distinct rollup_date strings (YYYY-MM-DD)
        touched by this sync, so the caller knows which dates need their
        daily_metrics_rollup recomputed.
    """
    touched_dates: "set[str]" = set()
    inserted_count = 0
    skipped_count = 0

    with pg_conn.cursor() as cur:
        for trace in mongo_collection.find({}):
            prompt_id = trace.get("prompt_id")
            triage = trace.get("triage") or {}
            escalation = trace.get("escalation") or {}

            attack_category = triage.get("attack_category")
            classifier_risk_tier = triage.get("classifier_risk_tier")
            final_disposition = trace.get("final_disposition")
            pipeline_status = trace.get("pipeline_status")
            triage_disposition = triage.get("disposition")
            judge = trace.get("judge") or {}
            judge_disposition = judge.get("judge_disposition")
            agents_disagreed = bool(escalation.get("agents_disagreed", False))
            escalation_severity = escalation.get("severity")
            notify_slack = bool(escalation.get("notify_slack", False))
            total_latency_ms = trace.get("total_latency_ms")
            sla_breached = (
                total_latency_ms is not None and float(total_latency_ms) > SLA_THRESHOLD_MS
            )
            event_timestamp_raw = trace.get("timestamp")

            # Skip malformed documents rather than crashing the whole sync --
            # a single bad record must never block the entire reporting
            # pipeline. Log it loudly so it can be investigated separately.
            if not (prompt_id and attack_category and classifier_risk_tier and final_disposition):
                logger.warning(f"Skipping malformed trace document: prompt_id={prompt_id}")
                skipped_count += 1
                continue

            try:
                event_timestamp = datetime.fromisoformat(event_timestamp_raw)
            except (TypeError, ValueError):
                event_timestamp = datetime.now(timezone.utc)
                logger.warning(
                    f"Could not parse timestamp for prompt_id={prompt_id}, "
                    "using current time as fallback."
                )

            cur.execute(
                """
                INSERT INTO incident_events (
                    prompt_id, attack_category, classifier_risk_tier,
                    final_disposition, pipeline_status, triage_disposition,
                    judge_disposition, agents_disagreed, escalation_severity,
                    notify_slack, total_latency_ms, sla_breached, event_timestamp
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (prompt_id) DO NOTHING
                RETURNING id;
                """,
                (
                    prompt_id, attack_category, classifier_risk_tier,
                    final_disposition, pipeline_status, triage_disposition,
                    judge_disposition, agents_disagreed, escalation_severity,
                    notify_slack, total_latency_ms, sla_breached, event_timestamp,
                ),
            )

            row = cur.fetchone()
            if row is not None:
                inserted_count += 1
                touched_dates.add(event_timestamp.date().isoformat())
            else:
                skipped_count += 1  # already existed, ON CONFLICT DO NOTHING fired

    pg_conn.commit()
    logger.info(
        f"Sync complete: {inserted_count} new incident_events inserted, "
        f"{skipped_count} skipped (duplicates or malformed)."
    )
    return touched_dates


# ---------------------------------------------------------------------------
# Step 2: Recompute daily_metrics_rollup for touched dates
# ---------------------------------------------------------------------------

def recompute_daily_rollup(pg_conn, touched_dates: "set[str]") -> None:
    """
    Recomputes the daily_metrics_rollup rows for every (date, category,
    risk_tier) combination touched by this sync run, using a single SQL
    aggregate query per date. Uses ON CONFLICT ... DO UPDATE so re-running
    this is always safe and always reflects the current state of
    incident_events for that date (idempotent, not additive).

    Args:
        pg_conn: Live psycopg2 connection.
        touched_dates: Set of ISO date strings (YYYY-MM-DD) to recompute.
    """
    if not touched_dates:
        logger.info("No new dates to recompute in daily_metrics_rollup.")
        return

    with pg_conn.cursor() as cur:
        for date_str in sorted(touched_dates):
            cur.execute(
                """
                INSERT INTO daily_metrics_rollup (
                    rollup_date, attack_category, risk_tier, total_requests,
                    escalated_count, disagreement_count, fail_closed_count,
                    avg_latency_ms, p95_latency_ms, sla_breach_count,
                    sla_breach_rate, updated_at
                )
                SELECT
                    %s::date AS rollup_date,
                    attack_category,
                    classifier_risk_tier AS risk_tier,
                    COUNT(*) AS total_requests,
                    COUNT(*) FILTER (WHERE final_disposition = 'escalate') AS escalated_count,
                    COUNT(*) FILTER (WHERE agents_disagreed) AS disagreement_count,
                    COUNT(*) FILTER (WHERE pipeline_status != 'success') AS fail_closed_count,
                    ROUND(AVG(total_latency_ms), 2) AS avg_latency_ms,
                    ROUND(
                        (PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms))::numeric,
                        2
                    ) AS p95_latency_ms,
                    COUNT(*) FILTER (WHERE sla_breached) AS sla_breach_count,
                    ROUND(
                        COUNT(*) FILTER (WHERE sla_breached)::numeric / NULLIF(COUNT(*), 0),
                        4
                    ) AS sla_breach_rate,
                    now() AS updated_at
                FROM incident_events
                WHERE event_timestamp::date = %s::date
                GROUP BY attack_category, classifier_risk_tier
                ON CONFLICT (rollup_date, attack_category, risk_tier) DO UPDATE SET
                    total_requests     = EXCLUDED.total_requests,
                    escalated_count    = EXCLUDED.escalated_count,
                    disagreement_count = EXCLUDED.disagreement_count,
                    fail_closed_count  = EXCLUDED.fail_closed_count,
                    avg_latency_ms     = EXCLUDED.avg_latency_ms,
                    p95_latency_ms     = EXCLUDED.p95_latency_ms,
                    sla_breach_count   = EXCLUDED.sla_breach_count,
                    sla_breach_rate    = EXCLUDED.sla_breach_rate,
                    updated_at         = EXCLUDED.updated_at;
                """,
                (date_str, date_str),
            )
            logger.info(f"Recomputed daily_metrics_rollup for {date_str}")

    pg_conn.commit()


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def run_rollup_sync() -> None:
    """
    Runs the full MongoDB -> Postgres rollup sync: incident_events first,
    then daily_metrics_rollup recomputation for any dates touched. Safe to
    run repeatedly (idempotent) -- run it after every batch of requests,
    or on a schedule.
    """
    mongo_collection = _get_mongo_collection()
    pg_conn = _get_postgres_connection()

    try:
        touched_dates = sync_incident_events(mongo_collection, pg_conn)
        recompute_daily_rollup(pg_conn, touched_dates)
        logger.info("=== Postgres rollup sync COMPLETE ===")
    finally:
        pg_conn.close()


if __name__ == "__main__":
    run_rollup_sync()
