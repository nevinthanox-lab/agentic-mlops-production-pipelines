-- ============================================================================
-- PromptShield - Postgres Reporting Schema
-- ============================================================================
-- This schema implements the Layer 4 requirement:
--   "aggregate incident metrics (counts, category breakdowns, response-time
--    SLAs) roll up into a Postgres/Supabase reporting table for dashboarding"
--
-- Two tables:
--   1. incident_events        - one row per PipelineTrace from MongoDB (raw feed)
--   2. daily_metrics_rollup   - pre-aggregated daily rollup for fast dashboarding
--
-- Run this once to initialize the schema:
--   docker exec -i promptshield-postgres psql -U postgres -d promptshield < postgres_schema.sql
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table 1: incident_events (raw, append-only log mirrored from MongoDB)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incident_events (
    id                      SERIAL PRIMARY KEY,
    prompt_id               VARCHAR(64)  NOT NULL UNIQUE,
    attack_category          VARCHAR(32)  NOT NULL,
    classifier_risk_tier    VARCHAR(16)  NOT NULL,
    final_disposition       VARCHAR(16)  NOT NULL,
    pipeline_status         VARCHAR(24)  NOT NULL,
    triage_disposition      VARCHAR(16),
    judge_disposition       VARCHAR(16),
    agents_disagreed        BOOLEAN      NOT NULL DEFAULT FALSE,
    escalation_severity     VARCHAR(16),
    notify_slack            BOOLEAN      NOT NULL DEFAULT FALSE,
    total_latency_ms        NUMERIC(12, 2),
    sla_breached            BOOLEAN      NOT NULL DEFAULT FALSE,
    event_timestamp         TIMESTAMPTZ  NOT NULL,
    ingested_at             TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Indexes to make the rollup aggregation queries and dashboard filters fast.
CREATE INDEX IF NOT EXISTS idx_incident_events_event_timestamp
    ON incident_events (event_timestamp);
CREATE INDEX IF NOT EXISTS idx_incident_events_category_tier
    ON incident_events (attack_category, classifier_risk_tier);
CREATE INDEX IF NOT EXISTS idx_incident_events_disposition
    ON incident_events (final_disposition);

-- ----------------------------------------------------------------------------
-- Table 2: daily_metrics_rollup (pre-aggregated, one row per date+category+tier)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_metrics_rollup (
    id                      SERIAL PRIMARY KEY,
    rollup_date             DATE         NOT NULL,
    attack_category          VARCHAR(32)  NOT NULL,
    risk_tier               VARCHAR(16)  NOT NULL,
    total_requests          INTEGER      NOT NULL DEFAULT 0,
    escalated_count         INTEGER      NOT NULL DEFAULT 0,
    disagreement_count      INTEGER      NOT NULL DEFAULT 0,
    fail_closed_count       INTEGER      NOT NULL DEFAULT 0,
    avg_latency_ms          NUMERIC(12, 2),
    p95_latency_ms          NUMERIC(12, 2),
    sla_breach_count        INTEGER      NOT NULL DEFAULT 0,
    sla_breach_rate         NUMERIC(5, 4),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (rollup_date, attack_category, risk_tier)
);

CREATE INDEX IF NOT EXISTS idx_daily_rollup_date
    ON daily_metrics_rollup (rollup_date);

-- ----------------------------------------------------------------------------
-- Convenience view: today's live snapshot across all categories/tiers,
-- useful for a quick dashboard widget without waiting for the daily rollup job.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW today_summary AS
SELECT
    attack_category,
    classifier_risk_tier AS risk_tier,
    COUNT(*)                                            AS total_requests,
    COUNT(*) FILTER (WHERE final_disposition = 'escalate') AS escalated_count,
    COUNT(*) FILTER (WHERE agents_disagreed)              AS disagreement_count,
    COUNT(*) FILTER (WHERE pipeline_status != 'success')  AS fail_closed_count,
    ROUND(AVG(total_latency_ms), 2)                      AS avg_latency_ms,
    COUNT(*) FILTER (WHERE sla_breached)                  AS sla_breach_count
FROM incident_events
WHERE event_timestamp::date = CURRENT_DATE
GROUP BY attack_category, classifier_risk_tier;
