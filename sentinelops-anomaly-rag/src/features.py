"""
STEP 3.1 - Feature Engineering via DuckDB
Computes rolling mean/std/z-score, rate-of-change, and cross-metric ratios
directly over the Parquet telemetry file (in-process OLAP, no warehouse).
"""

import duckdb
import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/telemetry.parquet"
OUT_PATH = Path("data/processed/features.parquet")
WINDOW_ROWS = 12  # 12 * 5min = 1 hour rolling window


def build_features() -> pd.DataFrame:
    con = duckdb.connect()

    query = f"""
    WITH base AS (
        SELECT * FROM read_parquet('{RAW_PATH}')
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


def main():
    df = build_features()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Feature matrix: {df.shape[0]:,} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print(f"Feature columns added: {[c for c in df.columns if 'zscore' in c or 'roll' in c or 'ratio' in c or 'change' in c]}")


if __name__ == "__main__":
    main()
