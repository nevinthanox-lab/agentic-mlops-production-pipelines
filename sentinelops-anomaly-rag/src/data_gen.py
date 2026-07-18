"""
STEP 1.1 - Synthetic Telemetry Generator
Generates multi-metric time series for N simulated microservices with
deliberately injected anomaly windows (spike / slow-drift / flatline) at
known timestamps. Ground truth labels are what makes this unsupervised
pipeline quantitatively evaluable later.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------
# Configuration
# ----------------------------
RANDOM_SEED = 42
N_SERVICES = 6
SERVICE_NAMES = [
    "checkout-service",
    "auth-service",
    "payments-service",
    "inventory-service",
    "notification-service",
    "search-service",
]
DAYS_OF_DATA = 7
INTERVAL_MINUTES = 5
START_TIME = datetime(2026, 7, 10, 0, 0, 0)

OUTPUT_PATH = Path("data/raw/telemetry.parquet")

np.random.seed(RANDOM_SEED)


def generate_baseline_metrics(n_points: int) -> dict:
    """Generate normal/healthy baseline telemetry for one service."""
    t = np.arange(n_points)

    cpu = 35 + 8 * np.sin(t / 50) + np.random.normal(0, 3, n_points)
    memory = 45 + 5 * np.sin(t / 80 + 1) + np.random.normal(0, 2, n_points)
    latency_p50 = 80 + 10 * np.sin(t / 60) + np.random.normal(0, 5, n_points)
    latency_p95 = latency_p50 * 1.8 + np.random.normal(0, 8, n_points)
    latency_p99 = latency_p50 * 2.5 + np.random.normal(0, 12, n_points)
    error_rate = np.abs(np.random.normal(0.5, 0.2, n_points))
    queue_depth = np.abs(5 + 2 * np.sin(t / 40) + np.random.normal(0, 1.5, n_points))
    thread_count = 50 + 5 * np.sin(t / 70) + np.random.normal(0, 2, n_points)

    return {
        "cpu_pct": np.clip(cpu, 1, 100),
        "memory_pct": np.clip(memory, 1, 100),
        "latency_p50_ms": np.clip(latency_p50, 1, None),
        "latency_p95_ms": np.clip(latency_p95, 1, None),
        "latency_p99_ms": np.clip(latency_p99, 1, None),
        "error_rate_pct": np.clip(error_rate, 0, 100),
        "queue_depth": np.clip(queue_depth, 0, None),
        "thread_count": np.clip(thread_count, 1, None),
    }


def inject_spike(metrics: dict, start_idx: int, duration: int, target_cols: list, magnitude: float = 4.0):
    """Sudden spike anomaly - e.g. latency spike with flat CPU (downstream dependency timeout)."""
    end_idx = min(start_idx + duration, len(metrics["cpu_pct"]))
    for col in target_cols:
        baseline_std = np.std(metrics[col])
        metrics[col][start_idx:end_idx] += magnitude * baseline_std * np.hanning(end_idx - start_idx)
    return start_idx, end_idx


def inject_slow_drift(metrics: dict, start_idx: int, duration: int, target_cols: list, magnitude: float = 3.0):
    """Gradual climbing drift - e.g. memory leak pattern."""
    end_idx = min(start_idx + duration, len(metrics["cpu_pct"]))
    ramp = np.linspace(0, magnitude * np.std(metrics[target_cols[0]]), end_idx - start_idx)
    for col in target_cols:
        metrics[col][start_idx:end_idx] += ramp
    return start_idx, end_idx


def inject_flatline(metrics: dict, start_idx: int, duration: int, target_cols: list):
    """Flatline anomaly - e.g. metric stops reporting/updating (stuck process)."""
    end_idx = min(start_idx + duration, len(metrics["cpu_pct"]))
    for col in target_cols:
        flat_value = metrics[col][start_idx]
        metrics[col][start_idx:end_idx] = flat_value
    return start_idx, end_idx


def build_service_dataframe(service_name: str, n_points: int, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    metrics = generate_baseline_metrics(n_points)

    anomaly_windows = []

    # Deterministic anomaly injection per service so ground truth is reproducible
    rng = np.random.RandomState(hash(service_name) % (2**32))

    n_anomalies = rng.randint(2, 4)
    used_ranges = []

    for _ in range(n_anomalies):
        anomaly_type = rng.choice(["spike", "slow_drift", "flatline"])
        start_idx = rng.randint(200, n_points - 300)
        duration = rng.randint(6, 24)  # 30 min to 2 hours at 5-min intervals

        if anomaly_type == "spike":
            # high latency + flat CPU -> downstream dependency timeout pattern
            s, e = inject_spike(metrics, start_idx, duration,
                                 ["latency_p95_ms", "latency_p99_ms", "queue_depth"], magnitude=5.0)
            desc = "latency spike with flat CPU (downstream dependency timeout pattern)"
        elif anomaly_type == "slow_drift":
            # climbing memory -> leak pattern
            s, e = inject_slow_drift(metrics, start_idx, duration * 3,
                                      ["memory_pct"], magnitude=4.0)
            desc = "steadily climbing memory (leak pattern)"
        else:
            # flatline -> stuck/frozen metric
            s, e = inject_flatline(metrics, start_idx, duration, ["thread_count"])
            desc = "flatlined thread count (stuck process pattern)"

        anomaly_windows.append({
            "service_name": service_name,
            "anomaly_type": anomaly_type,
            "description": desc,
            "start_timestamp": timestamps[s],
            "end_timestamp": timestamps[e - 1],
            "start_idx": s,
            "end_idx": e,
        })

    df = pd.DataFrame(metrics)
    df.insert(0, "timestamp", timestamps)
    df.insert(0, "service_name", service_name)
    df["is_anomaly_ground_truth"] = 0
    for w in anomaly_windows:
        df.loc[w["start_idx"]:w["end_idx"] - 1, "is_anomaly_ground_truth"] = 1

    return df, anomaly_windows


def main():
    n_points = int((DAYS_OF_DATA * 24 * 60) / INTERVAL_MINUTES)
    timestamps = pd.date_range(start=START_TIME, periods=n_points, freq=f"{INTERVAL_MINUTES}min")

    all_dfs = []
    all_anomaly_windows = []

    for service in SERVICE_NAMES:
        df, windows = build_service_dataframe(service, n_points, timestamps)
        all_dfs.append(df)
        all_anomaly_windows.extend(windows)

    full_df = pd.concat(all_dfs, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(OUTPUT_PATH, index=False)

    # Save ground truth anomaly window log separately (used for evaluation in Phase 4)
    gt_df = pd.DataFrame(all_anomaly_windows)
    gt_path = Path("data/raw/ground_truth_windows.parquet")
    gt_df.to_parquet(gt_path, index=False)

    print(f"Generated {len(full_df):,} rows across {N_SERVICES} services -> {OUTPUT_PATH}")
    print(f"Injected {len(gt_df)} ground-truth anomaly windows -> {gt_path}")
    print(gt_df[["service_name", "anomaly_type", "start_timestamp", "end_timestamp"]].to_string(index=False))


if __name__ == "__main__":
    main()
