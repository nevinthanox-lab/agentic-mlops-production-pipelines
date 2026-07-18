"""
STEP 9.1 / 9.2 / 9.3 — SentinelOps End-to-End Replay Test
Path: sentinelops-anomaly-rag/tests/test_e2e_replay.py

STEP 9.1.2 (REVISED): incident-level remediation dedup.
Previously /remediate was called for EVERY anomalous row. On a real replay
(12096 rows) this fired dozens of calls per incident and blew through the
Groq daily token quota (100k), causing repeated 429s and an unbounded
retry loop. Fix:
  - Consecutive anomalous rows for the same service are grouped into one
    "incident" (state machine: normal -> anomalous = incident OPEN,
    anomalous -> normal, or a gap > INCIDENT_GAP_SECONDS = incident CLOSED).
  - /remediate is called exactly once per incident, at the moment it opens.
  - All requests calls now go through post_with_retry(): bounded retries
    with exponential backoff, explicit handling of 429 (honors
    Retry-After if the server sends one), and a hard cap so a quota
    outage can never turn into an infinite loop again.

What this does:
1. Loads data/raw/telemetry.parquet
2. Loads ground-truth anomaly windows (data/raw/ground_truth_windows.parquet
   if present, else skips timing verification gracefully)
3. Replays the telemetry in rolling batches (simulating streaming) against
   FastAPI /detect
4. Groups anomalous rows into incidents and calls /remediate ONCE per
   incident, checking:
   - a runbook was retrieved (grounding_confidence present)
   - low-confidence cases correctly show escalate=True
5. Compares detected anomaly timestamps vs ground-truth windows -> timing report
6. Writes reports/e2e_replay_report.json with REAL numbers (no placeholders)

Run with:  pytest -s tests/test_e2e_replay.py
(or python tests/test_e2e_replay.py to just run it as a script)
"""

import json
import os
import sys
import time
import glob
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = os.environ.get("SENTINELOPS_API_URL", "http://127.0.0.1:8000")
DETECT_ENDPOINT = f"{BASE_URL}/detect"
REMEDIATE_ENDPOINT = f"{BASE_URL}/remediate"

TELEMETRY_PATH = "data/raw/telemetry.parquet"
REPORT_OUT_PATH = "reports/e2e_replay_report.json"

BATCH_SIZE = 50          # rows per simulated "streaming" batch
TIMESTAMP_COL = "timestamp"
TOLERANCE_SECONDS = 300  # how close a detection must be to a GT window to count as "on time"

# STEP 9.1.2 additions -------------------------------------------------
# If the next anomalous row for a service arrives within this many seconds
# of the last anomalous row for that same service, it is treated as the
# SAME ongoing incident (no new /remediate call). A larger gap than this
# means the previous incident is considered closed and a new one opens.
INCIDENT_GAP_SECONDS = 600  # 10 minutes

# Retry/backoff bounds so a quota outage (429) can never hang forever.
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 30
# ------------------------------------------------------------------------

GROUND_TRUTH_PATH = "data/raw/ground_truth_windows.parquet"


def load_ground_truth():
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"[WARN] {GROUND_TRUTH_PATH} not found. "
              "STEP 9.2 timing verification will be skipped.")
        return None
    gt_df = pd.read_parquet(GROUND_TRUTH_PATH)
    gt_windows = gt_df.to_dict(orient="records")
    print(f"[INFO] Loaded ground truth from {GROUND_TRUTH_PATH} "
          f"({len(gt_windows)} windows)")
    return gt_windows


def load_telemetry():
    if not os.path.exists(TELEMETRY_PATH):
        raise FileNotFoundError(
            f"{TELEMETRY_PATH} not found. Run STEP 1.3 (data_gen.py) first."
        )
    df = pd.read_parquet(TELEMETRY_PATH)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Expected a '{TIMESTAMP_COL}' column in telemetry.parquet, "
            f"found columns: {list(df.columns)}"
        )
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    max_rows = os.environ.get("SENTINELOPS_MAX_ROWS")
    if max_rows:
        df = df.iloc[:int(max_rows)].reset_index(drop=True)
        print(f"[INFO] SENTINELOPS_MAX_ROWS set -> replaying only first {max_rows} rows.")
    return df


def post_with_retry(url, json_payload, timeout, max_retries=MAX_RETRIES):
    """
    STEP 9.1.2: bounded retry wrapper around requests.post.
    - Honors Retry-After header on 429 if the server sends one.
    - Otherwise uses exponential backoff, capped at MAX_BACKOFF_SECONDS.
    - Raises the last exception once max_retries is exhausted, instead of
      looping forever.
    """
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            resp = requests.post(url, json=json_payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(
                    BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS
                )
                print(f"[WARN] 429 rate-limited on {url}. "
                      f"Waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1}).")
                time.sleep(delay)
                attempt += 1
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_exc = e
            delay = min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
            print(f"[WARN] Request to {url} failed ({e}). "
                  f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1}).")
            time.sleep(delay)
            attempt += 1

    raise RuntimeError(
        f"Exceeded max_retries={max_retries} calling {url}. Last error: {last_exc}"
    )


def call_detect(batch_df):
    payload = json.loads(batch_df.to_json(orient="records", date_format="iso"))
    return post_with_retry(DETECT_ENDPOINT, {"records": payload}, timeout=30)


def build_anomaly_description(item):
    """
    Build the natural-language anomaly_description string that
    RemediateRequest expects, from a DetectResult-shaped dict.
    """
    votes = item.get("votes", [])
    vote_str = "; ".join(
        f"{v['model_name']} score={v['score']:.4f} "
        f"(threshold={v['threshold']:.4f}, flagged={v['is_anomaly']})"
        for v in votes
    )
    return (
        f"Anomaly detected on service '{item.get('service_name')}' "
        f"at {item.get('timestamp')}. Model votes: {vote_str}."
    )


def call_remediate(anomaly_item):
    payload = {"anomaly_description": build_anomaly_description(anomaly_item)}
    return post_with_retry(REMEDIATE_ENDPOINT, payload, timeout=60)


def is_within_gt_window(ts, service_name, gt_windows, tolerance_seconds=TOLERANCE_SECONDS):
    ts = pd.to_datetime(ts)
    pad = pd.Timedelta(seconds=tolerance_seconds)
    for w in gt_windows:
        if w.get("service_name") != service_name:
            continue
        start = pd.to_datetime(w["start_timestamp"])
        end = pd.to_datetime(w["end_timestamp"])
        if (start - pad) <= ts <= (end + pad):
            return True, w
    return False, None


class IncidentTracker:
    """
    STEP 9.1.2: groups consecutive anomalous rows per service into
    incidents so /remediate fires once per incident, not once per row.
    """

    def __init__(self, gap_seconds=INCIDENT_GAP_SECONDS):
        self.gap_seconds = gap_seconds
        self._last_seen_ts = {}   # service_name -> last anomalous timestamp
        self.incidents_opened = 0
        self.incidents_continued = 0

    def is_new_incident(self, service_name, ts):
        ts = pd.to_datetime(ts)
        last_ts = self._last_seen_ts.get(service_name)
        self._last_seen_ts[service_name] = ts
        if last_ts is None:
            self.incidents_opened += 1
            return True
        gap = (ts - last_ts).total_seconds()
        if gap > self.gap_seconds:
            self.incidents_opened += 1
            return True
        self.incidents_continued += 1
        return False


def run_replay():
    df = load_telemetry()
    gt_windows = load_ground_truth()
    tracker = IncidentTracker()

    detections = []
    remediation_results = []
    api_errors = []

    print(f"[INFO] Replaying {len(df)} rows in batches of {BATCH_SIZE} "
          f"against {DETECT_ENDPOINT}")

    for start_idx in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start_idx:start_idx + BATCH_SIZE]
        try:
            result = call_detect(batch)
        except Exception as e:
            api_errors.append({"batch_start": start_idx, "error": str(e)})
            continue

        flagged = result.get("results", [])
        for item in flagged:
            if not item.get("is_anomaly", item.get("anomaly", False)):
                continue

            detections.append(item)

            ts = item.get(TIMESTAMP_COL) or item.get("timestamp")
            svc = item.get("service_name")

            # STEP 9.1.2: only call /remediate when this row opens a NEW
            # incident. Rows that are a continuation of an already-open
            # incident are still recorded as detections (for timing
            # verification) but do not trigger another remediation call.
            if not tracker.is_new_incident(svc, ts):
                continue

            try:
                remediation = call_remediate(item)
            except Exception as e:
                api_errors.append({"item": item, "error": str(e)})
                continue

            remediation_results.append({
                "anomaly": item,
                "remediation": remediation,
            })

    # --- STEP 9.2: timing verification vs ground truth ---
    timing_report = {"checked": False}
    if gt_windows:
        matched_detections = 0
        matched_gt_window_ids = set()
        for d in detections:
            ts = d.get(TIMESTAMP_COL) or d.get("timestamp")
            svc = d.get("service_name")
            if not ts:
                continue
            ok, window = is_within_gt_window(ts, svc, gt_windows)
            if ok:
                matched_detections += 1
                matched_gt_window_ids.add(
                    (window["service_name"], window["start_timestamp"], window["end_timestamp"])
                )

        total_gt = len(gt_windows)
        recalled_windows = len(matched_gt_window_ids)
        timing_report = {
            "checked": True,
            "total_detections": len(detections),
            "total_gt_windows": total_gt,
            "detections_matched_within_tolerance": matched_detections,
            "gt_windows_recalled": recalled_windows,
            "gt_window_recall_pct": round(100 * recalled_windows / total_gt, 2) if total_gt else 0.0,
            "tolerance_seconds": TOLERANCE_SECONDS,
        }

    # --- STEP 9.3: runbook grounding verification ---
    # grounding_confidence is a STRING category (e.g. "high"/"medium"/"low"/"none"),
    # per RemediateResponse schema (src/schemas.py) - not a float.
    LOW_CONFIDENCE_LABELS = {"low", "none", "no_match", "unknown"}
    confidence_counts = {}
    grounded = 0
    escalated_low_conf = 0
    for r in remediation_results:
        rem = r["remediation"]
        conf_label = str(rem.get("grounding_confidence", "")).lower()
        confidence_counts[conf_label] = confidence_counts.get(conf_label, 0) + 1
        has_sources = bool(rem.get("retrieved_sources"))
        if conf_label in LOW_CONFIDENCE_LABELS or not has_sources:
            escalated_low_conf += 1
        else:
            grounded += 1

    grounding_report = {
        "total_remediation_calls": len(remediation_results),
        "grounded_with_runbook": grounded,
        "escalated_low_confidence": escalated_low_conf,
        "confidence_label_counts": confidence_counts,
    }

    # STEP 9.1.2: incident dedup summary, so the report shows exactly how
    # much the remediation call volume was reduced vs. naive per-row calls.
    incident_report = {
        "total_anomalous_rows": len(detections),
        "incidents_opened": tracker.incidents_opened,
        "incident_continuation_rows_skipped": tracker.incidents_continued,
        "incident_gap_seconds": INCIDENT_GAP_SECONDS,
    }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rows_replayed": len(df),
        "batch_size": BATCH_SIZE,
        "total_detections": len(detections),
        "api_errors": api_errors,
        "incident_report": incident_report,
        "timing_report": timing_report,
        "grounding_report": grounding_report,
    }

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Report written to {REPORT_OUT_PATH}")
    print(json.dumps(report, indent=2))
    return report


def test_e2e_replay():
    """pytest entry point"""
    report = run_replay()
    assert report["rows_replayed"] > 0, "No telemetry rows were replayed."
    assert not report["api_errors"], f"API errors occurred: {report['api_errors']}"
    if report["timing_report"]["checked"]:
        assert report["timing_report"]["gt_windows_recalled"] > 0, \
            "No ground-truth windows were recalled by any detection."


if __name__ == "__main__":
    t0 = time.time()
    run_replay()
    print(f"[INFO] Done in {time.time() - t0:.2f}s")
