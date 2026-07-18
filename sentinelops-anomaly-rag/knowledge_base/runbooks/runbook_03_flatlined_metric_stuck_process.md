# Runbook: Flatlined Metric (Stuck Process Pattern)

## Symptom Pattern
A metric (commonly thread count, active connections, or request rate) stops
updating and holds a constant value for an extended period, even as related
metrics continue to fluctuate normally.

## Likely Root Cause
The reporting/collection path for that metric is stuck - a deadlocked
thread pool, a frozen event loop, or a crashed exporter/agent that isn't
restarting.

## Diagnostic Steps (read-only)
1. Check process/pod liveness - is the process still running and responsive
   to health checks?
2. Inspect thread dump for deadlocks or threads stuck in the same call stack.
3. Check the metrics exporter/agent logs for crash or connection errors.
4. Verify the metrics pipeline (scraper, aggregator) isn't the one that's
   actually broken.

## Remediation
- If the exporter/agent crashed: restart just the exporter, not the service.
- If it's a genuine application deadlock: capture a thread dump before any
  restart for postmortem analysis.

## REQUIRES HUMAN APPROVAL
- Restarting the application process/pod.
- Force-killing deadlocked threads.
