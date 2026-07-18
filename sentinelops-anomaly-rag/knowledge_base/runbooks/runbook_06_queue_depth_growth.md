# Runbook: Request Queue Depth Growing Without Bound

## Symptom Pattern
Queue depth climbs steadily while throughput (requests completed per
second) stays flat or declines - a classic sign the service is falling
behind incoming demand.

## Likely Root Cause
Either the consumer side is under-provisioned (too few worker
threads/pods) relative to arrival rate, or a downstream slowdown is
reducing effective throughput while arrivals continue at the same rate.

## Diagnostic Steps (read-only)
1. Compare arrival rate vs. completion rate over the window.
2. Check worker/thread pool utilization - is it saturated?
3. Check downstream call latency for signs of a slowdown reducing
   throughput.

## Remediation
- Scale out consumer capacity (more workers/replicas).
- If downstream-caused, follow the "High Latency With Flat CPU" runbook.

## REQUIRES HUMAN APPROVAL
- Manual scale-out beyond auto-scaler configured maximum.
