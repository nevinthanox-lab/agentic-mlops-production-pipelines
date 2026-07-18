# Runbook: Connection Pool Exhaustion

## Symptom Pattern
Latency spikes and error rate rises together, often with explicit
"connection pool exhausted" or "timeout waiting for connection" errors in
logs.

## Likely Root Cause
Either genuine load exceeds configured pool size, or connections are being
leaked (not returned to the pool after use), gradually starving the pool.

## Diagnostic Steps (read-only)
1. Check pool utilization metrics over time - gradual climb suggests a
   leak; sudden saturation suggests a load spike.
2. Search logs for connection-related exceptions and their source.
3. Check for recent code changes touching connection lifecycle management.

## Remediation
- If a leak: identify and fix the code path not releasing connections.
- If genuine load: increase pool size within DB's max-connection limits.

## REQUIRES HUMAN APPROVAL
- Increasing production connection pool size.
- Restarting the service to clear leaked connections immediately.
