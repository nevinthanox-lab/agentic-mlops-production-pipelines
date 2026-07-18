# Runbook: High Latency With Flat CPU (Downstream Dependency Timeout)

## Symptom Pattern
p95/p99 latency rises sharply (2-5x baseline) while CPU utilization stays flat
or near-baseline. Request queue depth typically climbs in parallel.

## Likely Root Cause
The service itself is not compute-bound. A downstream dependency (database,
cache, third-party API, or internal microservice) is responding slowly or
timing out, causing request threads to block while waiting rather than
consuming CPU.

## Diagnostic Steps (read-only)
1. Check distributed trace spans for the affected service - identify which
   downstream call has the longest duration.
2. Inspect downstream dependency's own latency/error dashboards.
3. Check connection pool metrics - exhausted connection pools cause exactly
   this flat-CPU-high-latency signature.
4. Review recent deploys to the downstream dependency in the same time window.

## Remediation
- If a downstream DB is slow: check for missing indexes or lock contention
  on the DB side (read-only query, do not modify schema without DBA review).
- If connection pool exhausted: temporarily increase pool size via config
  (requires deploy or feature-flag toggle).
- If a specific downstream is down: enable circuit breaker / fallback path
  if available.

## REQUIRES HUMAN APPROVAL
- Restarting the downstream service.
- Scaling up connection pool limits in production config.
- Failing over to a backup dependency instance.
