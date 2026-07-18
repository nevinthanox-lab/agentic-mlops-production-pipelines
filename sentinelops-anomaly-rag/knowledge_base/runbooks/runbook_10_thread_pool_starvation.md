# Runbook: Thread Pool Starvation

## Symptom Pattern
Latency increases sharply under moderate load; thread count metric may
flatline at the configured max pool size while queue depth grows.

## Likely Root Cause
All worker threads are blocked on a slow operation (often a slow downstream
call or lock contention), so no threads remain to process new requests even
though CPU is idle.

## Diagnostic Steps (read-only)
1. Capture a thread dump - look for many threads blocked on the same
   downstream call or lock.
2. Check thread pool size configuration vs. actual concurrent demand.
3. Check for recently introduced blocking calls on a previously async path.

## Remediation
- Identify and fix the blocking operation (make it async or add a timeout).
- Increase thread pool size as a short-term mitigation.

## REQUIRES HUMAN APPROVAL
- Changing production thread pool configuration.
- Deploying a code fix for the blocking call.
