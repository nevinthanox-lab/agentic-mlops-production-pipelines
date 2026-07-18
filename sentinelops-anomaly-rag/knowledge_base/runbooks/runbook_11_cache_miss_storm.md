# Runbook: Cache Miss Storm

## Symptom Pattern
Latency and downstream DB load both spike together, often after a cache
restart, cache eviction event, or TTL-aligned mass expiration.

## Likely Root Cause
A large portion of the cache expired or was invalidated simultaneously
(e.g. all keys sharing the same TTL), forcing a flood of requests to hit
the database directly.

## Diagnostic Steps (read-only)
1. Check cache hit-rate metric for a sudden drop.
2. Check cache eviction/restart events around the same timestamp.
3. Check whether cache keys share a synchronized TTL (thundering herd
   risk).

## Remediation
- Add TTL jitter so keys don't expire simultaneously.
- Implement cache warming after restarts.
- Add request coalescing for cache-miss stampedes.

## REQUIRES HUMAN APPROVAL
- Deploying cache configuration/TTL-jitter changes.
