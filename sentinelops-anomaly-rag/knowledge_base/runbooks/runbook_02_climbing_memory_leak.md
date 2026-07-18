# Runbook: Steadily Climbing Memory (Leak Pattern)

## Symptom Pattern
Memory percentage climbs in a near-linear ramp over minutes to hours, without
returning to baseline between request cycles. CPU and latency often stay
normal until memory pressure triggers GC thrashing or OOM.

## Likely Root Cause
A memory leak - objects, connections, or caches are being allocated but never
released. Common causes: unclosed DB/file/socket handles, unbounded in-memory
caches, or event listener accumulation.

## Diagnostic Steps (read-only)
1. Pull a heap snapshot / memory profile if the runtime supports it.
2. Compare object counts across two snapshots taken 15+ minutes apart.
3. Check for recently deployed code touching caching, connection pooling, or
   listener/subscription logic.
4. Correlate the ramp start time with the most recent deploy timestamp.

## Remediation
- Identify and patch the leaking code path (unclosed resource, unbounded
  cache growth).
- Add a cache eviction policy or TTL if none exists.
- Roll back the suspect deploy if the ramp start aligns with a release.

## REQUIRES HUMAN APPROVAL
- Restarting the affected service/pod to reclaim memory immediately.
- Rolling back a production deployment.
