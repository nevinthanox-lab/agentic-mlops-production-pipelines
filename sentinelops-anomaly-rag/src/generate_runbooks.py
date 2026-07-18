"""
STEP 5.1 (continued) - Generates the remaining SRE runbook markdown files.
Run once to populate knowledge_base/runbooks/ with 20 total runbooks.
"""

from pathlib import Path

OUT_DIR = Path("knowledge_base/runbooks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNBOOKS = {
    "runbook_02_climbing_memory_leak.md": """# Runbook: Steadily Climbing Memory (Leak Pattern)

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
""",
    "runbook_03_flatlined_metric_stuck_process.md": """# Runbook: Flatlined Metric (Stuck Process Pattern)

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
""",
    "runbook_04_error_rate_spike_normal_latency.md": """# Runbook: Error-Rate Spike With Normal Latency (Bad Deploy)

## Symptom Pattern
Error rate jumps sharply while latency percentiles remain within normal
range. Errors are often uniform across request types rather than isolated
to one endpoint.

## Likely Root Cause
A bad deployment introduced a logic bug, misconfiguration, or breaking
schema change. Because the failure is fast (validation error, null pointer,
config mismatch) rather than slow (timeout), latency stays normal.

## Diagnostic Steps (read-only)
1. Check deploy timeline - does the error spike start align with a recent
   rollout?
2. Sample recent error logs/stack traces for a common signature.
3. Check for recent config or feature-flag changes alongside the deploy.
4. Compare error rate across canary vs. stable instances if canary exists.

## Remediation
- If tied to a specific deploy: roll back to the previous known-good version.
- If tied to a config/flag change: revert the flag.

## REQUIRES HUMAN APPROVAL
- Rolling back a production deployment.
- Reverting a feature flag serving live traffic.
""",
    "runbook_05_cpu_spike_with_latency.md": """# Runbook: CPU Spike Correlated With Latency Increase

## Symptom Pattern
CPU utilization and latency rise together, proportionally. Queue depth also
tends to grow as the service becomes compute-bound.

## Likely Root Cause
Genuine compute-bound load - either a legitimate traffic spike, an
inefficient code path (e.g. an accidental O(n^2) loop, unindexed in-memory
scan), or a noisy-neighbor process consuming shared CPU.

## Diagnostic Steps (read-only)
1. Check request volume/RPS for the same window - is this proportional
   traffic growth or a fixed traffic level with rising CPU?
2. Profile CPU flame graph if available to isolate the hot function.
3. Check for recent deploys introducing an expensive code path.
4. Check node-level CPU (noisy neighbor on shared infrastructure).

## Remediation
- If legitimate traffic growth: scale out horizontally.
- If an inefficient code path: patch and redeploy.
- If noisy neighbor: relocate or isolate the workload.

## REQUIRES HUMAN APPROVAL
- Horizontal auto-scaling beyond configured limits.
- Redeploying a hotfix to production.
""",
    "runbook_06_queue_depth_growth.md": """# Runbook: Request Queue Depth Growing Without Bound

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
""",
    "runbook_07_intermittent_timeout_bursts.md": """# Runbook: Intermittent Timeout Bursts (Not Sustained)

## Symptom Pattern
Short bursts (a few minutes) of elevated timeouts/errors that self-resolve,
recurring periodically rather than being a single sustained incident.

## Likely Root Cause
Commonly a periodic batch job, cron task, or garbage-collection pause
competing for shared resources (DB connections, CPU, network) at a
predictable interval.

## Diagnostic Steps (read-only)
1. Check timestamps of the bursts for periodicity (e.g. every 15 min).
2. Cross-reference with scheduled jobs/cron logs running on the same
   infrastructure.
3. Check GC pause logs if the runtime has stop-the-world GC.

## Remediation
- Reschedule the competing batch job to a lower-traffic window.
- Tune GC settings if GC pauses are the cause.

## REQUIRES HUMAN APPROVAL
- Changing production cron schedules.
- Modifying JVM/runtime GC configuration.
""",
    "runbook_08_connection_pool_exhaustion.md": """# Runbook: Connection Pool Exhaustion

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
""",
    "runbook_09_disk_io_saturation.md": """# Runbook: Disk I/O Saturation

## Symptom Pattern
Latency rises on operations touching disk (writes, log flushes, local
cache reads) while CPU/memory stay normal. Often paired with elevated
queue depth on write-heavy paths.

## Likely Root Cause
Disk throughput or IOPS limit reached - could be excessive logging,
uncompacted local storage, or a co-located noisy-neighbor workload on
shared disk.

## Diagnostic Steps (read-only)
1. Check disk I/O utilization and IOPS metrics for the host/volume.
2. Check log verbosity - is debug logging accidentally enabled in prod?
3. Check for large uncompacted files (logs, local DB files) needing
   rotation/compaction.

## Remediation
- Reduce log verbosity if debug logging is on.
- Trigger compaction/rotation of local storage.
- Move to higher-throughput disk tier if consistently saturated.

## REQUIRES HUMAN APPROVAL
- Changing production log levels.
- Migrating to a different storage tier.
""",
    "runbook_10_thread_pool_starvation.md": """# Runbook: Thread Pool Starvation

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
""",
    "runbook_11_cache_miss_storm.md": """# Runbook: Cache Miss Storm

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
""",
    "runbook_12_upstream_rate_limit_errors.md": """# Runbook: Upstream Rate-Limit Errors

## Symptom Pattern
Error rate rises with a distinct pattern of 429/rate-limit-type errors,
latency largely unaffected, often correlated with a traffic burst.

## Likely Root Cause
The service (or a shared API key/quota) has exceeded a rate limit imposed
by an upstream third-party API or internal service.

## Diagnostic Steps (read-only)
1. Confirm error codes/messages match rate-limit signatures.
2. Check current request rate against the known upstream limit.
3. Check whether multiple services share the same API key/quota.

## Remediation
- Implement or tighten client-side rate limiting / backoff.
- Request a quota increase from the upstream provider if load is
  legitimate.
- Separate shared quotas per service if contention is the cause.

## REQUIRES HUMAN APPROVAL
- Requesting a quota increase (external dependency, cost implication).
- Changing shared credential/quota allocation.
""",
    "runbook_13_dns_resolution_delays.md": """# Runbook: DNS Resolution Delays

## Symptom Pattern
Latency increases specifically on the connection-establishment phase of
outbound calls, often intermittent and affecting multiple unrelated
downstream targets simultaneously.

## Likely Root Cause
DNS resolver slowness or failure - could be a local resolver cache issue,
an overloaded internal DNS server, or an expired/misconfigured DNS TTL.

## Diagnostic Steps (read-only)
1. Check if the latency increase affects multiple different downstream
   hosts at the same time (points to DNS, not one specific dependency).
2. Check local DNS resolver logs/metrics if available.
3. Test resolution time directly against the configured resolver.

## Remediation
- Increase local DNS cache TTL / enable local caching resolver.
- Failover to a secondary DNS resolver.

## REQUIRES HUMAN APPROVAL
- Changing DNS resolver configuration in production.
""",
    "runbook_14_certificate_expiry_warning.md": """# Runbook: TLS Certificate Nearing Expiry / Handshake Errors

## Symptom Pattern
Error rate rises specifically on TLS handshake, often with explicit
certificate-related error messages; typically not correlated with load.

## Likely Root Cause
A TLS certificate used by the service or a downstream dependency has
expired or is about to expire, or there is a certificate chain/trust store
mismatch after a rotation.

## Diagnostic Steps (read-only)
1. Check certificate expiry date for the affected endpoint.
2. Check recent certificate rotation events and trust store updates.
3. Confirm intermediate certificate chain is complete.

## Remediation
- Renew/rotate the certificate before expiry (should be automated;
  flag if automation failed).
- Fix trust store / chain configuration if a rotation broke it.

## REQUIRES HUMAN APPROVAL
- Manually rotating a production TLS certificate.
""",
    "runbook_15_gc_pause_latency_sawtooth.md": """# Runbook: GC Pause-Induced Latency Sawtooth

## Symptom Pattern
Latency shows a repeating sawtooth pattern - periods of normal latency
followed by brief sharp spikes, recurring at a roughly fixed interval, with
memory showing a corresponding sawtooth (climb then drop).

## Likely Root Cause
Garbage collection stop-the-world pauses in a managed runtime (JVM, .NET,
etc.), often worsened by insufficient heap size causing frequent
full-GC cycles.

## Diagnostic Steps (read-only)
1. Check GC logs for pause frequency and duration.
2. Correlate GC pause timestamps with latency spike timestamps.
3. Check current heap size vs. working set size.

## Remediation
- Tune GC algorithm/parameters for the workload.
- Increase heap size if consistently undersized.
- Reduce allocation rate in hot code paths if identified.

## REQUIRES HUMAN APPROVAL
- Changing production JVM/runtime memory or GC flags.
""",
    "runbook_16_dependency_version_mismatch.md": """# Runbook: Dependency Version Mismatch After Partial Rollout

## Symptom Pattern
Errors appear only for a subset of requests/instances, inconsistent
behavior across replicas of the same service, often after a rolling
deployment is interrupted or partially completed.

## Likely Root Cause
Some instances are running an old version while others run a new version,
and the two versions are incompatible in API contract, schema, or shared
cache format.

## Diagnostic Steps (read-only)
1. Check deployed version/build hash across all replicas.
2. Check if errors correlate with specific instance IDs.
3. Check rollout status - is the deployment stuck mid-rollout?

## Remediation
- Complete or roll back the deployment fully (avoid a split-version state).
- Ensure backward/forward compatibility for any in-flight rollout.

## REQUIRES HUMAN APPROVAL
- Forcing completion or rollback of an in-progress deployment.
""",
    "runbook_17_autoscaler_flapping.md": """# Runbook: Autoscaler Flapping (Scale Up/Down Oscillation)

## Symptom Pattern
Thread count / instance count metric oscillates rapidly, with correlated
brief latency spikes each time capacity drops before scaling back up.

## Likely Root Cause
Autoscaler thresholds are too tight relative to traffic variance, causing
it to scale down during brief lulls and immediately scale back up,
degrading performance during each transition.

## Diagnostic Steps (read-only)
1. Check autoscaler scale-up/scale-down event history and frequency.
2. Check traffic variance over the same window.
3. Check cooldown period configuration on the autoscaler.

## Remediation
- Widen the autoscaler thresholds or increase cooldown period.
- Set a higher minimum replica count if traffic is consistently bursty.

## REQUIRES HUMAN APPROVAL
- Changing production autoscaler configuration.
""",
    "runbook_18_shared_database_lock_contention.md": """# Runbook: Shared Database Lock Contention

## Symptom Pattern
Latency rises specifically on write/update operations while read
operations remain fast; queue depth grows on write paths.

## Likely Root Cause
Long-running transactions or missing indexes are causing row/table lock
contention, serializing writes that should otherwise be concurrent.

## Diagnostic Steps (read-only)
1. Check the database's active lock/blocking-session view.
2. Identify the longest-running transaction holding locks.
3. Check for missing indexes on frequently-updated columns.

## Remediation
- Kill or shorten the offending long-running transaction (coordinate with
  DBA).
- Add missing indexes to reduce lock scope/duration.

## REQUIRES HUMAN APPROVAL
- Killing an active database transaction/session.
- Adding a production database index (can lock table during creation).
""",
    "runbook_19_third_party_api_outage.md": """# Runbook: Third-Party API Outage

## Symptom Pattern
Error rate spikes sharply and uniformly for all calls to one specific
external dependency, while all other metrics for the service itself
(CPU, memory, internal latency) remain normal.

## Likely Root Cause
The third-party provider itself is experiencing an outage or degraded
service, external to anything under this team's control.

## Diagnostic Steps (read-only)
1. Check the provider's public status page.
2. Confirm the failure is isolated to that one dependency and not a
   broader network issue.
3. Check retry/circuit-breaker metrics for that dependency.

## Remediation
- Enable circuit breaker to fail fast and protect upstream capacity.
- Switch to a cached/degraded-mode response if available.
- Monitor provider status page for resolution; no fix possible on our side.

## REQUIRES HUMAN APPROVAL
- Enabling degraded-mode / feature-flag fallback in production.
""",
    "runbook_20_no_clear_pattern_low_confidence.md": """# Runbook: No Clear Pattern / Ambiguous Signal

## Symptom Pattern
Multiple metrics show mild, non-correlated deviation without a clear
dominant driver; the anomaly does not cleanly match spike, drift, or
flatline signatures described in other runbooks.

## Likely Root Cause
Could be measurement noise, an emerging issue too early to characterize,
or a novel failure mode not yet covered by existing runbooks.

## Diagnostic Steps (read-only)
1. Widen the observation window to see if a clearer pattern emerges over
   the next 15-30 minutes.
2. Check for any correlated deploys, config changes, or external events.
3. Compare against historical baseline for the same time-of-day/day-of-week.

## Remediation
No confident automated remediation should be generated for this pattern.
Escalate to a human SRE for manual investigation.

## REQUIRES HUMAN APPROVAL
- Any action taken here should be human-initiated, since the pattern is
  not confidently diagnosed by this runbook.
""",
}

for filename, content in RUNBOOKS.items():
    (OUT_DIR / filename).write_text(content, encoding="utf-8")

print(f"Wrote {len(RUNBOOKS)} additional runbooks to {OUT_DIR}/ (total 20 with runbook_01)")
