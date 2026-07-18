# Runbook: Disk I/O Saturation

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
