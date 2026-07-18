# Runbook: GC Pause-Induced Latency Sawtooth

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
