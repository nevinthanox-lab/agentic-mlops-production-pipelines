# Runbook: Intermittent Timeout Bursts (Not Sustained)

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
