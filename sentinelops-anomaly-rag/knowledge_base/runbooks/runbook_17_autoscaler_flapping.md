# Runbook: Autoscaler Flapping (Scale Up/Down Oscillation)

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
