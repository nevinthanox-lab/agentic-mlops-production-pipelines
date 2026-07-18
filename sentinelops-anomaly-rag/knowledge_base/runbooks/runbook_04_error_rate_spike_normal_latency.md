# Runbook: Error-Rate Spike With Normal Latency (Bad Deploy)

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
