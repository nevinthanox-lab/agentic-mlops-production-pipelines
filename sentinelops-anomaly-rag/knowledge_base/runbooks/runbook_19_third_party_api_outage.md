# Runbook: Third-Party API Outage

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
