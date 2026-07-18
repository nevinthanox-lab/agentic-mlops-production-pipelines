# Runbook: DNS Resolution Delays

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
