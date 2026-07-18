# Runbook: Upstream Rate-Limit Errors

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
