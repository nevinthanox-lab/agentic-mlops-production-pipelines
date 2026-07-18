# Runbook: Dependency Version Mismatch After Partial Rollout

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
