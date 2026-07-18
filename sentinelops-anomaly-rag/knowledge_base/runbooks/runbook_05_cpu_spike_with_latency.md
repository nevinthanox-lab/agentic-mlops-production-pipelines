# Runbook: CPU Spike Correlated With Latency Increase

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
