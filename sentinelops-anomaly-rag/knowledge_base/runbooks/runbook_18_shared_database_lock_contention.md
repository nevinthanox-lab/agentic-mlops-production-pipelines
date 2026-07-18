# Runbook: Shared Database Lock Contention

## Symptom Pattern
Latency rises specifically on write/update operations while read
operations remain fast; queue depth grows on write paths.

## Likely Root Cause
Long-running transactions or missing indexes are causing row/table lock
contention, serializing writes that should otherwise be concurrent.

## Diagnostic Steps (read-only)
1. Check the database's active lock/blocking-session view.
2. Identify the longest-running transaction holding locks.
3. Check for missing indexes on frequently-updated columns.

## Remediation
- Kill or shorten the offending long-running transaction (coordinate with
  DBA).
- Add missing indexes to reduce lock scope/duration.

## REQUIRES HUMAN APPROVAL
- Killing an active database transaction/session.
- Adding a production database index (can lock table during creation).
