# Runbook: TLS Certificate Nearing Expiry / Handshake Errors

## Symptom Pattern
Error rate rises specifically on TLS handshake, often with explicit
certificate-related error messages; typically not correlated with load.

## Likely Root Cause
A TLS certificate used by the service or a downstream dependency has
expired or is about to expire, or there is a certificate chain/trust store
mismatch after a rotation.

## Diagnostic Steps (read-only)
1. Check certificate expiry date for the affected endpoint.
2. Check recent certificate rotation events and trust store updates.
3. Confirm intermediate certificate chain is complete.

## Remediation
- Renew/rotate the certificate before expiry (should be automated;
  flag if automation failed).
- Fix trust store / chain configuration if a rotation broke it.

## REQUIRES HUMAN APPROVAL
- Manually rotating a production TLS certificate.
