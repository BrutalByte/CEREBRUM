# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.1.x (current) | Yes — active security support |
| 1.0.x | Yes — critical fixes only |
| 0.4.x and below | No |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

To report a vulnerability, email: **bryan.alexander@buchorn.com**

Include in your report:
- Description of the vulnerability and its impact
- Steps to reproduce or proof-of-concept code
- Affected versions
- Any suggested mitigations you have identified

You will receive a response within **48 hours** acknowledging receipt. After triage, you will receive a follow-up within **7 days** with our assessment and expected timeline.

We follow coordinated disclosure: please allow us 90 days to develop and release a fix before public disclosure.

## Security Architecture

### Authentication

CEREBRUM's API server uses **JWT Bearer token authentication** on all endpoints.

```
Authorization: Bearer <jwt_token>
```

Tokens are validated using HMAC-SHA256 with a configurable secret key set via the `CEREBRUM_JWT_SECRET` environment variable. Tokens carry:
- `sub`: subject (user/service identifier)
- `exp`: expiration timestamp (default: 1 hour)
- `scope`: optional permission scope (`read`, `write`, `admin`)

Anonymous access is disabled by default. The `CEREBRUM_ALLOW_ANONYMOUS` environment variable can be set to `true` for development deployments only.

### Path Provenance (HMAC-SHA256)

Every reasoning path returned by the query engine carries a cryptographic signature:

```json
{
    "answer": "Marie Curie",
    "path": ["Physics", "Nobel_Prize_1903", "Marie_Curie"],
    "hmac": "sha256:a3f4b2...",
    "signed_at": 1743000000
}
```

The HMAC is computed over the canonical JSON serialization of the path using the server's `CEREBRUM_HMAC_KEY`. This allows downstream consumers to verify that paths have not been tampered with between the reasoning engine and the consuming application.

### Input Validation

**Query inputs**: Entity strings are sanitized to remove control characters and length-capped at 512 characters. Relation type strings are validated against an allowlist when `strict_relations=True`.

**Graph ingestion**: The `IngestionPipeline` applies entity normalization (whitespace stripping, Unicode normalization) and confidence clamping to `[0.0, 1.0]`. Malformed edges are rejected at ingest, not silently dropped.

**Namespace inputs**: Namespace prefixes are validated to contain only alphanumeric characters, underscores, and hyphens. Colon characters are reserved as the namespace separator and are rejected in prefix inputs.

### Adversarial Hardening

**Causal Flood Protection**: The `STDPDiscretizer` includes two guards against adversarial spike injection:
- `min_causal_span` (seconds): blocks burst attacks by requiring co-occurrences to span a minimum wall-clock duration
- `use_chi_squared`: statistical uniformity test that detects and rejects non-organic spike distributions

Deployments receiving untrusted event streams should enable both guards:
```python
STDPDiscretizer(min_causal_span=1.0, use_chi_squared=True)
```

**Resource Governance**: The `ResourceGovernor` enforces per-query CPU and memory limits. Runaway traversals (pathological beam widths, deeply recursive graphs) are terminated before they can impact service availability.

### Network Security

- The API server should be deployed behind a reverse proxy (nginx, Traefik) that enforces TLS 1.2+.
- The SSE streaming endpoints (`/stream/*`) should be rate-limited at the proxy layer.
- Neo4j and remote adapter credentials should be provided via environment variables, never hardcoded.

### Data Privacy

- CEREBRUM does not log query content by default. Set `CEREBRUM_AUDIT_LOG=true` to enable query logging for compliance environments (logs include entity names and path results — treat as sensitive).
- The Holographic Index (Bloom filter) stores only probabilistic membership sketches — no entity content is transmitted during federated discovery.
- The `SignalEncoder` stores alignment matrices (rotation matrices), not raw sensor data. Raw signals are processed in memory and not persisted.

## Known Limitations

- JWT token revocation is not currently supported. Tokens are valid until their expiration timestamp. Deploy with short token lifetimes (≤1 hour) in high-security environments.
- The Neo4j adapter does not currently support Kerberos authentication. It uses username/password credentials.
- Graph data at rest is not encrypted by default. Use filesystem-level encryption or an encrypted database backend for sensitive data.

## Security Contacts

Primary: **Bryan Alexander Buchorn** — bryan.alexander@buchorn.com

---
**Copyright © 2026 Bryan Alexander Buchorn (AMP). All Rights Reserved.**
