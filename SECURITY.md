# Security policy

INFINITO handles personal data by design, so security reports are welcome.

## Reporting a vulnerability

Please report privately through GitHub: **Security** tab, then **Report a
vulnerability**. If that option is not available, contact the maintainer through
their GitHub profile. Do not open a public issue for security problems.

In scope: the core library (`src/infinito3/`) and the lab portal (`lab_portal/`).
The archived code in `legacy/` is not maintained.

## Safety model of the core

- Secrets (passwords, PINs, API keys, private keys) are blocked before memory and
  are never sent to the LLM.
- Emails and phone numbers may be used in the current turn but are not persisted
  by default.
- Retrieved memory reaches the model framed as untrusted reference data, not
  instructions.

The secret filter is pattern-based: it is a guardrail, not a data-loss-prevention
system. Details: [architecture](docs/INFINITO_3_ARCHITECTURE.md#safety-policy).
