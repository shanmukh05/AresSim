# Security Policy

## Supported versions

AresSim is pre-1.0 research software. Security fixes land on `main`.

## Reporting a vulnerability

Do not open a public issue for security reports.

Use GitHub's **Privately report a vulnerability** flow on this repository, or email the maintainers if that feature is unavailable.

Please include:

- A description of the issue and its impact
- Steps to reproduce, or a proof of concept
- Affected versions or commit hashes if you know them

You should receive an acknowledgement within a few days. We will work on a fix and coordinate disclosure after a patch is available.

## Scope notes

AresSim is designed as a **local** simulator and training stack:

- The FastAPI server binds to `127.0.0.1` by default and is not a hardened multi-user service.
- Do not deploy the API or UI to the public internet without authentication, TLS, and a threat model.
- Never commit secrets (`.env`, API tokens, W&B keys, credentials). Training extras talk to W&B only when you configure them.
