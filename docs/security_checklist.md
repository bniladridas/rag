# Security Checklist

Scope: This repository provides a local CLI/TUI assistant that fetches public data and runs local models. Use this checklist before releasing changes or deploying in shared environments.

## Core Risks
- Expression evaluation: calculator input is user-provided; avoid `eval` or untrusted execution.
- Network calls: external APIs can fail, hang, or return unexpected content.
- Model loading: failures can degrade behavior; users should see clear warnings.
- File paths: environment variables can point outside the project and overwrite data.
- Dependencies: model and NLP libraries may have known vulnerabilities.

## Pre-Release Checklist
- Verify calculator input parsing uses a safe parser (no `eval`).
- Confirm HTTP requests have timeouts and bounded retries.
- Ensure missing model warnings are user-visible in CLI and TUI.
- Confirm project paths are constrained to the repo root.
- Run dependency vulnerability scans (e.g., `pip-audit`).
- Document any data collection or telemetry.

## Minimal Threat Model
- Local user input can be malicious or malformed.
- External APIs can be unreliable or return unexpected payloads.
- Environment variables can be manipulated to alter file targets.
- Local model downloads can fail or be replaced if dependency integrity is compromised.

## Ownership
- This checklist should be reviewed for every release and after dependency updates.
