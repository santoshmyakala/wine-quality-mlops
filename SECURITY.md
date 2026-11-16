# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub: **Security → Report a vulnerability** on this repository. Please don't open a public issue.

You can expect an acknowledgement within a few days. Fixes are released on `main`.

## Supply-chain practices in this repository

- Third-party GitHub Actions are pinned to full commit SHAs; Dependabot proposes updates.
- Every push and pull request is scanned for committed secrets with gitleaks (full history).
- No credentials are stored in the repository; CI uses short-lived `GITHUB_TOKEN` / OIDC.
