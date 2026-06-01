# Security Policy

## Reporting

Please report security issues privately through GitHub security advisories for this repository. Do not open public issues for vulnerabilities or leaked data.

## Data Handling

The default LLM data policy is `external`, which redacts high-risk UI text before sending model context or writing model I/O logs. Use `trusted_local` only for private models that are allowed to receive raw UI text.

Never commit `.env`, cookies, storage state, reports, evidence, screenshots, downloads, `.ui_auto`, or model I/O logs.
