# Contributing

Thanks for improving AI Playwright. Keep contributions small, verifiable, and focused on reusable framework behavior rather than project-specific test data.

## Development

```bash
uv sync
uv run ai-playwright-install-browser
make check
```

Use `make format` before sending a pull request.

## Pull Requests

- Include a clear problem statement and the behavioral change.
- Add or update tests for framework behavior.
- Keep demo data generic and safe to publish.
- Do not commit real credentials, customer data, private endpoints, cookies, screenshots, reports, or model I/O logs.

## Test Data Policy

Only `test_data/demo/**` should be committed. Real project data belongs outside the repository or in ignored paths.
