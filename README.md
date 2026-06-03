# AI Playwright

[![CI](https://github.com/liyanqing90/ai-playwright-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/liyanqing90/ai-playwright-framework/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

English | [简体中文](README.zh-CN.md)

AI Playwright is a YAML-driven UI automation framework built on top of
Playwright and pytest. It keeps deterministic test assets in version control and
adds optional AI-assisted generation, smart selector recovery, and runtime agent
execution for flows that benefit from natural language intent.

The project is currently in alpha. Public APIs, YAML contracts, and AI runtime
behavior may still evolve, but the repository includes contract tests and CI
gates for the current design.

## Contents

- [Why AI Playwright](#why-ai-playwright)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Usage](#usage)
- [Configuration](#configuration)
- [AI And Data Boundary](#ai-and-data-boundary)
- [Project Layout](#project-layout)
- [Development](#development)
- [Contributing](#contributing)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Why AI Playwright

Most UI automation suites eventually face two competing pressures:

- Stable tests should be deterministic, reviewable, and runnable without a
  model.
- Fast-moving UI work often needs help generating YAML assets, repairing
  selectors, or executing exploratory intent without hard-coding business logic
  into the framework.

AI Playwright separates these concerns:

- `cases/`, `data/`, `elements/`, `modules/`, and `vars/` remain the source of
  truth.
- `strict` mode executes only explicit YAML assets.
- `smart` mode can recover selectors within the current step's intent.
- `agent_case` can execute a natural-language flow at runtime while still going
  through the same StepExecutor command pipeline.
- `gen` turns natural-language specs into formal YAML assets that can be
  reviewed and committed.

## Features

- YAML-first UI automation with pytest dynamic collection.
- Playwright browser automation with Chromium, Firefox, and WebKit support.
- Multi-project and multi-environment configuration.
- Reusable step modules and variable substitution.
- Flow control with conditional branches and loops.
- AI-assisted case generation from project context.
- Smart selector recovery using verified selector evidence.
- Runtime `agent_case` execution for natural-language steps or intent.
- LLM data boundary policies for trusted local and external models.
- Contract tests, schema validation, duplicate checks, formatting, and package
  smoke tests in CI.

## Demo

Click a preview to open the MP4 demo.

| Agent intent execution | AI case generation |
|---|---|
| [<img src="docs/assets/demo/agent-intent.jpg" alt="Agent intent execution demo" width="420">](docs/assets/demo/agent-intent.mp4) | [<img src="docs/assets/demo/agent-case-generation.jpg" alt="AI case generation demo" width="420">](docs/assets/demo/agent-case-generation.mp4) |

## Quick Start

### From Source

```bash
git clone https://github.com/liyanqing90/ai-playwright-framework.git
cd ai-playwright-framework

uv sync
uv run ai-playwright-install-browser
cp .env.example .env

uv run run_case -p demo -f saucedemo_ai --headless
```

### From A Built Wheel Or Local Package

```bash
uv venv
uv pip install .
uv run ai-playwright-install-browser

mkdir my-ai-playwright-workspace
cd my-ai-playwright-workspace
ai-playwright-init
cat > .env <<'EOF'
LLM_DATA_POLICY=external
BASE_URL=https://www.saucedemo.com/
EOF

run_case -p demo -f saucedemo_ai --headless
```

`ai-playwright-init` copies the safe starter `config/` template and the single
canonical `test_data/demo/` demo into the current directory. Source checkouts
already contain these files, so repository development usually does not need to
run it.

## Installation

### Requirements

- Python `>=3.12,<3.15`
- uv for repository development
- Playwright browser binaries

### Install uv

Windows:

```powershell
winget install --id=astral-sh.uv -e
```

If you need a China-friendly PyPI mirror:

```powershell
py -m pip install -U uv -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional `chsrc` helpers:

```bash
chsrc list python
chsrc measure python
chsrc set python tuna
chsrc get python
```

### Development Install

```bash
uv sync
uv run ai-playwright-install-browser
```

### Package Build

```bash
uv build
uv pip install dist/ai_playwright-*.whl
```

The package includes the default config template and the canonical demo
test-data so the CLI can run outside the source tree.

## Core Concepts

### Test Asset Layers

AI Playwright uses a layered YAML structure under `test_data/<project>/`.

| Directory | Purpose | Rule |
|---|---|---|
| `cases/` | Test order and names | Only declares `test_cases[].name` |
| `data/` | Test body | Stores `description`, `mode`, `steps`, or `type: agent_case` |
| `elements/` | Stable element keys | Maps semantic keys to selectors |
| `modules/` | Reusable flows | Stores shared step lists such as login |
| `vars/` | Project variables | Stores reusable values referenced with `${name}` |
| `generation/` | AI generation specs | Input specs for `gen`; not collected by pytest |

Minimal example:

```yaml
# test_data/demo/cases/saucedemo_ai.yaml
test_cases:
  - name: test_saucedemo_backpack_cart
```

```yaml
# test_data/demo/data/saucedemo_ai.yaml
test_data:
  test_saucedemo_backpack_cart:
    description: Standard user adds one product to the cart
    mode: smart
    steps:
      - use_module: saucedemo_login
        params:
          username: ${standard_username}
      - action: click
        selector: add_to_cart_backpack_btn
      - action: assert_text
        selector: shopping_cart_badge
        value: "1"
```

```yaml
# test_data/demo/elements/saucedemo_ai.yaml
elements:
  add_to_cart_backpack_btn: "[data-test='add-to-cart-sauce-labs-backpack']"
  shopping_cart_badge: .shopping_cart_badge
```

### Execution Modes

| Mode or capability | Where it is used | Behavior |
|---|---|---|
| `strict` | Standard YAML steps | Uses explicit selectors only; no model call or selector recovery |
| `smart` | Standard YAML steps | Keeps the action fixed but can recover selector resolution |
| `ai_step` | Single step | Compiles one natural-language instruction into one standard step |
| `agent_case` | Whole case | Executes natural-language steps or intent through runtime planning |
| `gen` | CLI generation | Generates formal YAML assets from a spec file |

Mode priority:

1. Step-level `mode`.
2. Case-level `data.<case>.mode`.
3. CLI `--ai-mode` or `UI_AI_MODE`.
4. `config/ai_config.yaml` `runtime.default_mode`.

`agent_case` is a separate case type and should not declare `mode`.

## Usage

### Run Tests

```bash
# Run all demo cases
uv run run_case -p demo --headless

# Run a specific case file
uv run run_case -p demo -f saucedemo_ai --headless

# Run with a keyword filter
uv run run_case -p demo -f saucedemo_ai -k backpack --headless

# Use smart as the default mode for standard cases
uv run run_case -p demo -f saucedemo_ai --ai-mode smart --headless
```

### Generate Cases

Generation specs live in `test_data/<project>/generation/*.yaml`.

```bash
# Generate, verify on a real browser, then write formal YAML assets
uv run gen -p demo saucedemo_ai

# Run generation verification in headless mode, for CI or remote servers
uv run gen -p demo saucedemo_ai --headless

# Refuse to overwrite existing generated files
uv run gen -p demo saucedemo_ai --no-overwrite
```

Generation is intentionally verification-first. The command writes the generated
payload into a temporary candidate workspace, runs the candidate case on a real
browser, and only writes the formal `cases/`, `data/`, `elements/`, `modules/`,
or `vars/` files after the candidate passes. The formal files are then validated
again. Verification is headed by default so local generation shows a browser;
pass `--headless` when a visible browser is not available. Failed generation
runs keep debugging artifacts under
`logs/generation_runs/`.

Example generation spec:

```yaml
description: "Generate a Saucedemo cart flow"
cases:
  - name: saucedemo_standard_user_cart_logout_flow
    steps:
      - "Open https://www.saucedemo.com/"
      - "Log in as the standard user"
      - "Assert the product page title is Products"
      - "Add Sauce Labs Backpack to the cart"
      - "Assert the cart badge is 1"
      - "Open the cart page"
      - "Assert the cart contains Sauce Labs Backpack"
    final:
      - "The cart contains Sauce Labs Backpack"
```

### Smart Target Step

```yaml
- action: click
  target: "Sauce Labs Backpack add to cart button"
  mode: smart
```

The action remains `click`; only selector resolution can be recovered.

### Agent Case

```yaml
test_data:
  test_saucedemo_agent_case_steps_cart_logout_flow:
    description: Runtime agent executes ordered natural-language steps
    type: agent_case
    steps:
      - "Open https://www.saucedemo.com/"
      - "Log in as the standard user"
      - "Add Sauce Labs Backpack to the cart"
      - "Open the cart page"
      - "Log out from the side menu"
    inputs:
      username: ${standard_username}
      password: ${common_password}
      product_name: Sauce Labs Backpack
    checkpoints:
      - "The cart badge is 1"
      - "The cart contains Sauce Labs Backpack"
    final:
      - "The login button is visible after logout"
```

`agent_case` policy is controlled globally by `agent_policy` in
`config/ai_config.yaml`. Individual cases should describe business intent,
inputs, checkpoints, and final acceptance criteria, not runtime limits or
guardrails.

### Reports And Logs

Runtime artifacts are written to ignored directories:

- `reports/allure-results/` for Allure results.
- `logs/` for execution logs, token usage, and failure model traces.
- `.ui_auto/` for selector registry and AI plan cache.

If the Allure CLI is installed:

```bash
allure serve reports/allure-results
```

## Configuration

### Environment Variables

Create `.env` from `.env.example` for local development.

```env
LLM_BASE_URL=http://localhost:4000/v1
LLM_API_KEY=
LLM_MODEL=
LLM_REASONING_EFFORT=medium
LLM_RESPONSE_FORMAT=auto
LLM_TIMEOUT_SECONDS=60
LLM_DATA_POLICY=external

BASE_URL=https://www.saucedemo.com/
```

Key fields:

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible base URL; `/chat/completions` is appended by the provider |
| `LLM_API_KEY` | Optional API key for the configured model gateway |
| `LLM_MODEL` | Model name or endpoint ID |
| `LLM_RESPONSE_FORMAT` | `auto`, `json_schema`, or `text` depending on model support |
| `LLM_DATA_POLICY` | `external` by default; use `trusted_local` only for trusted local or private models |
| `BASE_URL` | Optional base URL override for the active project |

### Project Config

`config/env_config.yaml` defines projects, environments, and browser defaults.

```yaml
projects:
  demo:
    test_dir: "test_data/demo"
    environments:
      prod: "https://www.saucedemo.com/"
    browser_config:
      viewport:
        width: 1280
        height: 720
```

`config/ai_config.yaml` controls AI runtime behavior, candidate limits, selector
registry preferences, generation context limits, and agent guardrails.

The CLI resolves config in this order:

1. `AI_PLAYWRIGHT_CONFIG_DIR` if set.
2. `./config` in the current working directory.
3. Packaged starter config template.

## AI And Data Boundary

AI Playwright is designed so deterministic tests can run without sending data to
an LLM. Model calls are only used when AI features are enabled and the selected
mode or command needs them.

The default data policy is `external`:

- DOM text sent to the model is compacted and redacted.
- High-risk values such as tokens, API keys, emails, phone numbers, ID-like
  values, and sensitive URL query values are masked.
- Successful runs clean model I/O traces.
- Failed runs may keep redacted model request and response JSON under
  `logs/model_io/<run_id>/` for debugging.

Use `trusted_local` only when the model endpoint is local or otherwise trusted
to receive raw UI text:

```env
LLM_DATA_POLICY=trusted_local
```

Screenshot-to-model resolution is outside this open-source package. The public
runtime focuses on text DOM context, selectors, YAML assets, and
OpenAI-compatible chat-completion providers.

## Project Layout

```text
ai-playwright-framework/
├── ai_playwright/                 # Framework package
│   ├── ai_generation/             # YAML generation and validation harness
│   ├── ai_runtime/                # Smart selector, agent runtime, LLM provider
│   ├── cli/                       # run_case, gen, ai-playwright-init
│   ├── page_objects/              # Playwright page helpers
│   ├── step_actions/              # Standard command executors
│   ├── templates/                 # Packaged config starter files
│   └── utils/                     # Config, YAML, logging, token usage
├── config/                        # Source checkout default config
├── test_data/demo/                # Single canonical open-source demo project
├── tests/                         # Contract and regression tests
├── .github/workflows/ci.yml       # CI gate
├── Makefile                       # Local quality gate
├── pyproject.toml                 # Package metadata and scripts
└── README.md
```

Ignored runtime directories include `logs/`, `reports/`, `evidence/`,
`downloads/`, `.ui_auto/`, `dist/`, and Python cache directories.

## Development

Run the full local gate before opening a pull request:

```bash
make check
```

The gate runs:

- Black formatting check.
- Python compile check.
- Unit and contract tests.
- YAML schema validation.
- Duplicate YAML definition check.
- pytest collection for generated YAML tests.
- uv metadata validation.
- Build and installed-package smoke test from a temporary directory.
- Whitespace validation with `git diff --check`.

Useful commands:

```bash
make format
make test
make schema
make duplicates
make package-check
make clean
```

CI runs the same core checks on Python 3.12, 3.13, and 3.14.

## Contributing

Contributions are welcome. Before submitting a change:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md).
2. Keep framework behavior generic; do not hard-code one site's business flow
   into `ai_playwright/`.
3. Add or update contract tests for YAML contracts, AI runtime behavior,
   selector recovery, or generation changes.
4. Run `make check`.
5. Avoid committing local configs, logs, caches, model traces, screenshots, or
   browser downloads.

For release notes, see [CHANGELOG.md](CHANGELOG.md). For community standards,
see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

Do not commit secrets, private endpoints, customer data, unredacted model
traces, or project-specific internal assets. Use `.env` for local values and keep
`.env.example` generic.

Report security issues through the process described in [SECURITY.md](SECURITY.md).

## Troubleshooting

### `run_case` cannot find project config

Run from a workspace containing `config/env_config.yaml`, or set:

```bash
export AI_PLAYWRIGHT_CONFIG_DIR=/path/to/config
```

For a package-installed workspace, run:

```bash
ai-playwright-init
```

### Playwright browser is missing

Install the browser binary:

```bash
uv run ai-playwright-install-browser
```

For non-uv syncs:

```bash
python -m ai_playwright.cli.install_browser
```

### AI calls fail or return invalid JSON

Check:

- `LLM_BASE_URL` points to an OpenAI-compatible endpoint.
- `LLM_MODEL` is supported by that endpoint.
- `LLM_RESPONSE_FORMAT=auto` for models with uncertain structured-output
  support.
- `LLM_DATA_POLICY=external` unless the model is trusted local/private.

### Generated YAML is not collected

Run:

```bash
uv run python validate_yaml_schema.py
uv run pytest --collect-only -q
```

Generation specs under `generation/` are inputs for `gen`; pytest collects only
`cases/*.yaml` and the referenced `data/*.yaml`.
