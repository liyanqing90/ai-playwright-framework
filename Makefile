PYTHON ?= uv run python
PYTEST ?= uv run pytest
BLACK ?= uv run black

.PHONY: format format-check compile test schema duplicates collect lock-check build package-check check clean

format:
	$(BLACK) ai_playwright tests conftest.py check_duplicates.py validate_yaml_schema.py

format-check:
	$(BLACK) --check .

compile:
	$(PYTHON) -m compileall -q ai_playwright conftest.py check_duplicates.py validate_yaml_schema.py

test:
	$(PYTEST) tests -q

schema:
	$(PYTHON) validate_yaml_schema.py

duplicates:
	$(PYTHON) check_duplicates.py

collect:
	$(PYTEST) --collect-only -q

lock-check:
	uv lock --locked

build:
	uv build

package-check: build
	set -e; \
	tmpdir="$$(mktemp -d)"; \
	pkgdir="$$tmpdir/pkg"; \
	workdir="$$tmpdir/work"; \
	repo_python="$$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	mkdir -p "$$pkgdir" "$$workdir"; \
	uv pip install --target "$$pkgdir" --no-deps dist/*.whl; \
	cd "$$workdir"; \
	PKGDIR="$$pkgdir" PYTHONPATH="$$pkgdir" "$$repo_python" -c 'import importlib.metadata as md, os, pathlib, ai_playwright; pkgdir = pathlib.Path(os.environ["PKGDIR"]).resolve(); package_file = pathlib.Path(ai_playwright.__file__).resolve(); assert package_file.is_relative_to(pkgdir), package_file; scripts = {ep.name for ep in md.entry_points(group="console_scripts") if ep.value.startswith("ai_playwright.")}; assert {"run_case", "gen", "ai-playwright-init", "ai-playwright-install-browser"} <= scripts, scripts'; \
	test -f "$$pkgdir/ai_playwright/templates/test_data/demo/cases/saucedemo_ai.yaml"; \
	PYTHONPATH="$$pkgdir" "$$repo_python" -m ai_playwright.cli.run_case --help; \
	PYTHONPATH="$$pkgdir" "$$repo_python" -m ai_playwright.cli.generate_case --help; \
	PYTHONPATH="$$pkgdir" "$$repo_python" -m ai_playwright.cli.init_project --help; \
	PYTHONPATH="$$pkgdir" "$$repo_python" -m ai_playwright.cli.install_browser --help; \
	set +e; \
	PYTHONPATH="$$pkgdir" "$$repo_python" -m ai_playwright.cli.run_case -p demo -f saucedemo_ai --headless -k __never_matches__; \
	status="$$?"; \
	set -e; \
	test "$$status" -eq 5

check: format-check compile test schema duplicates collect lock-check package-check
	git diff --check

clean:
	rm -rf dist build *.egg-info logs reports evidence downloads .ui_auto .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
