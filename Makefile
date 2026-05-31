PYTHON ?= poetry run python
PYTEST ?= poetry run pytest
BLACK ?= poetry run black

.PHONY: format format-check compile test schema duplicates collect poetry-check build package-check check clean

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

poetry-check:
	poetry check --lock

build:
	poetry build

package-check: build
	set -e; \
	venv="$$(poetry env info --path)"; \
	"$$venv/bin/python" -m ensurepip --upgrade; \
	tmpdir="$$(mktemp -d)"; \
	pkgdir="$$tmpdir/pkg"; \
	workdir="$$tmpdir/work"; \
	mkdir -p "$$pkgdir" "$$workdir"; \
	"$$venv/bin/python" -m pip install --target "$$pkgdir" --no-deps dist/*.whl; \
	cd "$$workdir"; \
	PKGDIR="$$pkgdir" PYTHONPATH="$$pkgdir" "$$venv/bin/python" -c 'import importlib.metadata as md, os, pathlib, ai_playwright; pkgdir = pathlib.Path(os.environ["PKGDIR"]).resolve(); package_file = pathlib.Path(ai_playwright.__file__).resolve(); assert package_file.is_relative_to(pkgdir), package_file; scripts = {ep.name for ep in md.entry_points(group="console_scripts") if ep.value.startswith("ai_playwright.")}; assert {"run_case", "gen", "ai-playwright-init"} <= scripts, scripts'; \
	PYTHONPATH="$$pkgdir" "$$venv/bin/python" -m ai_playwright.cli.run_case --help; \
	PYTHONPATH="$$pkgdir" "$$venv/bin/python" -m ai_playwright.cli.generate_case --help; \
	PYTHONPATH="$$pkgdir" "$$venv/bin/python" -m ai_playwright.cli.init_project --help; \
	set +e; \
	PYTHONPATH="$$pkgdir" "$$venv/bin/python" -m ai_playwright.cli.run_case -p demo -f saucedemo_ai --headless -k __never_matches__; \
	status="$$?"; \
	set -e; \
	test "$$status" -eq 5

check: format-check compile test schema duplicates collect poetry-check package-check
	git diff --check

clean:
	rm -rf dist build *.egg-info logs reports evidence downloads .ui_auto .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
