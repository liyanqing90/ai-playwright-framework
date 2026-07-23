# 复现与阅读路径

```bash
git clone https://github.com/liyanqing90/ai-playwright-framework.git
cd ai-playwright-framework
uv sync
uv run ai-playwright-install-browser
cp .env.example .env
uv run gen -p demo saucedemo_ai --headless
```

核心阅读文件：

```text
ai_playwright/cli/generate_case.py
ai_playwright/ai_generation/case_generator.py
.github/workflows/ci.yml
```

仓库级检查：

```bash
uv run black --check .
uv run pytest tests -q
uv run python validate_yaml_schema.py
uv run python check_duplicates.py
uv run pytest --collect-only -q
uv build
```
