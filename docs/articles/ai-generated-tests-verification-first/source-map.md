# 来源映射

- `README.md`：项目定位、verification-first 说明、生成命令和资产层级。
- `ai_playwright/cli/generate_case.py`：`--dry-run` 与 `--no-verify` 已移除，生成强制真实验证。
- `ai_playwright/ai_generation/case_generator.py`：有效断言门、候选目录、真实页面执行、正式写入和二次验证。
- `.github/workflows/ci.yml`：仓库级编译、格式、契约测试、Schema、重复定义、收集和安装检查。
- commits `328222e`、`98d17f7`：语义元素键绑定与已验证 Selector 资产复用。
