# 之家 UI 自动化测试框架

之家 UI 自动化测试框架是一个基于 Playwright 的 UI 自动化测试框架，采用模块化设计，支持 YAML
格式的测试用例定义，提供灵活、高效、可维护的自动化测试解决方案。

## 目录

- [框架特性](#框架特性)
- [安装与配置](#安装与配置)
- [使用指南](#使用指南)
- [AI 用例生成与智能执行](#ai-用例生成与智能执行)
- [架构概述](#架构概述)
- [项目结构](#项目结构)
- [测试开发指南](#测试开发指南)
- [高级功能](#高级功能)
- [最佳实践](#最佳实践)
- [常用命令](#常用命令)
- [元素定位技巧](#元素定位技巧)
- [常见问题](#常见问题)
- [相关资源](#相关资源)

## 框架特性

### 核心功能

- **多浏览器支持**：基于Playwright实现，支持Chrome、Firefox、Safari等主流浏览器
- **YAML用例编写方式**：支持YAML方式编写测试用例
- **丰富的UI操作**：内置40+种常用UI操作，覆盖各种交互场景
- **多项目管理**：支持多项目、多环境配置，便于管理大型测试套件
- **Page Object模式**：完整的Page Object模式实现，提高代码复用性
- **详细日志与报告**：详细的操作日志、失败截图和Allure测试报告
- **AI用例生成**：支持基于 YAML 规格生成当前项目三层结构用例，并优先复用既有元素、变量和公共组件
- **AI/Smart执行体系**：原生支持 strict、smart、agent_case steps 和 agent_case intent/explore；用例生成统一走 `gen`

### 高级特性

- **数据驱动能力**：
    - 多级变量管理：支持全局变量、测试用例级变量和临时变量
    - 表达式计算：支持条件判断、数学运算等
    - 模块化组件：可复用的测试步骤片段
    - 流程控制：支持if-then-else条件分支和for_each循环结构
    - AI辅助生成：支持自然语言规格生成用例，并通过 Harness 校验生成结果

- **性能优化功能**：
    - 浏览器资源池：减少浏览器启动开销，支持复用和健康检查
    - 智能截图策略：仅在测试失败时截图，支持压缩和区域截图
    - 日志轮转机制：自动归档和清理日志，减少存储占用

## 架构概述

之家 UI 自动化测试框架采用模块化设计，由以下核心组件构成：

### 1. 测试运行器 (Runner)

测试运行器负责加载和执行测试用例，管理测试环境和浏览器实例，收集测试结果并生成报告。

**主要文件**:

- `src/cli/run_case.py`: 推荐测试执行入口，对应 `poetry run run_case`
- `src/cli/generate_case.py`: 推荐用例生成入口，对应 `poetry run gen`
- `test_runner.py`: 根命令入口；日常推荐使用 Poetry scripts
- `src/runner.py`: 核心运行器实现，负责测试用例的加载和执行

### 2. 测试用例执行器 (TestCaseExecutor)

测试用例执行器负责解析和执行单个测试用例，管理测试步骤的执行流程和错误处理。

**主要文件**:

- `src/test_case_executor.py`: 测试用例执行器实现

### 3. 测试步骤执行器 (StepExecutor)

测试步骤执行器负责执行单个测试步骤，包括 UI 操作、断言、变量管理等。

**主要文件**:

- `src/test_step_executor.py`: 测试步骤执行器入口
- `src/step_actions/step_executor.py`: 测试步骤执行器核心实现
- `src/step_actions/action_types.py`: 操作类型定义
- `src/step_actions/commands/`: 命令模式实现目录

### 4. 页面对象 (Page Objects)

页面对象封装了与页面交互的方法，提供了更高级别的 API，使测试代码更加清晰和易于维护。

**主要文件**:

- `page_objects/base_page.py`: 基础页面类，提供通用的页面操作方法
- `page_objects/[specific_page].py`: 特定页面的实现

### 5. 工具类 (Utils)

工具类提供了各种辅助功能，如变量管理、配置管理、日志记录等。

**主要文件**:

- `utils/variable_manager.py`: 变量管理器
- `utils/config.py`: 配置管理
- `utils/logger.py`: 日志管理
- `utils/yaml_handler.py`: YAML 文件处理

## 安装与配置

### 环境要求

1. Python 3.12+
2. Poetry包管理工具（[安装指南](https://python-poetry.org/docs/#installation)）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/zhijia_ui.git
   cd zhijia_ui
   ```

2. **安装依赖**
   ```bash
   poetry install
   ```

3. **安装浏览器驱动**
   ```bash
   playwright install chromium
   # 或安装所有浏览器
   # playwright install
   ```

## 使用指南

### 运行测试

1. **运行指定项目的测试**
   ```bash
   poetry run run_case -p demo
   ```

2. **运行指定测试文件**
   ```bash
   poetry run run_case -p demo -f test_cases
   ```

3. **有头模式运行**
   ```bash
   poetry run run_case -p demo --headed
   ```

4. **使用 AI/Smart 执行模式运行**
   ```bash
   poetry run run_case -p demo -f saucedemo_ai -m smart
   ```

   `-m/--ai-mode` 只作为默认执行模式使用。若 `data` 层用例或单个步骤已经声明 `mode`，则优先使用 YAML 中的配置。

### 生成用例

1. **基于项目自然语言规格调用模型生成**
   ```bash
   poetry run gen -p demo saucedemo_ai
   ```

2. **只预览生成结果，不写入文件**
   ```bash
   poetry run gen -p demo saucedemo_ai --dry-run
   ```

生成规格放在 `test_data/<project>/generation/` 下；`poetry run gen -p demo saucedemo_ai` 会读取 `test_data/demo/generation/saucedemo_ai.yaml`，默认输出并覆盖 `test_data/demo/cases/saucedemo_ai.yaml` 和 `test_data/demo/data/saucedemo_ai.yaml`，并只在模型判断必须新增资产时写入 `elements/`、`modules/`、`vars/`。

### 生成报告

```bash
allure serve reports/allure-results
```

### 工具命令

1. **检查重复元素和用例**
   ```bash
   python check_duplicates.py
   ```

2. **代码格式化**
   ```bash
   poetry run black .
   ```

3. **校验 YAML action schema**
   ```bash
   python validate_yaml_schema.py
   ```

   该命令只校验 `cases/*.yaml` 实际引用的 `data` 和递归引用的 `modules`，用于在浏览器启动前发现缺字段、错字段、未知 selector、缺失 module 等问题。

4. **导出依赖清单**
   ```bash
   poetry export -f requirements.txt --output requirements.txt
   ```

## AI 用例生成与智能执行

### 最新架构

AI 能力是当前框架的原生增强层，不是单独 runner，也不替代已有三层测试数据结构。执行仍然走原来的链路：

```text
poetry run run_case
→ pytest 动态收集 cases/*.yaml
→ CaseExecutor
→ StepExecutor
→ command executor
→ page_objects/BasePage
→ Playwright
```

AI 只介入两个阶段：

1. 用例生成阶段：读取项目上下文，生成符合当前项目结构的 YAML。
2. 用例执行阶段：在 selector/target 解析失败或需要自然语言步骤时，提供智能定位、视觉兜底和自愈。

当前核心目录职责固定如下：

| 目录 | 职责 | AI 生成/执行要求 |
|---|---|---|
| `cases/` | 只组织用例顺序 | 只写 `test_cases[].name`，不写步骤、模式、描述 |
| `data/` | 存储用例主体 | 结构化用例写 `mode/steps`；Agent 用例写 `type: agent_case` 和 `intent/steps/inputs/checkpoints/final` |
| `elements/` | 存储元素 key 和 selector | YAML 是权威资产；运行时自愈只记录已验证 selector evidence，不自动回写 |
| `modules/` | 存储公共组件/复用流程 | 生成用例优先复用，例如登录组件 |
| `vars/` | 存储项目变量 | 生成用例优先引用变量，不硬编码测试数据 |

生成链路由 `GenerationHarness` 统一门禁：

- `cases` 层不能写 `description`、`mode`、`steps`。
- 普通结构化用例的 `mode` 只允许 `strict`、`smart`；`agent_case` 不声明 `mode`。
- 生成用例必须有项目格式的断言步骤，例如 `assert_text`、`assert_text_contains`、`assert_visible`。
- 元素、模块、变量引用必须存在于当前项目资产或本次生成资产中。
- 模型输出会经过规范化、校验、必要时修复，再落盘。

### 环境配置

项目根目录 `.env` 只使用通用字段，不再读取厂商专用字段：

```env
LLM_BASE_URL=http://10.168.78.49:4000/v1
LLM_API_KEY=
LLM_MODEL=ep-xxxxxxxx
LLM_REASONING_EFFORT=medium

UI_VISION_BASE_URL=http://10.168.78.49:5100
UI_VISION_API_KEY=
BASE_URL=
```

说明：

- `LLM_BASE_URL` 会自动拼接为 `{LLM_BASE_URL}/chat/completions`。
- `LLM_MODEL` 是模型或私有 endpoint ID。
- `UI_VISION_BASE_URL` 指向同级部署的 UI Vision Service。
- 本地/内网模型场景默认不做脱敏，DOM 候选会保留真实 UI 文本和属性。

`config/ai_config.yaml` 是 AI 行为开关：

| 配置 | 作用 |
|---|---|
| `runtime.default_mode` | 默认执行模式，建议保持 `strict` |
| `runtime.allow_ai_in_smart` | `smart` 定位失败后是否允许 LLM DOM 兜底 |
| `runtime.ai_enabled` | 是否启用 AI 定位、AI 步骤和生成 |
| `runtime.max_ai_calls_per_test` | 单用例 AI 调用上限 |
| `runtime.ai_step_candidate_limit` | `ai_step` 发送给模型的候选元素上限 |
| `runtime.llm_selector_candidate_scan_limit` | selector 兜底前从页面扫描的候选上限 |
| `runtime.llm_selector_candidate_limit` | selector 兜底实际发送给 LLM 的候选上限 |
| `runtime.agent_candidate_scan_limit` | Agent 每轮从页面扫描的候选上限 |
| `runtime.agent_candidate_limit` | Agent 每轮实际发送给 LLM 的候选上限 |
| `runtime.agent_context_items` | Agent 每轮发送的项目资产摘要上限 |
| `runtime.agent_history_limit` | Agent 每轮发送的最近执行历史上限 |
| `runtime.agent_reasoning_effort` | Agent 运行时模型调用的专用推理强度，建议低于全局 `LLM_REASONING_EFFORT` |
| `runtime.agent_timeout_seconds` | Agent 运行时模型调用的专用超时时间 |
| `runtime.ai_cache_sqlite_path` | 统一 AI 缓存 SQLite 库，默认 `.ui_auto/ai_cache.sqlite3` |
| `runtime.agent_case_plan_cache_enabled` | 是否启用 `agent_case` 编译计划缓存 |
| `native_observe.include_open_shadow_dom` | 是否递归扫描 open shadow DOM，默认开启 |
| `native_observe.include_iframes` | 是否扫描 iframe；当前默认关闭，避免生成执行层无法直接消费的 frame selector |
| `native_observe.ignore_selectors` | DOM 观察时排除广告、推荐、footer、浮层等干扰区域 |
| `generation.verify_after_generate` | `gen` 写入后是否默认执行生成用例验证 |
| `agent_policy.limits.max_steps` | `agent_case` 单用例最大运行时动作数 |
| `agent_policy.limits.max_model_calls` | `agent_case` 单用例最大模型决策次数 |
| `agent_policy.limits.max_duration_seconds` | `agent_case` 单用例最大执行时长 |
| `agent_policy.guardrails.*` | `agent_case` 全局安全边界，单条用例不声明 |
| `llm.response_format` | 模型响应格式，默认 `auto`；GGUF 模型会自动降级为 `text` |
| `selector_registry.sqlite_path` | selector 自愈和历史定位缓存库 |
| `selector_registry.min_score_to_use` | registry selector 低于该分数不参与定位 |
| `selector_registry.deprecated_after_failures` | registry selector 连续失败达到阈值后标记 deprecated |
| `self_healing.persist_elements` | 预留开关；运行时自动回写入口已断开，默认不写 `elements/*.yaml` |
| `self_healing.min_persist_confidence` | 预留给显式确认更新流程的最低置信度 |
| `self_healing.persist_assertion_selectors` | 预留给显式确认更新流程；断言 selector 默认不建议持久化 |
| `vision.enabled` | 是否启用 UI Vision 标准兜底 |
| `vision.service_url` | UI Vision Service 地址，可被 `.env` 覆盖 |
| `vision.allow_coordinate_fallback` | 是否允许无法映射 DOM 时使用坐标兜底，默认建议 `false` |
| `vision.send_dom_candidates` | 调用视觉服务时是否发送 DOM 候选和坐标 |

运行时只保留两类缓存语义：

1. `agent_case_plan` 写入 `runtime.ai_cache_sqlite_path`，只缓存 `agent_case` 自然语言编译出的内存动作计划。缓存 key 包含 `project/env/entry_scope/spec fingerprint/元素、模块、变量内容 fingerprint/model/prompt/schema version`；YAML 内容变化后会失效。它不缓存执行轨迹、不缓存 module 展开副本、不缓存生成资产。
2. `selector_registry` 写入 `selector_registry.sqlite_path`，只保存已通过当前页面执行验证的 selector evidence。`smart` 使用它时仍必须实时验证，不能把缓存当成通过依据。

`gen` 没有隐藏生成缓存；它的正式产物就是写入 `cases/data/elements/modules/vars` 的 YAML。`run_case agent_case` 不读写 `gen` 的生成结果缓存，也不写正式 YAML 资产。

### 用例生成

生成规格放在：

```text
test_data/<project>/generation/*.yaml
```

推荐命令：

```powershell
poetry run gen -p demo saucedemo_ai
```

预览但不写文件：

```powershell
poetry run gen -p demo saucedemo_ai --dry-run
```

默认写入后会立即执行生成出的 case 文件做验证；验证通过表示 YAML 资产可执行。需要只生成不执行时使用：

```powershell
poetry run gen -p demo saucedemo_ai --no-verify
```

生成规则：

1. 命令中的 `-p demo` 和规格相对路径决定项目，规格文件不需要写 `project`。
2. 输出文件名默认和规格名一致，并默认覆盖对应生成文件。
3. `description` 只是可选说明，不是必填输入；`mode` 省略时默认按 `smart` 生成。
4. 规格可以是“步骤式”或“意图式”，都不在描述里硬塞“必须复用某元素/某组件”。
5. 复用 `elements/modules/vars` 是系统提示词、项目上下文和 Harness 的职责。
6. 入口 URL 遵循就近原则：优先读取自然语言 `steps` 或 `description` 中的 URL；没有则复用已有 `modules` 里的 `goto/open/navigate`；再没有才使用项目环境配置的 `base_url`。
7. 生成完成后会写入 `cases/`、`data/`，必要时写入 `elements/`、`modules/`、`vars/`。

当前支持两种生成规格模式。

步骤式规格适合业务流程已经明确、希望模型严格按步骤生成：

```yaml
cases:
  - name: saucedemo_standard_user_cart_logout_flow
    steps:
      - "打开 https://www.saucedemo.com/ 登录页"
      - "使用标准用户登录"
      - "断言登录后进入商品列表页，页面标题为 Products"
      - "添加 Sauce Labs Backpack 到购物车"
      - "断言购物车角标数量为 1"
      - "打开购物车页面"
      - "断言进入购物车页，页面标题为 Your Cart"
      - "断言购物车中存在 Sauce Labs Backpack 商品"
      - "打开左侧菜单"
      - "断言 Logout 退出登录入口可见"
      - "点击 Logout 退出登录"
      - "断言已回到登录页，登录按钮可见"
    checkpoints:
      - "商品页标题为 Products"
      - "购物车角标数量为 1"
      - "购物车页标题为 Your Cart"
      - "购物车商品名称为 Sauce Labs Backpack"
      - "Logout 退出登录入口可见"
    final:
      - "退出登录后登录按钮可见"
```

意图式规格适合只描述业务目标，让模型根据项目上下文自行拆分步骤并补齐断言：

```yaml
cases:
  - name: saucedemo_standard_user_cart_logout_ai_generated
    intent: "打开 https://www.saucedemo.com/，标准用户登录 Saucedemo 后，添加指定商品到购物车，进入购物车确认商品和数量，然后从左侧菜单退出登录。"
    inputs:
      user_type: standard
      product_name: Sauce Labs Backpack
    checkpoints:
      - "登录后进入商品列表页，页面标题为 Products"
      - "购物车角标数量为 1"
      - "进入购物车页，页面标题为 Your Cart"
      - "购物车中存在 Sauce Labs Backpack 商品"
      - "左侧菜单中 Logout 退出登录入口可见"
    final:
      - "退出登录后回到登录页，Login 按钮可见"
```

意图式示例命令：

```powershell
poetry run gen -p demo saucedemo_ai_intent
```

生成后的结构示例：

```yaml
# test_data/demo/cases/saucedemo_ai.yaml
test_cases:
  - name: test_saucedemo_standard_user_cart_logout_flow
```

```yaml
# test_data/demo/data/saucedemo_ai.yaml
test_data:
  test_saucedemo_standard_user_cart_logout_flow:
    description: 标准用户完成登录、添加Backpack、查看购物车并退出登录
    mode: smart
    steps:
      - use_module: saucedemo_login
        params:
          username: ${standard_username}
      - action: click
        selector: add_to_cart_backpack_btn
      - action: assert_text
        selector: shopping_cart_badge
        value: '1'
      - action: click
        selector: shopping_cart_link
      - action: assert_text
        selector: sauce_page_title
        value: Your Cart
      - action: assert_text
        selector: cart_item_backpack_name
        value: Sauce Labs Backpack
      - action: click
        selector: menu_button
      - action: assert_visible
        selector: logout_sidebar_link
      - action: click
        selector: logout_sidebar_link
      - action: assert_visible
        selector: sauce_login_btn
```

### 模式、生成单元与能力边界

AI 相关能力按三类概念划分，避免把执行、生成和视觉兜底混成一种“万能 AI 模式”：

| 概念 | 类型 | 作用 |
|---|---|---|
| `strict` | Execution Mode | 确定性执行标准 steps |
| `smart` | Execution Mode | 标准 steps 的智能定位、自愈和兜底执行 |
| `ai_step` | Step Capability | 单条自然语言意图在执行时解析为一个标准 step |
| `agent_case` | Runtime Case | 一句话目标或自然语言步骤计划，运行时先编译为内存动作计划，再交给 StepExecutor 执行 |
| `gen` | Asset Generation | 需求/intent/自然语言流程生成正式 `cases/data/elements/modules/vars` 资产 |
| `vision` | Capability | 截图/OCR/视觉定位能力，只能作为 smart 或 Agent 定位链路的兜底 |

`vision` 不是 YAML action，不允许写 `vision_click`、`vision_fill`、`vision_assert`。最终执行必须落到框架已有的标准 action，例如 `click`、`fill`、`assert_text`、`assert_visible`。

`mode` 可以配置在命令行、用例数据层或步骤层。优先级从高到低：

1. 单个步骤的 `mode`。
2. `data.<case>.mode`。
3. `-m/--ai-mode` 或 `UI_AI_MODE`。
4. `config/ai_config.yaml` 的 `runtime.default_mode`。

普通结构化用例只支持两个 `mode` 值：

| YAML mode | 语义 | 硬边界 |
|---|---|---|
| `strict` | 确定性执行模式，只执行标准结构化 steps | 只允许使用已注册 `selector/element key`；不调用 LLM、Vision、自愈；失败即失败 |
| `smart` | 增强执行模式，步骤的 `action/target/value/assertion` 已确定 | 只在当前 step 的目标语义范围内修复定位；不改业务流程；不新增运行时资产 |

`agent_case` 是独立用例形态，不写 `mode`；框架内部按 Agent 运行。`mode` 只属于普通结构化 steps。

| YAML type | 语义 |
|---|---|
| `standard` 或不写 | 普通结构化用例 |
| `agent_case` | 运行时智能用例，使用顶层 `intent` 或 `steps`，再配合 `inputs/checkpoints/final`；执行前编译为内存动作计划，不写正式 YAML 资产 |

步骤级 smart：

```yaml
- action: click
  target: "Sauce Labs Backpack add to cart button"
  mode: smart
```

用例级 smart：

```yaml
test_data:
  test_login:
    description: 登录流程
    mode: smart
    steps:
      - action: click
        target: Open Menu
```

原生 AI 步骤只适合单一 UI 动作或单一断言：

```yaml
- action: ai_step
  instruction: "Click the shopping cart link in the top-right header."
```

`ai_step` 不是直接操作浏览器。它会先由模型编译为一个框架已有标准 step，然后继续走统一 command executor。允许的结果包括 `click`、`fill`、`press_key`、`wait` 等原子动作。

`ai_step` 的硬规则：

1. 一个 `ai_step` 只能编译为一个标准 step。
2. 如果指令包含两个或更多动作、断言或业务流程，例如“登录并打开购物车”，运行时必须失败。
3. 多动作需求应拆成多个结构化 steps，或改成 `agent_case`。

Agent 运行时用例用于自然语言用例的正式执行。框架会先把 `intent/steps/inputs/checkpoints/final` 编译成一次性的内存动作计划，再交给已有 StepExecutor 执行。编译成功且用例跑通后可以写入 `agent_case_plan` 缓存；下次在项目资产内容未变化时可直接复用动作计划，避免再次调用编译模型。

`agent_case_plan` 只缓存动作计划，例如 `click/fill/assert/use_module` 这些 step 结构；不缓存执行轨迹，不缓存 module 展开后的副本，也不缓存最终 selector。执行时 selector 仍按当前 YAML、`selectors.db` 和自愈链路实时解析。

Agent 执行器不内置具体页面业务流程。框架只负责压缩 DOM、提供项目元素/模块/变量资产、约束模型输出为单个标准 action，并把 action 交给 StepExecutor。某个页面的登录、加购、下单、退出等业务顺序必须来自 `data`、`generation`、`elements`、`modules` 或模型运行时决策，不能写进 `src` 通用代码。

步骤式 Agent 用例适合已经知道业务顺序，但希望每一步由 Agent 根据实时页面状态执行：

```yaml
test_data:
  test_saucedemo_agent_case_steps_cart_logout_flow:
    description: Agent运行时steps式
    type: agent_case
    steps:
      - "打开 https://www.saucedemo.com/ 登录页"
      - "使用标准用户登录 Saucedemo"
      - "添加 Sauce Labs Backpack 到购物车"
      - "打开购物车页面"
      - "打开左侧菜单并点击 Logout 退出登录"
    inputs:
      username: ${standard_username}
      password: ${common_password}
      product_name: Sauce Labs Backpack
    checkpoints:
      - "购物车角标数量为 1"
      - "购物车页面中存在 Sauce Labs Backpack"
    final:
      - "退出登录后回到登录页，Login 按钮可见"
```

意图式 Agent 用例适合只给一句目标，让 Agent 自己探索执行路径：

```yaml
test_data:
  test_saucedemo_agent_case_cart_checkout_flow:
    description: Agent运行时用例
    type: agent_case
    intent: "标准用户登录 Saucedemo，添加指定商品到购物车，进入购物车确认商品和数量，并完成下单流程。"
    inputs:
      user_type: standard
      username: ${standard_username}
      password: ${common_password}
      product_name: Sauce Labs Backpack
      checkout_info:
        first_name: Test
        last_name: User
        postal_code: "100000"
    checkpoints:
      - "登录成功后进入商品列表页，页面标题为 Products"
      - "购物车页面中存在 Sauce Labs Backpack"
      - "订单确认页中商品为 Sauce Labs Backpack"
    final:
      - "完成下单后进入 checkout-complete 页面"
      - "页面展示 Thank you for your order"
```

`agent_case` 的执行策略来自全局 `agent_policy`，不要在单条用例里写 `limits/guardrails`。这样用例文件只描述业务意图、输入数据和验收标准。

建议使用顺序：

1. 稳定业务用例优先写 `action + selector`。
2. selector 不稳定但目标明确时写 `action + target + mode: smart`。
3. 单个自然语言动作可以使用 `ai_step`，但不要把它当作多步骤用例。
4. 已知业务顺序但需要运行时观察执行时，使用 `type: agent_case` + `steps/checkpoints/final`。
5. 需要执行时边看边探索的一句话流程使用 `type: agent_case` + `intent/checkpoints/final`。
6. 需求、PRD 或一句话 intent 要沉淀成正式用例时，使用 `poetry run gen -p <project> <spec>`。
7. DOM 无法稳定定位时由 `smart/agent` 链路自动调用 `vision`，不要在用例中写视觉专用 action。

### 标准定位与兜底顺序

当前执行时的 selector 解析顺序：

```text
明确 selector 验证
→ selector registry 历史定位
→ DOM 规则/语义候选
→ UI Vision 截图理解 + DOM 候选映射
→ LLM DOM selector 选择
→ 可选坐标兜底
```

关键点：

- 稳定 selector 会直接通过，不调用 AI 或 Vision。
- DOM 规则和 registry 比模型便宜，优先执行。
- UI Vision 是标准兜底能力，不是新的 YAML action。
- LLM DOM selector 更省 token、模型要求更低；UI Vision 更适合 DOM 不可靠、OCR、图标、遮挡、视觉布局判断。
- `vision.allow_coordinate_fallback=false` 时，Vision 结果必须能映射回 selector，否则继续后续兜底或失败。

### DOM 上下文压缩

运行时传给文本模型的 DOM 信息会经过高质量压缩，不直接发送 Playwright 收集到的原始候选。模型看到的是结构化 `dom_context`，不是 HTML。

压缩原则：

1. 保留稳定定位和语义判断需要的字段：`element_id`、`role`、`name`、`text`、`near_text`、`selector_candidates`、`data_test`、`aria_label`、`placeholder`。
2. 删除文本模型低价值字段：`class_name`、`bbox`、`center`、`bbox_norm`、`center_norm`。
3. 根据当前 `intent/steps/target/assertions` 动态提取关键词并对候选打分，优先保留交互元素、`data-test` 元素、可断言文本和与当前任务语义匹配的元素；通用代码不内置具体页面业务关键词。
4. 文本字段会截断，避免一个商品列表或弹窗把上下文撑爆。
5. UI Vision 链路仍可使用几何信息；坐标只给视觉服务，不给普通文本 LLM。

标准结构：

```yaml
dom_context:
  meta:
    url: https://www.saucedemo.com/inventory.html
    title: Swag Labs
    route_hint: inventory
  page_summary:
    main_heading: Products
    visible_text_summary:
      - Products
      - Sauce Labs Backpack
      - Shopping cart badge: 1
  forms: []
  business_objects:
    cards:
      - name: Sauce Labs Backpack
        summary: Sauce Labs Backpack $29.99 Add to cart
        actions:
          add:
            element_id: e12
            selector_candidates:
              - button[data-test="add-to-cart-sauce-labs-backpack"]
  interactive_elements:
    - id: e12
      role: button
      name: Add to cart
      near_text: Sauce Labs Backpack $29.99
      selector_candidates:
        - button[data-test="add-to-cart-sauce-labs-backpack"]
  assertion_candidates:
    - id: a18
      type: badge
      text: "1"
      near_text: Shopping cart
      selector_candidates:
        - span[data-test="shopping-cart-badge"]
  compression:
    raw_element_count: 120
    kept_element_count: 34
    context_level: 2
```

模型必须返回候选中的 `element_id`。框架负责把 `element_id` 映射到 `selector_candidates` 并通过 Playwright 验证。

默认策略：

```yaml
runtime:
  candidate_limit: 120
  ai_step_candidate_limit: 40
  llm_selector_candidate_scan_limit: 120
  llm_selector_candidate_limit: 40
  agent_candidate_scan_limit: 120
  agent_candidate_limit: 40
  agent_context_items: 40
  agent_history_limit: 10

native_observe:
  include_open_shadow_dom: true
  include_iframes: false
  ignore_selectors:
    - ".ads"
    - ".recommend"
    - "footer"
```

含义是：框架可以从页面扫描较多候选，但真正发给文本模型的是压缩后的前 40 个高价值候选。这样比简单截取前 N 个 DOM 更稳定，也比全量 DOM 更省 token。

### AI 返回压缩

模型返回也必须压缩、结构化，不能把长篇分析进入下一轮上下文。

Agent 返回协议：

```yaml
status: ok
action: click
element_id: e12
reason: 点击目标商品加购按钮
expected: 购物车角标变为 1
confidence: 0.94
criteria_update:
  passed: []
  failed: []
  pending:
    - 购物车页面中存在 Sauce Labs Backpack
```

Smart Locator 返回协议：

```yaml
status: ok
selected_element_id: e12
reason: 候选元素位于目标商品卡片内
confidence: 0.96
```

边界规则：

1. 每次模型只允许返回一个动作或一个元素选择。
2. `reason`、`expected` 是短摘要，不是完整推理过程。
3. `reason`、`expected` 会被框架截断到短文本后进入日志和压缩历史。
4. 原始模型请求/响应只保存到 `logs/model_io/<run_id>/`，默认不进入下一轮上下文。
5. 下一轮只传 `agent_state`、`recent_actions`、`criteria` 摘要和新的 `dom_context`。
6. 信息不足时返回 `status: need_more_context`；危险或外部不可控状态返回 `status: blocked`。
7. 模型返回 `element_id` 后，框架映射 selector 并校验；模型直接返回 selector 会被拒绝，不能绕过候选验证。

### Selector 自愈与缓存

当 `smart` 模式发现 selector 失效时，会触发自愈：

1. 记录原 selector 和失败原因。
2. 从 `selectors.db`、DOM 语义、LLM 或 Vision 中寻找候选。
3. 当前步骤立即使用新 selector 继续执行。
4. 用例最终通过后，把已验证 selector evidence 提交到 `selectors.db`。
5. 如果自愈找到了比 YAML 更新的 selector，只记录日志提示，不自动改写 `elements/*.yaml`。

执行顺序：

- `strict`：只用 YAML/显式 selector，失败即失败。
- `smart` 有 `selector` 或 element key：先验证当前 YAML/显式 selector，失败后才查 `selectors.db` 和自愈。
- `smart` target-only：没有 YAML selector 可读，可先查 `selectors.db` 候选，但候选必须在当前页面短超时验证。
- `agent_case`：plan cache 只决定动作计划；动作执行时仍按上述规则实时解析 selector。

持久化边界：

- `selectors.db` 是跨用例的 verified selector evidence，不是唯一产物。
- `elements/*.yaml` 是用户维护的权威 selector 资产。
- 运行时不会自动回写 `elements/*.yaml`；旧回写实现保留为未来“用户显式确认更新”入口，当前入口已断开。

### UI Vision Service

UI Vision Service 独立部署在当前项目同级目录：

```text
D:\project\ui_vision_service
```

它只负责视觉识别，不负责执行点击、输入、断言流程：

```text
UI 自动化执行机
→ 截图 + DOM candidates
→ UI Vision Service
→ OCR + 视觉模型
→ 返回 selector / candidate / box / center / reason
→ Playwright 执行
```

Docker 启动：

```powershell
cd D:\project\ui_vision_service
docker compose up -d --build
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5100/health
```

执行 demo 视觉兜底用例：

```powershell
poetry run run_case -p demo -f vision_showcase
```

日志中应能看到类似：

```text
AI执行模式: smart | 定位来源: UI Vision DOM兜底 | 选择器: #login-button | vision_reason: ...
```

### Demo 覆盖用例

当前 demo 已提供以下 AI 能力样例：

| 文件 | 覆盖能力 | 执行命令 |
|---|---|---|
| `saucedemo_ai` | 生成用例、模块复用、smart 稳定执行 | `poetry run run_case -p demo -f saucedemo_ai` |
| `ai_modes_showcase` | strict、步骤级 smart、Agent steps、Agent intent/explore 运行时用例 | `poetry run run_case -p demo -f ai_modes_showcase` |
| `vision_showcase` | 标准 UI Vision DOM 兜底 | `poetry run run_case -p demo -f vision_showcase` |

全量 demo 验证：

```powershell
poetry run run_case -p demo
```

### 日志与排查

主步骤日志保持统一格式：

```text
执行步骤: click | 选择器: #login-button | 值: None
```

AI/Smart 会追加定位说明：

```text
AI执行模式: smart | 定位来源: DOM语义匹配 | 目标: Open Menu | 选择器: ... | AI兜底: 否
```

Agent 用例会输出编译计划和执行轨迹：

```text
Agent用例执行开始: case=test_saucedemo_agent_case_cart_checkout_flow | max_steps=30 | max_model_calls=18 | input_type=intent | intent=...
Agent plan缓存命中: case=... | steps=9 | key_prefix=...
Agent内存编译完成: case=... | steps=9 | model_calls=1 | cache_hit=False | elements=0 | modules=0
Agent执行动作: case=... | step=4/30 | action=click | target=Sauce Labs Backpack Add to cart button
Agent编译执行完成: case=... | steps_executed=12 | model_calls=1 | reason=compiled_steps_completed
```

断言成功日志会输出预期和实际：

```text
断言通过: action=assert_text | selector=.shopping_cart_badge | 预期结果=1 | 实际结果=1
```

模型 I/O 追踪：

```text
logs/model_io/<run_id>/*.json
```

规则：

- 成功 run 自动清理模型 I/O 明细。
- 失败 run 保留模型请求和原始响应，并在 token usage summary 中写入 `model_io_dir`。
- 用例生成失败会保留 `logs/generation_runs/<timestamp>_<project>_<spec>/`，用于查看模型输入、模型原始输出、Harness 修复输入和失败原因。

## 项目结构

```
zhijia_ui/
├── .env                     # 本地模型和被测系统环境变量，不提交真实密钥
├── .ui_auto/                # AI/Smart 运行时缓存，例如 ai_cache.sqlite3、selectors.db
├── config/                  # 配置文件目录
│   ├── ai_config.yaml       # AI生成与智能执行配置
│   ├── env_config.yaml      # 环境配置
│   └── test_config.yaml     # 测试配置
├── evidence/                # 测试证据目录
│   └── screenshots/         # 测试截图
├── logs/                    # 运行日志
├── page_objects/            # 页面对象目录
│   └── base_page.py         # 基础页面类
├── reports/                 # 测试报告目录
│   └── allure-results/      # Allure报告数据
├── src/                     # 框架核心代码
│   ├── ai_generation/       # AI用例生成与Harness校验
│   ├── ai_runtime/          # AI/Smart执行、定位解析和选择器缓存
│   ├── runner.py            # 动态用例收集与执行
│   ├── test_case_executor.py # 测试用例执行器
│   ├── test_step_executor.py # 测试步骤执行器入口
│   └── step_actions/        # 步骤操作实现
├── test_data/               # 测试数据目录
│   ├── <project>/           # 各项目的测试数据
│   │   ├── cases/           # 用例顺序组织，只声明 name
│   │   ├── data/            # 用例描述、mode 和 steps
│   │   ├── elements/        # 页面元素定义
│   │   ├── generation/      # AI自然语言生成规格，不参与运行时执行
│   │   ├── modules/         # 可复用测试模块
│   │   └── vars/            # 项目变量
│   └── common/              # 公共测试数据
├── utils/                   # 工具类目录
│   ├── config.py            # 配置处理工具
│   ├── logger.py            # 日志处理工具
│   ├── report_handler.py    # 报告处理工具
│   ├── variable_manager.py  # 变量管理工具
│   └── yaml_handler.py      # YAML处理工具
├── conftest.py              # Pytest配置文件
├── check_duplicates.py      # 重复检查工具
├── poetry.lock              # Poetry依赖锁定文件
├── pyproject.toml           # Poetry项目配置
├── pytest.ini               # Pytest配置文件
├── README.md                # 项目文档
└── test_runner.py           # 根命令入口，日常推荐使用 poetry run run_case / poetry run gen
```

## 测试开发指南

### 测试用例编写

#### 用例组织结构

测试用例由 `cases`、`data`、`elements` 三层核心文件协同定义。`modules` 和 `vars` 是复用资产，供步骤引用。

1. **用例组织文件** (`cases/*.yaml`)
    - 只负责组织用例执行顺序。
    - 每个条目只声明 `name`。
    - 不在 `cases` 层写 `description`、`mode`、`steps`。
    - 示例：
      ```yaml
      test_cases:
        - name: test_login_success
        - name: test_login_error
      ```

2. **用例步骤文件** (`data/*.yaml`)
    - 定义用例说明、默认执行模式和具体步骤。
    - `description`、`mode`、`steps` 都放在 `data.<case_name>` 下。
    - 支持条件分支、循环、模块引用和断言。
    - 示例：
      ```yaml
      test_data:
        test_login_success:
          description: "用户登录成功"
          mode: strict
          steps:
            - action: goto
              value: "${base_url}/login"

            - action: fill
              selector: username_input
              value: "${username}"

            - action: fill
              selector: password_input
              value: "${password}"

            - action: click
              selector: login_button

            - action: assert_visible
              selector: welcome_message
      ```

3. **元素定义文件** (`elements/*.yaml`)
    - 集中定义页面元素的定位策略。
    - 步骤中的 `selector` 优先引用这里的元素 key。
    - 支持 CSS、XPath、Text 等定位方式。
    - 示例：
      ```yaml
      elements:
        username_input: "#username"
        password_input: "#password"
        login_button: "button.login-btn"
        welcome_message: ".welcome-text"
      ```

4. **公共组件文件** (`modules/*.yaml`)
    - 存放可复用步骤片段。
    - 在 `data` 步骤中使用 `use_module` 引用。

5. **变量文件** (`vars/*.yaml`)
    - 存放项目级测试变量。
    - 步骤中使用 `${variable_name}` 引用。

#### 用例编写流程

1. **规划测试用例**
    - 确定测试目标和范围
    - 设计测试步骤和预期结果

2. **定义页面元素**
    - 在 `elements/elements.yaml` 中添加所需元素
    - 使用最稳定的定位方式
    - 添加清晰的描述信息

3. **编写用例组织文件**
    - 在 `cases/*.yaml` 中添加用例顺序
    - 每条用例只声明 `name`
    - 确保 `name` 能在同名或已加载的 `data/*.yaml` 中找到

4. **编写测试步骤**
    - 在 `data/*.yaml` 中添加 `description`、`mode` 和 `steps`
    - 按顺序编写清晰的操作步骤
    - 使用变量和参数化数据
    - 每条生成用例必须包含至少一个 `assert_*` 断言步骤

## 高级功能

我们对原有 Playwright UI 自动化测试框架进行了一系列增强，使其更加灵活、可维护、高效。主要新增功能包括：

1. **模块化测试片段系统**：支持将常用测试步骤封装为可重用模块
2. **条件分支执行**：支持在测试用例中使用 if-then-else 条件分支
3. **循环执行**：支持循环遍历数据列表执行重复操作
4. **增强的变量管理**：支持全局/测试用例/临时多级作用域变量
5. **AI 用例生成与智能执行**：支持生成当前项目格式的 YAML 用例，并在执行时按需启用 smart 定位能力

### 模块化测试片段

模块化测试片段允许将常用操作封装为可重用模块，减少重复编写相同步骤的工作。

#### 定义模块

在 `test_data/<项目>/modules/` 目录下创建 YAML 文件，定义可重用的步骤：

```yaml
# test_data/demo/modules/login.yaml
login:
  - action: navigate
    value: "/login"
    description: "打开登录页面"

  - action: fill
    selector: "username_input"
    value: "${username}"
    description: "输入用户名"

  - action: fill
    selector: "password_input"
    value: "${password}"
    description: "输入密码"

  - action: click
    selector: "login_button"
    description: "点击登录按钮"
```

#### 使用模块

在测试用例中引用模块：

```yaml
# test_data/demo/data/test.yaml
test_data:
  login_test:
    description: "登录模块复用示例"
    mode: strict
    steps:
      - use_module: login
        description: "使用登录模块"

      - action: assert_text
        selector: welcome_message
        expected: "欢迎回来"
        description: "验证登录成功"
```

### 条件分支执行

支持在测试用例中使用 if-then-else 条件分支，根据不同条件执行不同的测试步骤。

#### 基本语法

```yaml
- if: "${{ condition_expression }}"
  then:
    - action: some_action
      # ... 条件为真时执行的步骤
  else:
    - action: other_action
      # ... 条件为假时执行的步骤（可选）
```

#### 支持的条件类型

**1. 变量比较**
```yaml
- if: "${{ ${user_type} == 'admin' }}"
  then:
    - action: click
      selector: "admin_panel_button"
  else:
    - action: navigate
      value: "/user-dashboard"
```

**2. UI元素状态判断**
```yaml
# 元素可见性判断
- if: "${{ element_visible('#login-button') }}"
  then:
    - action: click
      selector: "#login-button"
  else:
    - action: wait
      value: 2

# 元素存在性判断
- if: "${{ element_exists('.error-message') }}"
  then:
    - action: assert_text_contains
      selector: ".error-message"
      expected: "错误信息"

# 元素启用状态判断
- if: "${{ element_enabled('#submit-btn') }}"
  then:
    - action: click
      selector: "#submit-btn"
  else:
    - action: wait
      value: 1
```

**3. 元素内容判断**
```yaml
# 元素文本内容判断
- if: "${{ element_text('#status') == 'ready' }}"
  then:
    - action: click
      selector: "#submit"

# 元素属性值判断
- if: "${{ element_attribute('#input', 'disabled') == None }}"
  then:
    - action: fill
      selector: "#input"
      value: "test data"

# 元素数量判断
- if: "${{ element_count('.item') > 0 }}"
  then:
    - action: click
      selector: ".item:first-child"
```

**4. 复合条件判断**
```yaml
# 多条件组合
- if: "${{ element_visible('#modal') and element_text('#modal .title') == 'Confirm' }}"
  then:
    - action: click
      selector: "#modal .confirm-btn"

# 复杂逻辑判断
- if: "${{ ${retry_count} < 3 and element_exists('.loading') }}"
  then:
    - action: wait
      value: 1
```

#### 可用的UI元素检查函数

| 函数名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `element_exists(selector)` | 元素选择器 | boolean | 检查元素是否存在于DOM中 |
| `element_visible(selector)` | 元素选择器 | boolean | 检查元素是否可见 |
| `element_enabled(selector)` | 元素选择器 | boolean | 检查元素是否启用（非disabled） |
| `element_text(selector)` | 元素选择器 | string | 获取元素的文本内容 |
| `element_attribute(selector, attr_name)` | 元素选择器, 属性名 | string/None | 获取元素的指定属性值 |
| `element_count(selector)` | 元素选择器 | number | 获取匹配元素的数量 |

#### 实际应用示例

```yaml
# 智能登录流程
test_cases:
  - name: test_smart_login

test_data:
  test_smart_login:
    description: "智能登录流程"
    mode: smart
    steps:
      - action: navigate
        value: "/login"

      # 检查是否已登录
      - if: "${{ element_exists('.user-profile') }}"
        then:
          - action: assert_visible
            selector: ".user-profile"
        else:
          # 执行登录流程
          - action: fill
            selector: "#username"
            value: "${username}"
          - action: fill
            selector: "#password"
            value: "${password}"
          - action: click
            selector: "#login-btn"
      
      # 检查登录结果
      - if: "${{ element_visible('.error-message') }}"
        then:
          - action: assert_text_contains
            selector: ".error-message"
            expected: "用户名或密码错误"
        else:
          - action: assert_visible
            selector: ".dashboard"
```

### 循环执行

循环执行允许对列表中的数据项执行重复操作，适用于批量处理场景。

```yaml
# 循环执行示例
- action: store_variable
  name: "product_ids"
  value: [ 1, 2, 3 ]
  scope: "test_case"
  description: "设置产品ID列表"

- for_each: "${product_ids}"
  as: "product_id"
  do:
    - action: click
      selector: "product_${product_id}_view"
      description: "查看产品${product_id}详情"

    - action: click
      selector: "add_to_cart_button"
      description: "添加到购物车"
```

### 变量管理

框架支持多级作用域的变量管理，包括全局变量、测试用例变量和临时变量。

```yaml
# 设置变量
- action: store_variable
  name: "username"
  value: "test_user"
  scope: "global"  # global, test_case, temp
  description: "存储用户名"

# 使用变量
- action: fill
  selector: "username_input"
  value: "${username}"
  description: "输入用户名"

# 变量表达式计算
- if: "${{ ${count} > 5 }}"
  then:
  # 当 count 大于 5 时执行的步骤
```

## 最佳实践

### 用例组织与管理

1. **模块化组织**
    - 按功能模块组织测试用例
    - 将常用操作封装为可复用模块
    - 使用标签对用例进行分类

2. **命名规范**
    - 使用描述性的用例名称，如 `test_login_valid_credentials`
    - 元素ID采用功能描述性命名，如 `login_button`
    - 测试步骤添加清晰的描述信息

3. **数据管理**
    - 使用变量文件管理测试数据，避免硬编码
    - 分离测试数据与测试逻辑
    - 使用参数化实现数据驱动测试

### 编写技巧

1. **元素定位**
    - 优先使用ID、CSS选择器等稳定的定位方式
    - 避免使用绝对路径和索引定位
    - 为复杂元素添加详细注释

2. **等待策略**
    - 使用显式等待而非固定等待时间
    - 为关键操作添加适当的超时时间
    - 利用自适应等待机制提高测试稳定性

3. **断言与验证**
    - 每个关键步骤后添加验证
    - 使用精确的断言而非模糊的检查
    - 在流程转换点添加状态验证

4. **错误处理**
    - 添加适当的错误捕获和恢复机制
    - 为失败用例提供详细的错误信息
    - 实现智能重试机制处理闪现问题

## 常用命令

```bash
# 导出依赖清单
$ poetry export -f requirements.txt --output requirements.txt

# 录制测试脚本
$ playwright codegen "https://example.com"

# 代码格式化
$ poetry run black .

# 检查重复元素和用例
$ python check_duplicates.py

# 运行指定项目和文件
$ poetry run run_case -p demo -f test_cases

# 使用 smart 模式执行用例
$ poetry run run_case -p demo -f saucedemo_ai -m smart

# 基于自然语言规格调用模型生成用例
$ poetry run gen -p demo saucedemo_ai
```

## 元素定位技巧

### 元素提取工具

- **Chrome插件**: SelectorsHub
- **PyCharm插件**: Test Automation
- **Playwright录制功能**: 自动生成元素定位符
- **AI辅助**: 优先使用框架内置 smart 模式、selector 自愈和 UI Vision 兜底，不建议在用例中手工复制模型生成的临时 selector

### 元素定位最佳实践

1. **优先使用稳定定位符**
    - 优先级: ID > 数据属性 > CSS选择器 > XPath
    - 避免使用索引和绝对路径

2. **使用语义化命名**
    - 元素ID应描述其功能而非位置
    - 例如: `login_button` 而非 `button_1`

3. **添加详细注释**
    - 为复杂元素添加清晰的描述
    - 说明元素的用途和位置

### 代码规范

- 遵循 PEP 8 Python 代码风格指南
- 为所有新功能添加测试
- 保持文档和代码注释的同步更新

## 常见问题

### 安装问题

**Q: 安装依赖时出现错误**

A: 尝试以下解决方案：

1. 确保您使用的是 Python 3.12 或更高版本
2. 更新 Poetry: `pip install --upgrade poetry`
3. 清除 Poetry 缓存: `poetry cache clear --all pypi`

**Q: Playwright 浏览器安装失败**

A: 手动安装浏览器：

```bash
python -m playwright install chromium
```

### 运行问题

**Q: 测试运行不稳定，经常失败**

A: 可能的原因和解决方案：

1. 增加等待时间或使用显式等待
2. 检查元素定位符是否稳定
3. 在调试模式下运行以获取更多信息

**Q: 如何调试复杂的测试用例？**

A: 使用以下方法：

1. 添加 `page.pause()` 暂停浏览器
2. 使用有头模式运行: `poetry run run_case -p demo --headed`
3. 检查生成的日志和截图

**Q: smart 模式会不会影响历史用例？**

A: 默认执行模式仍是 `strict`。只有命令行指定 `-m/--ai-mode smart`，或 `data`/步骤中声明 `mode: smart` 时才会启用智能定位。`run_case` 不再支持运行时 `ai_case` 编译；自然语言探索使用 `agent_case`，正式资产生成使用 `gen`。

**Q: 生成用例为什么必须有断言？**

A: 生成链路会强制每条用例至少包含一个项目已有格式的 `assert_*` 步骤。没有断言的用例无法证明业务结果，Harness 会拒绝写入。

**Q: AI 生成失败时应该先检查什么？**

A: 先检查 `.env` 中 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL` 是否正确，再检查生成规格是否能复用当前项目已有的元素、模块和变量。

### 其他问题

**Q: 如何实现数据驱动测试？**

A: 请参考文档中的"变量管理"部分，使用全局变量或测试用例变量来实现数据驱动。

**Q: 如何添加自定义操作？**

A: 可以通过扩展 `test_step_executor.py` 添加新的操作类型，或者使用模块化测试片段封装复杂操作。

## 相关资源

- [录制文档](https://test-crdqcu2hkpbr.feishu.cn/docx/YpIRdgOXGo1CQdxp1cacxwGbnZd?from=from_copylink)
- [Playwright 官方文档](https://playwright.dev/python/docs/intro)
- [PyTest 文档](https://docs.pytest.org/)
