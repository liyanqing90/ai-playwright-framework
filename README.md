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
- [贡献指南](#贡献指南)
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
- **AI/Smart执行模式**：在不破坏历史用例的前提下，支持 strict、smart、ai 三种执行模式

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

- `test_runner.py`: 主入口文件，负责解析命令行参数和启动测试
- `src/runner.py`: 核心运行器实现，负责测试用例的加载和执行

### 2. 测试用例执行器 (TestCaseExecutor)

测试用例执行器负责解析和执行单个测试用例，管理测试步骤的执行流程和错误处理。

**主要文件**:

- `src/test_case_executor.py`: 测试用例执行器实现

### 3. 测试步骤执行器 (StepExecutor)

测试步骤执行器负责执行单个测试步骤，包括 UI 操作、断言、变量管理等。

**主要文件**:

- `src/test_step_executor.py`: 测试步骤执行器入口（兼容性保留）
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
   python test_runner.py --project demo
   ```

2. **运行指定测试文件**
   ```bash
   python test_runner.py --project demo --test-file test_cases
   ```

3. **有头模式运行**
   ```bash
   python test_runner.py --project demo --headed
   ```

4. **使用 AI/Smart 执行模式运行**
   ```bash
   python test_runner.py --project demo --test-file saucedemo_ai --ai-mode smart
   ```

   `--ai-mode` 只作为默认执行模式使用。若 `data` 层用例或单个步骤已经声明 `mode`，则优先使用 YAML 中的配置。

### 生成用例

1. **基于项目自然语言规格调用模型生成**
   ```bash
   python test_runner.py --project demo --env prod --generate-case generation_specs/demo/saucedemo_ai.yaml --output-name saucedemo_ai --overwrite
   ```

2. **只预览生成结果，不写入文件**
   ```bash
   python test_runner.py --project demo --generate-case generation_specs/demo/saucedemo_ai.yaml --output-name saucedemo_ai --dry-run
   ```

生成规格按项目放在 `generation_specs/<project>/` 下；生成命令会写入 `test_data/<project>/cases/`、`data/`，并只在模型判断必须新增资产时写入 `elements/`、`modules/`、`vars/`。

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

### 设计边界

AI 能力是对现有框架的增强，不替代原有 YAML 三层结构：

1. `cases` 目录只组织用例执行顺序，只允许声明 `name`。
2. `data` 目录存储用例说明、执行模式和步骤。
3. `elements` 目录存储元素定位。
4. `modules` 目录存储公共组件，生成用例应优先复用。
5. `vars` 目录存储项目变量，生成用例应优先引用变量而不是硬编码数据。

生成链路由 `GenerationHarness` 做结构门禁：

- `cases` 层不能写入 `description`、`mode`、`steps`。
- `data.<case>.mode` 只允许 `strict`、`smart`、`ai`。
- 每条生成用例必须至少包含一个项目已有格式的断言步骤。
- 断言必须使用项目已有 `assert_*` 格式，例如 `assert_text`、`assert_text_contains`、`assert_visible`。
- 元素、模块引用必须能在当前项目资产或本次生成资产中找到。

### 环境配置

在项目根目录创建或维护 `.env`，用于放置模型服务配置。不要把真实私钥写入文档或提交到公共仓库。

```env
LLM_API_KEY=
LLM_BASE_URL=http://10.168.78.49:4000/v1
LLM_MODEL=ep-xxxxxxxx
LLM_REASONING_EFFORT=medium
UI_VISION_BASE_URL=http://10.168.78.49:5100
BASE_URL=
```

大模型接口地址只读取 `LLM_BASE_URL`。框架会自动请求 `{LLM_BASE_URL}/chat/completions`，不再读取厂商专用或其他兼容 URL 字段。
`UI_VISION_BASE_URL` 用于局域网 UI Vision Service，默认不做鉴权；如果 `config/ai_config.yaml` 中 `vision.enabled=false`，历史执行不会调用该服务。

AI 默认配置在 `config/ai_config.yaml`：

- `runtime.default_mode`: 默认执行模式，默认 `strict`。
- `runtime.allow_ai_in_smart`: `smart` 模式无法通过显式选择器、历史定位、规则定位解析时，是否允许 AI 兜底。
- `runtime.ai_enabled`: 是否启用 AI 定位和 AI 生成。
- `selector_registry.sqlite_path`: 智能定位结果缓存库，默认 `.ui_auto/selectors.db`。
- `vision.enabled`: 是否启用 UI Vision 兜底。默认 `false`，避免影响历史用例。
- `vision.service_url`: UI Vision Service 局域网地址，也可用 `.env` 中的 `UI_VISION_BASE_URL` 覆盖。
- `vision.allow_coordinate_fallback`: 是否允许在无法映射 DOM selector 时使用坐标点击/输入兜底。默认 `false`，推荐先保持关闭。
- `vision.send_dom_candidates`: 调用视觉服务时是否发送 DOM 候选及坐标，默认 `true`，用于让视觉结果优先回落到可复用 selector。

### 生成规格

生成规格只写业务意图，不把元素、变量、公共组件复用要求写到用例描述里。复用策略由系统提示词、项目上下文和 `GenerationHarness` 负责：

- 规格目录必须和 `--project` 对应，例如 `generation_specs/demo/*.yaml` 只能配合 `--project demo`。
- `project` 字段如果存在，必须和命令行 `--project` 一致。
- 推荐每个用例写成对象，并用 `steps` 的 `-` 列表表达自然语言步骤；不要把多个步骤塞进一个长字符串里让模型靠标点拆分。
- 模型会读取当前项目已有 `elements`、`modules`、`vars` 和历史用例。
- Prompt 要求优先复用已有公共组件、元素 key 和变量 key。
- Harness 会校验输出仍符合三层结构、断言格式、元素/模块引用合法性。

自然语言步骤规格示例：

```yaml
project: demo
description: "生成百度搜索流程用例"
mode: smart
cases:
  - name: baidu_search_keyword
    description: "百度搜索关键词"
    steps:
      - "打开百度网页"
      - "点击搜索输入框"
      - "输入百度"
      - "点击搜索按钮"
      - "断言搜索结果页打开成功"
```

业务场景规格示例：

```yaml
project: demo
description: "基于当前 demo 项目生成 Saucedemo 补充用例"
mode: smart
cases:
  - name: saucedemo_backpack_cart
    description: "标准用户添加单个商品到购物车"
    steps:
      - "打开 Saucedemo 登录页"
      - "使用标准用户登录"
      - "添加 Sauce Labs Backpack 到购物车"
      - "断言购物车数量为 1"
  - name: saucedemo_locked_user_error
    description: "锁定用户登录失败提示"
    steps:
      - "打开 Saucedemo 登录页"
      - "使用锁定用户登录"
      - "断言页面展示 locked out 错误提示"
```

输出文件仍然遵守项目三层结构：

```yaml
# test_data/demo/cases/saucedemo_ai.yaml
test_cases:
  - name: test_saucedemo_standard_user_add_backpack_to_cart
```

```yaml
# test_data/demo/data/saucedemo_ai.yaml
test_data:
  test_saucedemo_standard_user_add_backpack_to_cart:
    description: 标准用户登录后添加单个Sauce Labs Backpack到购物车
    mode: smart
    steps:
      - use_module: saucedemo_login
      - action: click
        selector: saucedemo_backpack_add
      - action: assert_text
        selector: saucedemo_cart_badge
        value: "1"
```

### 执行模式

执行模式可以配置在命令行、用例数据层或步骤层。优先级如下：

1. 单个步骤的 `mode`。
2. `data.<case>.mode`。
3. `--ai-mode` 或 `UI_AI_MODE` 环境变量。
4. `config/ai_config.yaml` 中的 `runtime.default_mode`。

三种模式说明：

- `strict`: 历史默认模式，只使用明确 selector，不做智能定位。
- `smart`: 先验证显式 selector；失效时依次尝试历史定位、规则定位；配置允许时再调用 AI 兜底。
- `ai`: 保留确定性定位优先级，但允许使用 AI 解析语义目标。

步骤级 AI 适合只让某个不稳定步骤进入智能定位：

```yaml
- action: click
  target: "登录按钮"
  mode: smart
```

用例级 `mode` 适合作为该用例所有步骤的默认模式：

```yaml
test_data:
  test_login:
    description: 登录流程
    mode: smart
    steps:
      - action: fill
        selector: username_input
        value: "${username}"
```

### UI Vision 兜底链路

UI Vision 不是新的 YAML 操作类型，而是 `smart`/`ai` 模式下的最后一层定位兜底：

```text
显式 selector
→ selector registry
→ 规则定位
→ LLM DOM 候选选择
→ UI Vision 截图理解 + DOM 候选映射
→ 可选坐标兜底
```

关键规则：

- 用例仍然写在 `cases`、`data`、`elements` 三层结构中，不写 `vision_click` 之类的新 action。
- UI Vision Service 通过 `UI_VISION_BASE_URL` 局域网访问，框架发送当前截图、目标描述、页面 URL 和 DOM 候选坐标。
- 服务如果能返回 `selector`、`selected_candidate_index` 或可映射到 DOM 候选的 `box/center`，框架会继续用 Playwright selector 执行并写入 selector registry。
- 只有 `vision.allow_coordinate_fallback=true` 且视觉结果无法映射 DOM 时，才会使用坐标点击/输入兜底；默认关闭，避免掩盖真实 selector 质量问题。
- 本地大模型场景默认不脱敏，DOM 候选会保留真实 UI 文本和属性，便于模型理解页面。

服务端代码位于当前项目同级目录 `D:\project\ui_vision_service`，按独立虚拟环境或 Docker 安装，避免 OCR/Paddle 重依赖影响主 UI 自动化运行环境：

```powershell
cd D:\project\ui_vision_service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 5100
```

Docker 部署：

```powershell
cd D:\project\ui_vision_service
copy .env.example .env
docker compose up -d --build
```

服务启动后可用以下命令验证健康状态：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5100/health
```

临时启用真实视觉兜底执行：

```powershell
$env:UI_VISION_ENABLED="true"
$env:UI_VISION_BASE_URL="http://127.0.0.1:5100"
$env:UI_VISION_API_KEY="sk-ui-vision-local"  # 如果服务端配置了 UI_VISION_API_KEY
python test_runner.py --project demo --test-file vision_probe --ai-mode smart
```

### 日志输出

为了兼容历史用例日志，主步骤日志保持原格式：

```text
执行步骤: click | 选择器: #login-button | 值: None
```

AI/Smart 模式会追加一行辅助日志：

```text
AI执行模式: smart | 定位来源: 显式选择器 | 选择器: #login-button | AI兜底: 否
```

断言成功会输出项目统一断言日志：

```text
断言通过: action=assert_text | selector=.shopping_cart_badge | expected=1
```

## 项目结构

```
zhijia_ui/
├── .env                     # 本地模型和被测系统环境变量，不提交真实密钥
├── .ui_auto/                # AI/Smart 运行时缓存，例如 selectors.db
├── config/                  # 配置文件目录
│   ├── ai_config.yaml       # AI生成与智能执行配置
│   ├── env_config.yaml      # 环境配置
│   └── test_config.yaml     # 测试配置
├── generation_specs/        # 按项目隔离的自然语言用例生成规格
│   └── <project>/           # 例如 demo/saucedemo_ai.yaml
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
│   ├── test_step_executor.py # 测试步骤执行器兼容入口
│   └── step_actions/        # 步骤操作实现
├── test_data/               # 测试数据目录
│   ├── <project>/           # 各项目的测试数据
│   │   ├── cases/           # 用例顺序组织，只声明 name
│   │   ├── data/            # 用例描述、mode 和 steps
│   │   ├── elements/        # 页面元素定义
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
└── test_runner.py           # 测试运行器与用例生成入口
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
5. **AI 用例生成与智能执行**：支持生成当前项目格式的 YAML 用例，并在执行时按需启用 smart/ai 定位能力

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
$ python test_runner.py --project demo --test-file test_cases

# 使用 smart 模式执行用例
$ python test_runner.py --project demo --test-file saucedemo_ai --ai-mode smart

# 基于自然语言规格调用模型生成用例
$ python test_runner.py --project demo --generate-case generation_specs/demo/saucedemo_ai.yaml --output-name saucedemo_ai --overwrite
```

## 元素定位技巧

### 元素提取工具

- **Chrome插件**: SelectorsHub
- **PyCharm插件**: Test Automation
- **Playwright录制功能**: 自动生成元素定位符
- **AI辅助**: 使用ChatGPT或DeepSeek分析HTML并生成定位符

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
2. 使用有头模式运行: `python test_runner.py --project demo --headed`
3. 检查生成的日志和截图

**Q: smart 模式会不会影响历史用例？**

A: 默认执行模式仍是 `strict`。只有命令行指定 `--ai-mode smart/ai`，或 `data`/步骤中声明 `mode: smart`、`mode: ai` 时才会启用智能定位。历史用例没有声明 mode 时按 strict 执行。

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
