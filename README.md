# 之家 UI 自动化测试框架

之家 UI 自动化测试框架是一个基于 Playwright 的 UI 自动化测试框架，采用模块化设计，支持 YAML
格式的测试用例定义，提供灵活、高效、可维护的自动化测试解决方案。

## 目录

- [框架特性](#框架特性)
- [安装与配置](#安装与配置)
- [使用指南](#使用指南)
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

### 高级特性

- **数据驱动能力**：
    - 多级变量管理：支持全局变量、测试用例级变量和临时变量
    - 表达式计算：支持条件判断、数学运算等
    - 模块化组件：可复用的测试步骤片段
    - 流程控制：支持if-then-else条件分支和for_each循环结构

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

1. Python 3.10+
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

3. **导出依赖清单**
   ```bash
   poetry export -f requirements.txt --output requirements.txt
   ```

## 项目结构

```
zhijia_ui/
├── config/                  # 配置文件目录
│   ├── cookie.json          # Cookie配置
│   ├── env_config.yaml      # 环境配置
│   ├── storage_state.json   # 浏览器存储状态
│   └── test_config.yaml     # 测试配置
├── evidence/                # 测试证据目录
│   └── screenshots/         # 测试截图
├── page_objects/            # 页面对象目录
│   ├── base_page.py         # 基础页面类
│   └── <project>/           # 各项目的页面对象
├── reports/                 # 测试报告目录
│   └── allure-results/      # Allure报告数据
├── src/                     # 框架核心代码
│   ├── fixtures.py          # Pytest fixtures
│   ├── load_data.py         # 数据加载模块
│   ├── test_case_executor.py # 测试用例执行器
│   ├── test_step_executor.py # 测试步骤执行器
│   └── step_actions/        # 步骤操作实现
├── test_data/               # 测试数据目录
│   ├── <project>/           # 各项目的测试数据
│   │   ├── excel/           # Excel格式的测试数据
│   │   ├── variables.json   # 项目级变量定义
│   │   ├── cases/           # 测试用例定义
│   │   ├── data/            # 测试数据定义
│   │   ├── elements/        # 页面元素定义
│   │   └── modules/         # 可复用测试模块
│   └── common/              # 公共测试数据
├── utils/                   # 工具类目录
│   ├── allure_logger.py     # Allure日志工具
│   ├── config.py            # 配置处理工具
│   ├── excel_handler.py     # Excel处理工具
│   ├── logger.py            # 日志处理工具
│   ├── report_handler.py    # 报告处理工具
│   ├── variable_manager.py  # 变量管理工具
│   └── yaml_handler.py      # YAML处理工具
├── conftest.py              # Pytest配置文件
├── create_structure.py      # 项目结构创建工具
├── check_duplicates.py      # 重复检查工具
├── poetry.lock              # Poetry依赖锁定文件
├── pyproject.toml           # Poetry项目配置
├── pytest.ini               # Pytest配置文件
├── README.md                # 项目文档
└── test_runner.py           # 测试运行器
```

## 测试开发指南

### 测试用例编写

#### 用例组织结构

测试用例由三个核心文件协同定义，采用模块化组织方式：

1. **用例元数据文件** (`cases/cases.yaml`)
    - 定义测试用例的基本信息
    - 包含用例名称、描述、标签、前置条件等
    - 示例：
      ```yaml
      test_login:
        description: "用户登录测试"
        tags: ["smoke", "login"]
        fixtures: ["setup_browser"]
        depends_on: []
      ```

2. **测试步骤文件** (`data/index.yaml`)
    - 定义测试用例的具体执行步骤
    - 包含操作类型、目标元素、参数值等
    - 支持条件分支、循环和模块引用
    - 示例：
      ```yaml
      test_login:
        steps:
          - action: goto
            value: "${base_url}/login"
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
 
          - action: expect_visible
            selector: "welcome_message"
            timeout: 5000
            description: "验证登录成功"
      ```

3. **元素定义文件** (`elements/elements.yaml`)
    - 集中定义页面元素的定位策略
    - 支持多种定位方式（CSS、XPath、Text等）
    - 每个元素包含选择器和描述信息
    - 示例：
      ```yaml
      username_input:  "#username" # "用户名输入框"
 
      password_input:  "#password" # "密码输入框"
 
      login_button:  "button.login-btn" # "登录按钮"
 
      welcome_message: ".welcome-text" # "欢迎信息"
      ```

#### 用例编写流程

1. **规划测试用例**
    - 确定测试目标和范围
    - 设计测试步骤和预期结果

2. **定义页面元素**
    - 在 `elements/elements.yaml` 中添加所需元素
    - 使用最稳定的定位方式
    - 添加清晰的描述信息

3. **编写测试用例元数据**
    - 在 `cases/cases.yaml` 中添加用例定义
    - 指定用例名称、描述和标签
    - 配置必要的前置条件

4. **编写测试步骤**
    - 在 `data/index.yaml` 中添加测试步骤
    - 按顺序编写清晰的操作步骤
    - 使用变量和参数化数据
    - 添加必要的断言和验证步骤

## 高级功能

我们对原有 Playwright UI 自动化测试框架进行了一系列增强，使其更加灵活、可维护、高效。主要新增功能包括：

1. **模块化测试片段系统**：支持将常用测试步骤封装为可重用模块
2. **条件分支执行**：支持在测试用例中使用 if-then-else 条件分支
3. **循环执行**：支持循环遍历数据列表执行重复操作
4. **增强的变量管理**：支持全局/测试用例/临时多级作用域变量

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
login_test:
  steps:
    - use_module: login
      params:
        username: "test_user"
        password: "password123"
      description: "使用登录模块"

    - action: assert_text
      selector: "welcome_message"
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
    - action: log
      value: "提交按钮未启用"
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
    - action: set_variable
      variable_name: "retry_count"
      value: "${{ ${retry_count} + 1 }}"
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
  - name: "智能登录测试"
    steps:
      - action: navigate
        value: "/login"
      
      # 检查是否已登录
      - if: "${{ element_exists('.user-profile') }}"
        then:
          - action: log
            value: "用户已登录，跳过登录步骤"
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
          - action: assert_element_visible
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

1. 确保您使用的是 Python 3.10 或更高版本
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

### 其他问题

**Q: 如何实现数据驱动测试？**

A: 请参考文档中的"变量管理"部分，使用全局变量或测试用例变量来实现数据驱动。

**Q: 如何添加自定义操作？**

A: 可以通过扩展 `test_step_executor.py` 添加新的操作类型，或者使用模块化测试片段封装复杂操作。

## 相关资源

- [录制文档](https://test-crdqcu2hkpbr.feishu.cn/docx/YpIRdgOXGo1CQdxp1cacxwGbnZd?from=from_copylink)
- [Playwright 官方文档](https://playwright.dev/python/docs/intro)
- [PyTest 文档](https://docs.pytest.org/)
