# 自研 Python Native Resolver 技术方案

## 1. 方案定位

本方案用于构建一套面向 Web UI 自动化测试的 **Python Native Resolver 架构**。

目标不是复刻 Stagehand、Midscene 或通用浏览器 Agent，而是建设一套更适合测试框架的能力：

- **Playwright Python 作为唯一确定性执行层**
- **AI 只参与语义解析、候选选择、selector 自愈**
- **所有 AI 输出必须结构化、可验证、可审计**
- **所有成功执行的定位经验必须沉淀到 selector registry**
- **优先使用显式 selector、registry、规则启发式，必要时才调用 LLM**
- **不引入 Stagehand SDK、不引入 Node sidecar、不把执行层迁移到 Node**
- **不做多步骤 autonomous agent，先把单步骤智能解析做到稳定、低成本、可复用**

最终框架形态：

```text
YAML / Python DSL / ai_step
  ↓
ActionIntent
  ↓
Resolver Pipeline
  ↓
ResolvedAction
  ↓
Playwright Command Executor
  ↓
Report / Trace / Token Usage / Selector Registry
```

一句话：

> AI 负责判断“哪个元素最符合意图”，框架负责决定“如何稳定执行、如何验证、如何沉淀”。

---

## 2. 核心原则

### 2.1 AI 不直接操作浏览器

禁止以下行为：

- 让 LLM 直接点击页面
- 让 LLM 直接生成复杂 XPath 并执行
- 让 LLM 控制多步骤流程
- 让 LLM 绕过框架已有日志、截图、报告和失败处理链路

允许 AI 做的事情：

- 从候选元素中选择最符合意图的 `element_id`
- 判断多个候选元素的语义相关性
- 在 selector 失效时，从当前页面候选中选择替代元素
- 为失败原因提供结构化解释
- 辅助生成标准动作，不直接执行动作

### 2.2 执行必须确定性

真正操作浏览器的永远是 Playwright Command Executor。

所有动作必须落到标准命令：

```text
click
fill
select
check
uncheck
press
hover
wait_for
assert_visible
assert_hidden
assert_text
assert_url
extract_text
extract_table
```

### 2.3 模型只选择，不创作 selector

LLM 不直接写 selector。

正确流程：

```text
Native Observe Collector
  ↓
ElementCandidate
  ↓
SelectorCandidateBuilder
  ↓
LLM 选择 element_id
  ↓
框架取回 selector candidates
  ↓
Playwright 验证 selector
  ↓
执行成功后写入 registry
```

### 2.4 成功结果才入库

Selector Registry 只保存经过 Playwright 实际验证并成功执行的 selector。

禁止保存：

- 未执行的模型建议
- 未验证的 selector
- 单纯看起来合理的 XPath
- 坐标点击结果
- 低置信度候选

### 2.5 一个 ai_step 只允许一个原子动作

`ai_step` 是智能单步，不是 Agent。

允许：

```yaml
- ai_step: 点击登录按钮
- ai_step: 在手机号输入框输入 13800138000
- ai_step: 断言页面出现提交成功
```

拒绝：

```yaml
- ai_step: 登录系统并进入订单列表搜索北京订单
```

复杂流程必须拆成多个步骤。

---

## 3. 总体架构

```text
┌────────────────────────────────────────────┐
│               Test Case Layer              │
│      YAML / Python DSL / ai_step           │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│              Intent Compiler               │
│        Step → ActionIntent                  │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│             Resolver Pipeline              │
│                                            │
│  1. ExplicitSelectorResolver                │
│  2. SelectorRegistryResolver                │
│  3. PlaywrightHeuristicResolver             │
│  4. NativeObserveResolver                   │
│  5. DomLLMResolver                          │
│  6. SelfHealingResolver                     │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│              ResolvedAction                 │
│      method + selector candidates           │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│          Playwright Command Executor        │
│      validate → execute → retry → trace     │
└─────────────────────┬──────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│          Report / Registry / Metrics        │
│   selector history / token / screenshot     │
└────────────────────────────────────────────┘
```

---

## 4. 目录结构建议

不需要大规模重构已有框架，但建议新增清晰的 native resolver 模块边界。

```text
ui_auto/
├── core/
│   ├── intent.py
│   ├── resolved_action.py
│   ├── selector_candidate.py
│   ├── errors.py
│   └── types.py
│
├── observe/
│   ├── native_observe_collector.py
│   ├── element_candidate.py
│   ├── selector_candidate_builder.py
│   ├── context_extractor.py
│   ├── candidate_fingerprint.py
│   └── ignore_filter.py
│
├── resolver/
│   ├── pipeline.py
│   ├── base.py
│   ├── explicit_selector_resolver.py
│   ├── registry_resolver.py
│   ├── heuristic_resolver.py
│   ├── native_observe_resolver.py
│   ├── dom_llm_resolver.py
│   └── self_healing_resolver.py
│
├── executor/
│   ├── command_executor.py
│   ├── selector_validator.py
│   ├── locator_adapter.py
│   ├── retry_policy.py
│   └── wait_policy.py
│
├── registry/
│   ├── selector_registry.py
│   ├── registry_storage.py
│   ├── scoring.py
│   └── models.py
│
├── llm/
│   ├── client.py
│   ├── schema.py
│   ├── prompts/
│   │   ├── resolve_element.md
│   │   └── heal_selector.md
│   └── providers/
│       ├── openai_provider.py
│       ├── doubao_provider.py
│       └── local_provider.py
│
├── report/
│   ├── step_reporter.py
│   ├── ai_trace_writer.py
│   ├── token_usage.py
│   └── artifacts.py
│
└── dsl/
    ├── step_compiler.py
    ├── ai_step_parser.py
    └── schema.py
```

如果当前框架已有同类目录，不需要照搬新目录，只需要保证职责边界一致。

---

## 5. 核心数据结构

### 5.1 ActionIntent

`ActionIntent` 表示用户想做什么。

```python
from dataclasses import dataclass, field
from typing import Any, Literal

ActionName = Literal[
    "click",
    "fill",
    "select",
    "check",
    "uncheck",
    "press",
    "hover",
    "wait_for",
    "assert_visible",
    "assert_hidden",
    "assert_text",
    "assert_url",
    "extract_text",
    "extract_table",
]

@dataclass(frozen=True)
class ActionIntent:
    action: ActionName
    instruction: str

    target: str | None = None
    value: str | None = None
    area: str | None = None

    # assertion support
    expected: Any | None = None
    operator: str | None = None      # equals / contains / regex / exists / not_exists
    index: int | None = None         # when user explicitly asks first/second/nth

    timeout_ms: int = 10_000
    strict: bool = False

    source_step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

示例：

```python
ActionIntent(
    action="click",
    instruction="点击登录按钮",
    target="登录按钮",
)
```

```python
ActionIntent(
    action="fill",
    instruction="在手机号输入框输入 13800138000",
    target="手机号输入框",
    value="13800138000",
)
```

```python
ActionIntent(
    action="assert_text",
    instruction="断言订单状态是已支付",
    target="订单状态",
    expected="已支付",
    operator="contains",
)
```

---

### 5.2 ElementCandidate

`ElementCandidate` 是 Native Observe 采集到的页面候选元素。

```python
@dataclass(frozen=True)
class ElementCandidate:
    element_id: str

    tag: str
    role: str | None = None
    type: str | None = None

    text: str | None = None
    inner_text: str | None = None
    accessible_name: str | None = None
    aria_label: str | None = None
    placeholder: str | None = None
    label: str | None = None
    title: str | None = None

    id_attr: str | None = None
    name_attr: str | None = None
    class_name: str | None = None
    test_id: str | None = None
    href: str | None = None

    visible: bool = True
    enabled: bool = True
    disabled: bool = False

    rect: dict[str, int] = field(default_factory=dict)

    # context
    near_text: list[str] = field(default_factory=list)
    parent_text: str | None = None
    section_heading: str | None = None
    form_labels: list[str] = field(default_factory=list)
    row_text: str | None = None

    # frame / shadow support
    frame_id: str | None = None
    frame_url: str | None = None
    in_shadow: bool = False

    selector_candidates: list["SelectorCandidate"] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
```

重点字段：

| 字段 | 用途 |
|---|---|
| `text` | 文本按钮、链接、断言 |
| `accessible_name` | role 定位 |
| `placeholder` | 输入框定位 |
| `label` | 表单项定位 |
| `near_text` | 多个同名按钮时判断上下文 |
| `row_text` | 表格行内按钮定位 |
| `section_heading` | 页面区域识别 |
| `frame_id` | iframe 内元素执行 |
| `selector_candidates` | 程序生成的可验证 selector |

---

### 5.3 SelectorCandidate

```python
@dataclass(frozen=True)
class SelectorCandidate:
    selector: str
    selector_type: str       # testid / id / role / label / placeholder / text / css / xpath
    score: float

    reason: str | None = None
    source: str | None = None    # builder / registry / heuristic / healing

    is_verified: bool = False
    match_count: int | None = None
    visible_count: int | None = None
    last_error: str | None = None
```

selector 优先级建议：

```text
data-testid / data-test
  > role + accessible name
  > label
  > placeholder
  > stable id
  > text
  > stable css
  > xpath fallback
```

避免优先使用：

- 动态 class
- 过长 CSS 路径
- 复杂 XPath
- 坐标
- index-only selector

---

### 5.4 ResolvedAction

```python
@dataclass(frozen=True)
class ResolvedAction:
    intent: ActionIntent
    method: str

    selectors: list[SelectorCandidate]
    confidence: float

    source: str
    # explicit / registry / heuristic / native_observe / dom_llm / self_healing

    selected_element_id: str | None = None
    frame_id: str | None = None

    ai_called: bool = False
    llm_skip_reason: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)
```

执行层只接收 `ResolvedAction`，不关心它来自显式 selector、registry、heuristic 还是 LLM。

---

## 6. Native Observe 设计

### 6.1 采集目标

Native Observe 只采集对测试有价值的元素，不传完整 DOM。

采集范围：

```text
button
input
textarea
select
a
label
h1 / h2 / h3
table / tr / td / th
[role]
[aria-label]
[placeholder]
[contenteditable="true"]
[onclick]
[tabindex]
[data-test]
[data-testid]
```

### 6.2 过滤规则

默认过滤：

- `display: none`
- `visibility: hidden`
- 尺寸为 0
- 不在 viewport 且不可滚动定位
- 被 `ignore_selectors` 命中的区域
- 纯装饰节点
- 重复空文本节点

禁用元素不直接丢弃，但降权：

```text
disabled = true
enabled = false
score penalty
```

原因：断言场景可能需要检查按钮禁用状态。

---

### 6.3 上下文提取

上下文是解决多个同名元素的关键。

对每个 candidate 提取：

```text
near_text
parent_text
section_heading
form_labels
row_text
dialog_title
breadcrumb
```

#### 示例：筛选表单里的“查询”

```json
{
  "id": "e12",
  "tag": "button",
  "text": "查询",
  "context": {
    "section_heading": "订单管理",
    "parent_text": "订单编号 手机号 订单状态 查询 重置",
    "form_labels": ["订单编号", "手机号", "订单状态"],
    "near_text": ["订单编号", "手机号", "订单状态"]
  }
}
```

#### 示例：表格行里的“删除”

```json
{
  "id": "e33",
  "tag": "button",
  "text": "删除",
  "context": {
    "row_text": "订单号 10001 用户 张三 状态 待支付 编辑 删除"
  }
}
```

---

### 6.4 iframe 支持

P0 就建议支持基础 iframe，不要完全后置。

策略：

```text
遍历 page.frames
每个 frame 单独执行 collector
candidate 增加 frame_id / frame_url / frame_name
执行时通过 frame locator 或 frame object 执行 selector
```

candidate ID 格式：

```text
main:e12
frame_1:e4
frame_2:e9
```

示例：

```json
{
  "id": "frame_1:e4",
  "frame_id": "frame_1",
  "frame_url": "/order/search",
  "tag": "button",
  "text": "查询"
}
```

### 6.5 Shadow DOM 支持

支持 open shadow DOM，暂不承诺 closed shadow DOM。

策略：

```text
递归扫描 open shadowRoot
candidate 标记 in_shadow=true
selector builder 尽量生成可执行路径
如果无法生成稳定 selector，则降级为普通候选，不写 registry
```

---

## 7. SelectorCandidateBuilder

### 7.1 生成规则

对每个 `ElementCandidate` 生成多个 selector candidates。

优先生成：

```text
[data-testid="xxx"]
[data-test="xxx"]
get_by_role(role, name=accessible_name)
get_by_label(label)
get_by_placeholder(placeholder)
#id
text selector
stable css selector
```

### 7.2 稳定性评分

基础分：

| 类型 | 初始分 |
|---|---:|
| data-testid / data-test | 0.98 |
| role + accessible name | 0.92 |
| label | 0.90 |
| placeholder | 0.86 |
| stable id | 0.82 |
| exact text | 0.76 |
| css | 0.65 |
| xpath | 0.50 |

扣分项：

| 情况 | 扣分 |
|---|---:|
| selector 命中多个元素 | -0.15 |
| 元素不可见 | -0.25 |
| 元素 disabled | -0.20 |
| text 过短 | -0.10 |
| class 疑似 hash | -0.20 |
| selector 路径过长 | -0.15 |
| 需要 nth/index | -0.20 |

---

## 8. Resolver Pipeline

### 8.1 Pipeline 顺序

```text
ActionIntent
  ↓
ExplicitSelectorResolver
  ↓
SelectorRegistryResolver
  ↓
PlaywrightHeuristicResolver
  ↓
NativeObserveResolver
  ↓
DomLLMResolver
  ↓
SelfHealingResolver
  ↓
ResolvedAction
```

### 8.2 ExplicitSelectorResolver

触发条件：

- YAML step 明确提供 selector
- Python DSL 明确指定 locator
- ai_step fast-path 解析出明确 selector

处理逻辑：

```text
验证 selector 是否存在
验证 selector 是否可见 / 可交互
验证 action 是否匹配
成功则返回 ResolvedAction
失败则进入 self-healing
```

### 8.3 SelectorRegistryResolver

触发条件：

- 当前 page_key / url_path / action / target 能命中历史记录

处理逻辑：

```text
按 registry score 排序
逐个验证 selector
高置信 selector 成功则直接返回
失败则记录 failure_count
连续失败则降权或 deprecated
```

### 8.4 PlaywrightHeuristicResolver

不调用 LLM，直接通过规则生成候选。

示例：

```text
click 登录按钮
  → get_by_role("button", name="登录")
  → text=登录

fill 手机号输入框
  → get_by_label("手机号")
  → get_by_placeholder("请输入手机号")
  → input[name*="phone"]
```

命中条件：

```text
唯一匹配
可见
可交互
action 类型匹配
```

### 8.5 NativeObserveResolver

采集页面候选元素并生成 selector candidates。

如果出现高置信唯一候选，可不调用 LLM：

```text
data-testid 唯一命中
role/name 唯一命中
placeholder 唯一命中
label 唯一命中
```

否则交给 DomLLMResolver。

### 8.6 DomLLMResolver

输入：

```json
{
  "task": {
    "action": "click",
    "instruction": "点击登录按钮",
    "target": "登录按钮",
    "area": "登录表单"
  },
  "page": {
    "url": "/login",
    "title": "登录"
  },
  "elements": [
    {
      "id": "e1",
      "tag": "input",
      "role": "textbox",
      "placeholder": "请输入手机号",
      "label": "手机号",
      "selector_summary": ["placeholder", "label"]
    },
    {
      "id": "e2",
      "tag": "button",
      "role": "button",
      "text": "登录",
      "near_text": ["手机号", "验证码"],
      "selector_summary": ["role", "text"]
    }
  ]
}
```

输出：

```json
{
  "element_id": "e2",
  "method": "click",
  "confidence": 0.94,
  "reason": "button text and nearby login form context match the intent"
}
```

硬性规则：

- 只能选择输入中的 `element_id`
- 不能生成 selector
- 不能输出多步动作
- 不能返回自然语言
- 低置信时返回 `null`

### 8.7 SelfHealingResolver

触发条件：

- 显式 selector 失败
- registry selector 失败
- selector 命中元素但语义不匹配
- 执行动作超时

输入：

```json
{
  "intent": {
    "action": "click",
    "target": "提交按钮",
    "instruction": "点击提交按钮"
  },
  "failed": {
    "selector": "button:has-text('提交')",
    "error": "Timeout waiting for visible"
  },
  "current_candidates": [...]
}
```

输出：

```json
{
  "element_id": "e8",
  "method": "click",
  "confidence": 0.91,
  "change_type": "text_changed",
  "reason": "original submit button appears to be renamed to 确认提交"
}
```

成功后：

```text
新 selector 写入 registry
旧 selector failure_count + 1
旧 selector 降权或标记 unstable
报告记录 old_selector / new_selector
```

---

## 9. ai_step 处理规则

### 9.1 ai_step 编译

`ai_step` 先进入本地 fast-path。

明确语法直接解析：

```text
click <selector>
fill <selector> with <value>
press <key>
wait <duration>
assert visible <selector>
```

示例：

```yaml
- ai_step: click #login
- ai_step: fill input[name=phone] with 13800138000
- ai_step: wait 2s
```

如果不是明确语法，则编译为 `ActionIntent`：

```yaml
- ai_step: 点击登录按钮
```

编译结果：

```python
ActionIntent(
    action="click",
    instruction="点击登录按钮",
    target="登录按钮",
)
```

### 9.2 多动作拒绝

拒绝例子：

```yaml
- ai_step: 输入手机号和验证码，然后点击登录
```

提示：

```text
ai_step 只支持一个原子动作，请拆成多个 steps：
1. 输入手机号
2. 输入验证码
3. 点击登录
```

### 9.3 ai_step 不绕过标准链路

`ai_step` 最终必须生成标准 step 或 `ResolvedAction`，再进入 Command Executor。

---

## 10. Selector Registry

### 10.1 存储对象

```python
@dataclass
class RegistrySelectorRecord:
    project: str
    env: str
    page_key: str
    url_path: str

    action: str
    target_hash: str
    target_text: str | None

    selector: str
    selector_type: str

    score: float
    status: str   # active / unstable / deprecated

    success_count: int = 0
    failure_count: int = 0

    last_success_at: str | None = None
    last_failure_at: str | None = None

    source: str | None = None
    created_by: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
```

### 10.2 写入规则

只写入：

```text
Playwright 验证成功
动作执行成功
语义与 intent 匹配
```

不写入：

```text
未执行的候选
LLM 原始输出
坐标点击
低置信 selector
失败 selector
```

### 10.3 降权规则

推荐初始公式：

```python
score = base_score \
    + min(success_count, 10) * 0.02 \
    - min(failure_count, 10) * 0.05 \
    - non_unique_penalty \
    - stale_penalty
```

状态规则：

| 条件 | 状态 |
|---|---|
| 成功执行 | active |
| 最近失败 1 次但历史成功 | unstable |
| 连续失败 ≥ 3 次 | deprecated |
| 命中多个可见元素 | unstable |
| 新 selector 自愈成功 | active |

---

## 11. 缓存与成本控制

### 11.1 不调用 LLM 的场景

```text
显式 selector 可用
registry 高置信命中
data-testid / data-test 唯一命中
role + name 唯一命中
label / placeholder 唯一命中
ai_step fast-path 可解析
wait / press / assert_url 等非定位动作
```

### 11.2 调用 LLM 的场景

```text
多个候选分数接近
自然语言 target 无法唯一映射
selector 失效需要自愈
新页面没有 registry
表格行内同名按钮需要语义判断
弹窗和主页面存在同名按钮
```

### 11.3 缓存 key

```text
project
env
page_key
url_path
action
target_hash
candidate_fingerprint
collector_version
selector_builder_version
prompt_version
schema_version
model
```

缓存命中后仍必须执行 Playwright 验证。

### 11.4 输入压缩策略

模型输入只保留：

```text
task
page url/title
element id
tag
role
text
label
placeholder
aria label
enabled/visible
near_text
row_text
section_heading
selector_summary
```

不传：

```text
完整 HTML
完整 CSS
完整 JS
隐藏 DOM
大段 innerHTML
无关列表
广告区
footer
推荐区
```

---

## 12. Command Executor

### 12.1 执行流程

```text
ResolvedAction
  ↓
按 selector score 排序
  ↓
逐个验证 selector
  ↓
匹配唯一性检查
  ↓
执行 action
  ↓
成功则写 registry
  ↓
失败则 fallback / self-healing
```

### 12.2 selector 验证

验证内容：

```text
selector 是否可解析
match_count
visible_count
是否在正确 frame
是否可交互
是否与 action 类型匹配
```

### 12.3 执行伪代码

```python
class PlaywrightCommandExecutor:
    async def execute(self, page, action: ResolvedAction):
        errors = []

        for candidate in sorted(action.selectors, key=lambda x: x.score, reverse=True):
            try:
                validation = await self.validator.validate(page, candidate, action)
                if not validation.ok:
                    errors.append(validation.error)
                    continue

                locator = validation.locator

                if action.method == "click":
                    await locator.click(timeout=action.intent.timeout_ms)

                elif action.method == "fill":
                    await locator.fill(action.intent.value or "", timeout=action.intent.timeout_ms)

                elif action.method == "press":
                    await locator.press(action.intent.value or "Enter")

                elif action.method == "assert_visible":
                    await locator.wait_for(state="visible", timeout=action.intent.timeout_ms)

                elif action.method == "assert_text":
                    await expect(locator).to_contain_text(action.intent.expected)

                else:
                    raise UnsupportedActionError(action.method)

                await self.registry.record_success(action, candidate)
                return ExecutionResult.success(candidate)

            except Exception as e:
                errors.append({
                    "selector": candidate.selector,
                    "error": str(e),
                })

        return await self.self_healing.try_heal(page, action, errors)
```

---

## 13. 报告与可观测性

每个 step 必须记录：

```json
{
  "step_id": "s12",
  "instruction": "点击登录按钮",
  "action": "click",
  "source": "registry",
  "ai_called": false,
  "llm_skip_reason": "registry_high_confidence",
  "selector_used": "button:has-text('登录')",
  "selector_type": "text",
  "candidate_count": 8,
  "confidence": 0.93,
  "duration_ms": 421,
  "fallback_count": 0,
  "token_usage": null
}
```

AI 调用时记录：

```json
{
  "source": "dom_llm",
  "ai_called": true,
  "model": "doubao-xxx",
  "prompt_version": "resolve_element_v1",
  "schema_version": "element_choice_v1",
  "candidate_hash": "abc123",
  "input_tokens": 842,
  "output_tokens": 48,
  "selected_element_id": "e12",
  "confidence": 0.94
}
```

Self-healing 时记录：

```json
{
  "source": "self_healing",
  "old_selector": "button:has-text('提交')",
  "new_selector": "button:has-text('确认提交')",
  "change_type": "text_changed",
  "healing_confidence": 0.91
}
```

失败时保留 artifacts：

```text
screenshot
dom_candidates.json
llm_input.json
llm_output.json
failed_selectors.json
trace.zip
current_url.txt
```

---

## 14. 基准对比

为了验证方案是否更快、更稳、更省 token，必须做同环境基准。

### 14.1 固定环境

```text
同一机器
同一浏览器
同一 headed/headless 模式
同一账号
同一测试数据
同一 base_url
同一模型配置
不复用浏览器上下文
```

### 14.2 对比线路

```text
baseline-current
new-native-cold
new-native-warm
new-native-broken-selector
```

含义：

| 线路 | 说明 |
|---|---|
| baseline-current | 当前实现 |
| new-native-cold | 清空 registry/cache 后首次运行 |
| new-native-warm | 保留 registry/cache 后再次运行 |
| new-native-broken-selector | 注入失效 selector，测试自愈能力 |

### 14.3 指标

```text
总耗时
每步骤耗时
通过率
失败步骤
失败原因
LLM 调用次数
input tokens
output tokens
total tokens
registry 命中次数
heuristic 命中次数
self-healing 触发次数
self-healing 成功次数
平均 selector resolution time
warm run 耗时下降比例
warm run token 下降比例
```

---

## 15. 开发落地顺序

不人为拆过多阶段，但为了 AI 开发高效，建议按下面顺序一次性推进。

### 15.1 基础结构

先实现：

```text
ActionIntent
ElementCandidate
SelectorCandidate
ResolvedAction
ExecutionResult
```

### 15.2 Native Observe

实现：

```text
NativeObserveCollector
ContextExtractor
IgnoreFilter
CandidateFingerprint
```

覆盖：

```text
主文档
iframe
open shadow DOM
可交互元素
断言元素
表格行上下文
表单上下文
弹窗上下文
```

### 15.3 Selector Candidate Builder

实现：

```text
data-testid
data-test
role/name
label
placeholder
id
text
stable css
xpath fallback
```

并实现 selector validation。

### 15.4 Resolver Pipeline

实现统一 pipeline：

```text
explicit
registry
heuristic
native_observe
dom_llm
self_healing
```

### 15.5 ai_step 接入

实现：

```text
fast-path
single action detection
multi-action rejection
ActionIntent compiler
```

### 15.6 Registry

实现：

```text
selector storage
success/failure update
score calculation
active/unstable/deprecated status
```

### 15.7 报告和指标

实现：

```text
step report
AI trace
token usage
selector resolution metrics
failure artifacts
benchmark report
```

---

## 16. 验收标准

### 16.1 功能验收

必须支持：

```yaml
- click: 登录按钮
- fill:
    target: 手机号输入框
    value: 13800138000
- press: Enter
- assert_visible: 首页
- assert_text:
    target: 订单状态
    expected: 已支付
```

必须支持：

```yaml
- ai_step: 点击登录按钮
- ai_step: 在手机号输入框输入 13800138000
- ai_step: 断言页面出现提交成功
```

必须拒绝：

```yaml
- ai_step: 输入手机号和验证码然后点击登录
```

### 16.2 稳定性验收

```text
显式 selector 可用时不调用 LLM
registry 命中时不调用 LLM
高置信唯一候选时不调用 LLM
LLM 输出非法 element_id 时直接失败
LLM 输出低置信时不执行
selector 失败后能触发 self-healing
self-healing 成功后 registry 更新
```

### 16.3 报告验收

每个步骤必须能回答：

```text
这一步为什么这样定位？
有没有调用 AI？
如果没调用 AI，原因是什么？
如果调用 AI，输入输出是什么？
最终用了哪个 selector？
selector 从哪里来？
失败时有哪些候选？
token 花了多少？
```

---

## 17. 配置建议

```yaml
ai_resolver:
  enabled: true

  model:
    provider: doubao
    name: doubao-xxx
    temperature: 0
    max_output_tokens: 256

  native_observe:
    enabled: true
    include_iframes: true
    include_open_shadow_dom: true
    max_candidates: 80
    max_text_length: 120
    ignore_selectors:
      - ".ads"
      - ".recommend"
      - "footer"
      - "[data-testid='floating-chat']"

  llm:
    call_when_ambiguous: true
    min_confidence_to_execute: 0.80
    min_confidence_to_cache: 0.90

  registry:
    enabled: true
    storage: ".ui_auto/selectors.db"
    min_score_to_use: 0.75
    deprecated_after_failures: 3

  report:
    save_ai_io: true
    save_candidates_on_failure: true
    save_screenshot_on_failure: true
    save_trace_on_failure: true
```

---

## 18. 不接 Stagehand / Midscene 的理由

当前主线不接 Stagehand / Midscene。

原因：

```text
引入额外运行时变量
增加 session / sidecar / SDK server 复杂度
削弱 Python Playwright 主执行层的一致性
不利于 selector registry 沉淀
不利于统一报告和成本统计
不符合当前只做 Web UI 自动化测试的目标
```

未来如需接入，只能作为可插拔 fallback provider：

```text
VisionFallbackResolver
MidsceneResolver
```

触发条件：

```text
DOM candidates 为空
页面是 canvas / 复杂 SVG
低代码平台 DOM 质量极差
普通 resolver 连续失败
```

但不进入主链路，不作为默认能力。

---

## 19. 关键实现注意事项

### 19.1 不要让 fast-path 变成隐形 Agent

fast-path 只解析明确语法。

例如：

```text
click #login
fill input[name=phone] with 13800138000
wait 2s
```

自然语言仍进入 ActionIntent。

### 19.2 不要过度相信 text selector

`text=登录` 很容易匹配多个元素。

必须验证：

```text
match_count
visible_count
可交互性
上下文
```

### 19.3 不要把坐标点击写入 registry

坐标只能作为极端兜底，不作为长期资产。

### 19.4 不要让缓存绕过验证

缓存命中只能减少 LLM 调用，不能跳过 Playwright 验证。

### 19.5 模型输出必须严格校验

校验项：

```text
JSON schema
element_id 是否存在
method 是否匹配 intent
confidence 是否达标
是否为单动作
```

---

## 20. 最终推荐实现目标

最终你要得到的不是一个“AI 点网页”的工具，而是一套：

```text
确定性执行
语义化编写
智能定位
失败自愈
成本可控
报告可审计
经验可沉淀
```

的 Python UI 自动化测试框架。

核心资产是：

```text
ActionIntent 标准化
Native Observe 候选采集
SelectorCandidate 生成与验证
Resolver Pipeline
Selector Registry
Self-Healing
AI Trace Report
Benchmark Metrics
```

这套架构比直接接 Stagehand 或 Midscene 更适合你的目标：  
**稳定测试、低 token 成本、Python 技术栈、可持续演进、可审计报告、可沉淀 selector 资产。**
