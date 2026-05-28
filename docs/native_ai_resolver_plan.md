# 自研 Python Native Resolver 实施方案

## 1. 结论

最终路线采用 **方案 C：借鉴 Stagehand observe 思路，自研 Python Native Resolver**。

核心原则：

- Python Playwright 仍是唯一确定性执行层。
- AI 只参与语义识别、候选元素选择和 selector 自愈，不直接操作浏览器。
- 不引入 Stagehand SDK 作为运行时依赖。
- 不默认做 Agent；先把单步骤 `ai_step` 和 smart selector 解析做快、稳、可审计。
- 能用 selector registry、显式 selector、启发式规则解决的步骤，不调用 LLM。

## 2. 要解决的真实问题

当前慢的根因不是“缺少 Stagehand”，而是 AI 执行链路里存在这些不稳定因素：

- 自然语言步骤没有先标准化为框架内部动作意图，导致运行时判断路径不够清晰。
- DOM 候选和模型输入已经有压缩能力，但还没有形成独立、可复用的 Native Observe 层。
- `ai_step` 的快路径、DOM LLM 路径、selector registry、自愈能力分散在不同逻辑里，缺少统一 pipeline。
- 缺少同一环境下的时间、token、准确率基准对比，无法判断新方案是否真的更快。
- Stagehand 这类外部 observe 方案会引入 session、CDP、模型密钥、SDK server 等额外变量，不适合作为主路径。

## 3. 非目标

第一阶段不做以下事情：

- 不接入 Stagehand SDK。
- 不新增 Node sidecar。
- 不把 Playwright 执行层迁移到 Node。
- 不让 LLM 直接生成复杂 XPath 或直接执行浏览器动作。
- 不做多步骤 autonomous agent。
- 不重构整个框架目录结构。

## 4. 目标架构

```text
YAML / Step
  ↓
ActionIntent 编译
  ↓
Resolver Pipeline
  1. 显式 selector / element key
  2. Selector Registry
  3. Playwright heuristic
  4. Native Observe: DOM candidates + selector candidates
  5. DOM LLM Resolver
  6. Self-Healing Resolver
  ↓
ResolvedAction
  ↓
Playwright Command Executor
  ↓
报告 / token / selector registry 更新
```

关键边界：

- Resolver 只返回“怎么执行”的结构化结果。
- Executor 只执行框架已有标准 action，例如 `click`、`fill`、`press_key`、`assert_visible`。
- Registry 只沉淀验证成功的 selector，不保存未验证的模型建议。

## 5. 核心数据结构

### 5.1 ActionIntent

`ActionIntent` 表示“用户想做什么”。

```python
@dataclass(frozen=True)
class ActionIntent:
    action: str
    instruction: str
    target: str | None = None
    value: str | None = None
    area: str | None = None
    timeout_ms: int = 10_000
    strict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

来源：

- 标准 YAML step：`action + selector/target/value`
- `ai_step`：`instruction`
- smart 模式失败后的自愈上下文：`action + failed_selector + target`

### 5.2 SelectorCandidate

`SelectorCandidate` 表示一个可尝试的 selector。

```python
@dataclass(frozen=True)
class SelectorCandidate:
    selector: str
    selector_type: str
    score: float
    reason: str | None = None
```

selector 来源优先级：

- `data-testid` / `data-test`
- `id`
- role + accessible name
- label / placeholder
- text
- 稳定 CSS

### 5.3 ResolvedAction

`ResolvedAction` 表示“最终准备怎么执行”。

```python
@dataclass(frozen=True)
class ResolvedAction:
    intent: ActionIntent
    method: str
    selectors: list[SelectorCandidate]
    confidence: float
    source: str
    raw: dict[str, Any] = field(default_factory=dict)
```

执行层按 `score` 从高到低尝试 selector，成功后记录 registry。

## 6. Native Observe 设计

Native Observe 不传完整 DOM，只抽取可交互元素和断言候选。

### 6.1 DOM 候选采集

采集范围：

```text
button
input
textarea
select
a
[role]
[aria-label]
[placeholder]
[contenteditable="true"]
[onclick]
[tabindex]
label
h1/h2/h3
[data-test]
[data-testid]
```

过滤规则：

- 必须可见。
- 尺寸不能为 0。
- `display:none`、`visibility:hidden`、禁用元素默认降权。
- 支持配置 `ignore_selectors`，排除广告、浮层、推荐区、footer 等干扰区域。

### 6.2 a11y 信息处理

不要把 `page.accessibility.snapshot()` 作为硬依赖。

原因：

- Playwright 版本差异会影响可用性和字段稳定性。
- a11y tree 适合作为语义增强，不适合作为唯一输入。

落地策略：

- P0：只用 DOM evaluate 采集可交互元素和语义字段。
- P1：如果当前 Playwright 版本可用，再增加 a11y snapshot 合并层。
- 合并后仍输出统一 candidate 结构，不让 resolver 依赖原始 a11y 字段。

### 6.3 selector candidates 构建

模型不直接写 selector。程序先为每个元素生成 selector 候选：

```json
{
  "id": "e12",
  "tag": "button",
  "role": "button",
  "text": "登录",
  "enabled": true,
  "selector_candidates": [
    "[data-testid='login-button']",
    "button:has-text('登录')",
    "text=登录"
  ]
}
```

模型只返回：

```json
{
  "element_id": "e12",
  "method": "click",
  "confidence": 0.94,
  "reason": "button text matches login intent"
}
```

程序再根据 `element_id` 取回已生成并已验证的 selector。

## 7. Resolver Pipeline

### 7.1 标准步骤 smart 定位

执行顺序：

1. 如果有显式 selector，先验证 selector。
2. 如果 selector 失败且有 target，进入自愈。
3. 查 selector registry。
4. 用 Playwright heuristic 生成 role/text/css 候选。
5. Native Observe 构建候选列表。
6. 候选唯一且高置信度时，无需 LLM。
7. 多候选冲突或低置信度时调用 DOM LLM Resolver。
8. 成功执行后写入 selector registry。

### 7.2 `ai_step`

执行顺序：

1. 本地 fast-path 解析单动作，例如 `Click #cart`、`Fill username with xxx`、`Wait 2s`。
2. 如果 fast-path 命中，直接验证并执行。
3. 否则编译成 `ActionIntent`。
4. 走 Native Observe + DOM LLM Resolver。
5. 返回 `AiStepOperation`，再编译为框架标准 step。
6. 最终仍走 command executor。

硬约束：

- 一个 `ai_step` 只能对应一个原子动作。
- 多动作流程必须拒绝，提示拆成多个 steps 或使用已有 `agent_case`。
- `ai_step` 不直接绕过日志、断言、截图和报告链路。

### 7.3 Self-Healing

触发条件：

- 显式 selector 不可用。
- registry selector 不可用。
- selector 语义不匹配 target。

修复流程：

1. 记录失败 selector 和错误信息。
2. Native Observe 当前页面。
3. DOM LLM 在候选元素中选择最接近 intent 的元素。
4. 对新 selector 做 Playwright 验证。
5. 执行成功后更新 registry。
6. 旧 selector 降权或标记 unstable/deprecated。

## 8. 缓存与成本控制

### 8.1 不调用 LLM 的场景

- 显式 selector 可用。
- registry 高置信命中。
- `data-testid` / `data-test` 唯一命中。
- role/text/placeholder 规则唯一命中。
- `ai_step` fast-path 可解析。
- 简单 wait、press 等非定位动作。

### 8.2 必须调用 LLM 的场景

- 多个候选元素分数接近。
- 文案或布局变化导致 selector 失效。
- 自然语言 target 无法被 heuristic 唯一映射。
- 当前页面是新页面，没有 registry 记录。

### 8.3 缓存 key

建议使用：

```text
project
env
page_key
url_path
action
target/instruction hash
candidate_fingerprint
prompt_version
schema_version
model
```

缓存只能作为建议，必须通过 Playwright 验证后才可执行。

## 9. 报告与可观测性

每个 AI/smart 步骤记录：

- `source`: explicit / registry / heuristic / native_observe / dom_llm / self_healing
- `ai_called`: true/false
- `selector_used`
- `candidate_count`
- `candidate_hash`
- `confidence`
- `prompt_version`
- `schema_version`
- `model`
- `duration_ms`
- `fallback_count`
- `old_selector`
- `new_selector`
- `token_usage`

失败时保留：

- 候选元素 JSON
- 模型输入输出
- 失败 selector 列表
- 截图
- 当前 URL/title

## 10. 基准对比方案

为了证明“简单、更快”，实施前后必须做同环境对比。

### 10.1 环境约束

- 同一台机器。
- 同一浏览器类型和 headed/headless 配置。
- 同一项目、同一 env、同一 base_url。
- 同一测试账号和数据。
- 每轮前清理 `.ui_auto/ai_cache.sqlite3`、`.ui_auto/selectors.db`。
- 每轮前清理相关 token usage 日志和模型 I/O 日志。
- 禁止复用上一轮浏览器上下文。

### 10.2 对比线路

至少跑两条：

- baseline：当前 native/smart 实现。
- new-native-observe：自研 Native Observe + pipeline。

不再把 Stagehand 作为主对比线路。

### 10.3 指标

必须输出：

- 总耗时。
- 每个步骤耗时。
- 通过率。
- 失败步骤和失败原因。
- LLM 调用次数。
- input tokens / output tokens / total tokens。
- registry 命中次数。
- heuristic 命中次数。
- self-healing 成功次数。
- 平均 selector resolution time。

## 11. 实施阶段

### P0：Native Observe 基础闭环

目标：不引入新依赖，先形成可跑通的单动作 resolver。

范围：

- 新增 `ActionIntent` / `ResolvedAction` / `SelectorCandidate` 数据结构。
- 抽出 `NativeObserveCollector`，从页面采集可交互候选。
- 抽出 `SelectorCandidateBuilder`，为元素生成 selector 候选。
- 将 `ai_step` LLM 输出改为只选 `element_id`。
- 保留现有 command executor。
- 增加合约测试和 demo 对比用例。

验收：

- `poetry run pytest tests/test_framework_contracts.py`
- `poetry run python validate_yaml_schema.py`
- `poetry run run_case -p demo -f ai_modes_showcase`
- 输出 baseline vs new-native-observe 对比报告。

### P1：统一 pipeline 和 registry

目标：让 smart selector、ai_step、自愈共用同一候选和缓存逻辑。

范围：

- 把当前分散在 `SmartResolver` 内的 fast-path、heuristic、LLM 选择拆成 resolver pipeline。
- registry 支持一个 action 多 selector 候选及降权。
- self-healing 成功后写回 registry。
- 报告增加 source、confidence、fallback 证据链。

验收：

- 老用例不回归。
- 新旧路径 token 对比下降。
- selector 失效场景可自动修复并持久化。

### P2：a11y 语义增强

目标：在不破坏 P0/P1 的基础上增强页面语义。

范围：

- 检查当前 Playwright 版本是否稳定支持 a11y snapshot。
- 增加可选 a11y collector。
- 合并 DOM + a11y 信号到统一 candidates。
- 增加 `ignore_selectors` 配置。

验收：

- a11y 关闭时行为不变。
- a11y 开启时候选排序更准，token 不明显增加。

## 12. 风险与边界

- LLM 仍可能选错元素，所以所有模型结果必须经过 selector 验证和语义校验。
- DOM 采集脚本可能漏掉 shadow DOM、iframe、canvas 类 UI，第一阶段不承诺覆盖。
- 页面强动态变化会影响 candidate fingerprint，需要以“执行前重新验证”为准。
- token usage 只能统计框架自身 LLM provider，外部不可观测调用不纳入比较。
- 自愈写回必须只写成功验证过的 selector，不能把模型原始建议直接持久化。

## 13. 待确认事项

请确认以下默认取舍：

1. 第一阶段只做 P0，不碰 Stagehand、不做 sidecar、不做 Agent。
2. `ai_step` 继续要求单原子动作，多动作指令直接 reject。
3. selector registry 默认开启，但基准对比时每轮清空，保证环境纯粹。
4. a11y snapshot 放到 P2，不作为 P0 必需能力。
5. 新方案先覆盖 demo 的 `ai_modes_showcase`，通过后再扩展到其它项目。

