# AI 生成的测试用例，为什么不能直接落盘？

最近在整理 AI Playwright 的用例生成链路时，我把两个看起来很方便的选项直接废掉了：`--dry-run` 和 `--no-verify`。

现在执行 `gen`，生成结果必须跑一次真实页面验证。模型写出的 YAML 即使格式正确，也不能直接进入正式的 `cases/`、`data/` 和 `elements/` 目录。

这项限制会让生成过程变慢一些，也会让实现复杂不少。但在自动化测试里，我更担心另一件事：一批结构完整、命名规范、甚至能通过 Schema 校验的用例，被过早当成可信资产。

它们看起来已经完成，实际上可能还没有在页面上走过一步。

## Schema 通过，只能说明“它像一条用例”

AI 很擅长生成结构完整的内容。给它一段业务描述，它可以很快给出测试名称、步骤、元素和断言。

以 AI Playwright 的 YAML 为例，一条用例可能长这样：

```yaml
test_data:
  test_add_product_to_cart:
    mode: smart
    steps:
      - action: click
        target: Sauce Labs Backpack 的加入购物车按钮
      - action: assert_text
        selector: shopping_cart_badge
        value: "1"
```

这段 YAML 可以被解析，字段也可能符合 Schema。但它仍然留下了几个没有回答的问题：

- `target` 能不能在当前页面找到对应元素？
- 点击后页面是否真的发生了预期变化？
- 断言是否验证了业务结果，还是只验证了某个无关文本？
- 当前环境、账号和数据是否允许这条路径成立？

Schema 负责检查资产结构。它无法替浏览器回答这些问题。

所以在生成链路里，我加了另一道更早的门：如果候选用例没有有效断言，它连真实页面验证都不能进入。

核心检查并不复杂：

```python
def _assert_effective_verification_payload(context, payload):
    for case in payload.get("cases") or []:
        case_data = (payload.get("data") or {}).get(case["name"], {})
        if not _steps_have_effective_assertion(
            case_data.get("steps") or [],
            payload.get("modules") or {},
            seen=set(),
        ):
            raise ValueError("生成用例缺少有效信息断言")
```

这里的重点不是要求每条用例多写几个 `assert`。点击、输入和跳转只能证明动作被执行；至少要有一项可观察结果，才能说明这条路径准备验证什么。

## 候选结果和正式资产必须分开

我没有让模型直接修改正式测试目录，而是先复制一份项目上下文，建立独立的候选目录。

候选 Payload 会经历：

1. 规范化；
2. Schema 校验；
3. 有效断言检查；
4. 写入候选 YAML；
5. pytest 和 Playwright 真实页面执行。

只有候选执行通过后，系统才会把它写入正式资产，并在正式位置再执行一次。

```python
def _verify_candidate_persist_formal(...):
    _assert_effective_verification_payload(context, payload)
    _write_and_verify_candidate(...)

    _write_payload(
        result,
        overwrite=overwrite,
        verify=lambda: _validate_written_case_file(result),
        post_verify=lambda: _verify_generated_case(
            context=context,
            env=env,
            result=result,
            stage="正式存储后",
        ),
    )
```

这段实现真正划分的是资产权威。

候选目录里的文件可以失败、被修复、被删除。正式目录里的文件会被 pytest 收集，会进入代码评审和 CI，也会影响后续回归测试。两者不应拥有相同地位。

[图 1｜AI Playwright 的 verification-first 用例生成链路。回答：候选结果需要经过哪些门，才能进入正式测试资产。]

## 验证时，我反而关闭了 AI 自愈

候选用例的真实执行阶段会把 `UI_AI_MODE` 设置为 `strict`。

```python
os.environ["UI_AI_MODE"] = "strict"
exit_code = pytest.main(
    _verification_pytest_args(case_file, browser=browser_name)
)
```

这看起来有些反直觉。既然框架支持智能定位和运行时 Agent，为什么验证生成结果时不用它们帮忙？

因为这里要验证的是刚刚生成的正式资产能不能独立工作，而不是模型能不能在运行时把错误掩盖过去。

如果生成步骤写错了元素，运行时又临时猜中另一个选择器，最终结果可能通过，但代码库里留下的仍然是错误资产。下一次关闭模型、切换环境或执行回归时，问题还会出现。

生成阶段可以使用模型理解自然语言。准入验证阶段需要尽量确定，明确检查提交到仓库的内容本身。

这也是我现在理解“测试前移”的一个具体方式：不是在需求评审会上多说几句测试意见，而是把可执行验证直接嵌进资产生成事务里。

## 失败不能只返回一句“生成失败”

真实页面验证失败后，系统会保留候选目录、pytest 输出和生成过程中的中间产物。它也允许模型基于具体错误修复候选，再重新走一遍规范化和验证。

最近两次修正还处理了一个更细的问题：

- 自然语言目标要绑定到稳定的语义元素键，而不是把临时描述直接当 Selector；
- 浏览器已经验证过的 Selector 修复，即使候选最终失败，也不应被全部丢弃。

这两项改动分别落在 `328222e` 和 `98d17f7` 两个提交里。

失败在这里不是一个布尔值。它至少应该留下：失败发生在哪一步、浏览器实际看到了什么、哪些 Selector 已经被证明可用、下一次修复从哪里继续。

否则 AI 每次都从头猜一遍，生成速度再快，也只是更快地重复同一类错误。

## 真实页面通过，也只是拿到了入场券

浏览器执行成功很重要，但我不认为它能证明一条用例已经完整可信。

一次通过只能说明：在当前代码、当前页面、当前账号和当前数据条件下，这条路径得到了预期结果。

它不能自动证明：

- 需求已经覆盖完整；
- 边界和异常路径没有遗漏；
- 断言选择的是最重要的业务结果；
- 这条用例不会与其他资产重复；
- 打包安装后仍然可用。

所以仓库 CI 还会继续执行编译、格式检查、框架契约测试、YAML Schema 校验、重复定义检查、pytest 用例收集和安装包冒烟测试。

真实浏览器验证解决“这条路径能不能跑”。仓库质量门继续回答“这项变更能不能被长期维护”。

## 我现在更关心的不是生成数量

过去谈 AI 自动化测试，最容易展示的是模型一次生成了多少用例。这个结果直观，也容易做 Demo。

但生成数量不是我现在最关心的指标。

我更关心的是，一条候选结果怎样获得正式资产的资格：它有没有明确断言，能不能在真实页面执行，失败是否留下证据，修复后的选择器能不能被复用，写入代码库后是否还能通过完整 CI。

这条链路会比“自然语言直接生成脚本”慢，也不够炫。但它更接近真实的软件交付。

下一次把 AI 接进代码或测试资产生成流程时，可以先检查一个问题：

> 模型输出之后、正式写入之前，系统要求它拿出什么证据？

如果这个阶段什么都没有，后面的测试很可能只能负责清理已经进入代码库的问题。
