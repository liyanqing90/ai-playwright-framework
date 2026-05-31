import json
import os
import sys
import types
from pathlib import Path
from typing import Generator, Any

import pytest
from _pytest.python import Module
from playwright.sync_api import Page, Browser, sync_playwright

from ai_playwright.constants import DEFAULT_TIMEOUT
from ai_playwright.page_objects.base_page import BasePage
from ai_playwright.ai_runtime.element_store import wait_for_pending_element_updates
from ai_playwright.case_utils import run_test_data, load_test_cases, load_modules
from ai_playwright.runner import TestCaseGenerator
from ai_playwright.step_actions.step_executor import StepExecutor
from ai_playwright.step_actions.step_executor import (
    commit_pending_selector_cache,
    discard_pending_selector_cache,
)
from ai_playwright.yaml_schema import validate_pytest_targets
from ai_playwright.yaml_schema import YamlSchemaValidationError
from ai_playwright.utils.config import Config
from ai_playwright.utils.logger import configure_file_logger, logger
from ai_playwright.utils.token_usage import (
    format_token_usage_summary,
    get_token_usage_tracker,
)

config = Config()


def pytest_addoption(parser):
    parser.addoption(
        "--skip-yaml-schema",
        action="store_true",
        default=False,
        help="跳过 test_data YAML action schema 校验",
    )


def pytest_sessionstart(session):
    configure_file_logger()
    tracker = get_token_usage_tracker()
    if tracker.active_run_kind is None:
        session.config._owns_token_usage_run = True
        tracker.start_run(
            run_kind="pytest",
            metadata={
                "cwd": str(Path.cwd()),
                "command": " ".join(sys.argv),
                "project": os.getenv("TEST_PROJECT"),
                "env": os.getenv("TEST_ENV"),
            },
        )
    else:
        session.config._owns_token_usage_run = False
    if session.config.getoption("--skip-yaml-schema"):
        return
    try:
        validate_pytest_targets(session.config.args)
    except YamlSchemaValidationError as exc:
        _exit_for_yaml_schema_error(exc)


def _exit_for_yaml_schema_error(exc: YamlSchemaValidationError) -> None:
    message = str(exc)
    logger.error(message)
    get_token_usage_tracker().finish_run(
        status="failed",
        metadata={"phase": "yaml_schema", "error": message},
    )
    pytest.exit(message, returncode=2)


@pytest.fixture(scope="session")
def browser() -> Generator[Browser, None, None]:
    """
    创建浏览器实例，session 级别的 fixture
    """
    with sync_playwright() as playwright:
        launch_options = _browser_launch_options()
        logger.info(
            "Launching browser: "
            f"browser={config.browser.value}, "
            f"headed={config.headed}, "
            f"headless={launch_options['headless']}, "
            f"slow_mo={launch_options.get('slow_mo', 0)}"
        )
        browser = getattr(playwright, config.browser).launch(**launch_options)
        yield browser
        browser.close()


def _browser_launch_options() -> dict[str, Any]:
    options: dict[str, Any] = {"headless": not config.headed}
    if config.slow_mo:
        options["slow_mo"] = config.slow_mo
    return options


@pytest.fixture(scope="function")
def context(browser, request):
    """创建浏览器上下文"""
    context_options = _browser_context_options(config.browser_config or {})
    cookie_file = context_options.pop("_cookie_file", None)
    tracing_options = context_options.pop("_tracing_options", None)
    browser_context = browser.new_context(**context_options)
    browser_context.set_default_timeout(DEFAULT_TIMEOUT)
    _apply_context_cookie_file(browser_context, cookie_file)
    tracing_started = _start_tracing(browser_context, tracing_options)
    yield browser_context
    _stop_tracing(browser_context, tracing_options, tracing_started, request)
    browser_context.close()


def _browser_context_options(raw_options: dict | None) -> dict:
    options = dict(raw_options or {})
    cookie_path = options.pop("cookie_file", None) or options.pop("cookies_file", None)
    if cookie_path:
        options["_cookie_file"] = cookie_path
    tracing = options.pop("tracing", None) or options.pop("trace", None)
    if tracing:
        options["_tracing_options"] = tracing
    for key in ("storage_state", "storageState"):
        if options.get(key):
            options["storage_state"] = _resolve_context_file(options[key])
            if key != "storage_state":
                options.pop(key, None)
            break
    return options


def _start_tracing(browser_context, tracing_options: Any | None) -> bool:
    mode, options = _normalize_tracing_options(tracing_options)
    if mode == "off":
        return False
    browser_context.tracing.start(**options)
    return True


def _stop_tracing(browser_context, tracing_options, tracing_started: bool, request):
    if not tracing_started:
        return
    mode, _ = _normalize_tracing_options(tracing_options)
    failed = bool(getattr(getattr(request.node, "rep_call", None), "failed", False))
    if mode == "on" or failed:
        trace_path = _trace_path_for_node(request.node.nodeid)
        browser_context.tracing.stop(path=str(trace_path))
        logger.info(f"Playwright trace saved: {trace_path}")
        return
    browser_context.tracing.stop()


def _normalize_tracing_options(raw_options: Any | None) -> tuple[str, dict[str, bool]]:
    if isinstance(raw_options, str):
        return raw_options.strip().lower(), {
            "screenshots": True,
            "snapshots": True,
            "sources": True,
        }
    if isinstance(raw_options, dict):
        mode = str(raw_options.get("mode", "retain-on-failure")).strip().lower()
        return mode, {
            "screenshots": bool(raw_options.get("screenshots", True)),
            "snapshots": bool(raw_options.get("snapshots", True)),
            "sources": bool(raw_options.get("sources", True)),
        }
    if raw_options:
        return "retain-on-failure", {
            "screenshots": True,
            "snapshots": True,
            "sources": True,
        }
    return "off", {}


def _trace_path_for_node(nodeid: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in nodeid)
    trace_dir = Path("evidence/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{safe_name[:180]}.zip"


def _apply_context_cookie_file(browser_context, cookie_path: Any | None) -> None:
    cookie_path = cookie_path or os.environ.get("UI_COOKIE_FILE")
    if not cookie_path:
        return
    path = Path(_resolve_context_file(cookie_path))
    if not path.exists():
        raise FileNotFoundError(f"cookie file not found: {path}")
    cookies = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(cookies, dict):
        cookies = cookies.get("cookies") or []
    browser_context.add_cookies(convert_cookies(cookies))


def _resolve_context_file(value: Any) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path)


@pytest.fixture(scope="function")
def page(context) -> Generator[Page, Any, None]:
    """
    创建页面，function级别的fixture
    """
    page = context.new_page()
    yield page
    page.close()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    pytest hook，用于获取测试结果状态
    供screenshot_fixture使用
    """
    # 执行hook的其余部分
    outcome = yield
    rep = outcome.get_result()

    # 设置测试节点的rep_call属性
    setattr(item, f"rep_{rep.when}", rep)
    if rep.failed:
        discard_pending_selector_cache(
            reason=f"{getattr(rep, 'outcome', 'failed')} during {rep.when}"
        )
    if rep.when == "call":
        setattr(item, "_selector_cache_call_passed", bool(rep.passed))
    elif (
        rep.when == "teardown"
        and rep.passed
        and getattr(item, "_selector_cache_call_passed", False)
        and not os.environ.get("UI_SELECTOR_CACHE_HOLD_UNTIL_GENERATION_VERIFIED")
    ):
        commit_pending_selector_cache()


# 将 expirationDate 转换为 Playwright 所需的 expires 字段
def convert_cookies(cookies):
    for cookie in cookies:
        # 将 expirationDate 转换为 Playwright 所需的 expires 字段
        if "expirationDate" in cookie:
            cookie["expires"] = int(
                cookie["expirationDate"]
            )  # Playwright 需要的是 Unix 时间戳
            del cookie["expirationDate"]  # 删除原有的 expirationDate 字段

        if cookie.get("sameSite") == "unspecified":
            cookie["sameSite"] = "None"  # 或者 'Lax' 或 'Strict'，根据实际需求
        # 如果 cookie 是会话 cookie，则删除 expires 字段
        if cookie.get("session", False):
            if "expires" in cookie:
                del cookie["expires"]
    return cookies


@pytest.fixture(scope="function")
def ui_helper(page):
    """
    创建UIHelper的fixture,用于封装页面操作
    :param page:
    :return:
    """
    ui = BasePage(page)
    yield ui


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """测试结束时输出 AI token usage 汇总。"""
    usage_summary = get_token_usage_tracker().last_summary
    if usage_summary and getattr(
        terminalreporter.config, "_owns_token_usage_run", True
    ):
        terminalreporter.write_sep("-", "AI Token Usage")
        terminalreporter.write_line(format_token_usage_summary(usage_summary))
        terminalreporter.write_line(f"details: {usage_summary['summary_file']}")


def pytest_collect_file(file_path: Path, parent):  # noqa
    if file_path.suffix not in {".yaml", ".yml"}:
        return None
    if test_cases := load_test_cases(file_path):
        datas = run_test_data()
        py_module, module = create_py_module(file_path, parent, test_cases, datas)
        py_module._getobj = lambda: module  # 返回 pytest 模块对象
        return py_module
    return None


def create_py_module(file_path: Path, parent, test_cases, datas):
    """创建并生成 py 模块"""
    py_module = Module.from_parent(parent, path=file_path)
    module = types.ModuleType(file_path.stem)  # 动态创建 module
    # 解析 YAML 并生成测试函数
    generator = TestCaseGenerator.from_parent(
        parent, module=module, name=module.__name__, test_cases=test_cases, datas=datas
    )
    generator.generate()
    return py_module, module


@pytest.fixture()
def login(page, ui_helper, request):
    elements = run_test_data().get("elements")
    login_modules = load_modules().get("login")
    step_executor = StepExecutor(page, ui_helper, elements)
    for step in login_modules:
        step_executor.execute_step(step)

    return None


@pytest.fixture()
def fixture_demo():
    logger.debug("fixture demo")
    return "fixture demo"


def pytest_generate_tests(metafunc):  # noqa
    """测试用例参数化功能实现"""
    # 获取测试函数对应的测试数据
    func_name = metafunc.function.__name__
    params_data = getattr(metafunc.module, f"{func_name}_data", None)

    # 如果没有数据，跳过参数化
    if not params_data:
        return

    # 确保测试数据是列表形式
    if not isinstance(params_data, list):
        params_data = [params_data]

    # 生成测试ID
    ids = [
        value.get("description", f"用例{i + 1}") for i, value in enumerate(params_data)
    ]

    # 参数化
    metafunc.parametrize(
        "value",
        params_data,
        ids=ids,
        scope="function",
    )


@pytest.fixture()
def get_test_name(request):
    """返回当前测试用例的完整名称，包括参数化ID"""
    test_name = request.node.name
    # 将Unicode转义序列解码为实际的中文字符
    try:
        # 首先尝试标准的unicode_escape解码
        decoded_name = test_name.encode("utf-8").decode("unicode_escape")

        # 如果解码后看起来像乱码（包含特殊字符），尝试其他方法
        if any(ord(c) > 127 and ord(c) < 256 for c in decoded_name):
            # 这可能是UTF-8字节被错误解释，尝试重新编码
            try:
                decoded_name = decoded_name.encode("latin-1").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                # 如果还是失败，保持原来的解码结果
                pass

    except (UnicodeDecodeError, UnicodeEncodeError):
        # 如果解码失败，使用原始名称
        decoded_name = test_name

    # 移除DEBUG日志，减少重复信息
    return decoded_name


@pytest.fixture()
def current_test_name(request):
    """返回当前测试用例的基础名称（不包含参数化部分）"""
    test_name = request.node.name
    # 提取基础测试名称（去掉参数化部分）
    base_name = test_name.split("[")[0] if "[" in test_name else test_name
    logger.debug(f"当前测试用例基础名称: {base_name}")
    return base_name


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    wait_for_pending_element_updates(timeout_seconds=2.0)
    if getattr(session.config, "_owns_token_usage_run", True):
        get_token_usage_tracker().finish_run(
            status="passed" if exitstatus == 0 else "failed",
            metadata={
                "exit_code": exitstatus,
                "project": os.getenv("TEST_PROJECT"),
                "env": os.getenv("TEST_ENV"),
            },
        )
    """
    测试会话结束时执行的钩子函数
    用于清理测试数据文件（在所有测试完成后只执行一次）
    """
    try:
        variables_file = Path("test_data/variables.json")
        if variables_file.exists():
            variables_file.unlink()
            logger.info(f"已在测试会话结束时删除临时测试数据文件: {variables_file}")
    except Exception as e:
        logger.error(f"删除测试数据文件时出错: {e}")


def pytest_collection_modifyitems(items) -> None:
    # item表示每个测试用例，解决用例名称中文显示问题
    for item in items:
        item.name = item.name.encode().decode("unicode-escape")
        item._nodeid = item._nodeid.encode().decode("unicode-escape")
