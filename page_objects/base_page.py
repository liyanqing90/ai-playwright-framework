import functools
import json
import os
import re
import time
from typing import Callable, Literal, Optional, List, Any, Dict

import allure
from jsonpath_ng import parse
from playwright.sync_api import Page, expect
from pytest_check import check

from constants import DEFAULT_TIMEOUT, DEFAULT_TYPE_DELAY
from utils.logger import logger
from utils.variable_manager import VariableManager


def handle_page_error(description: str = "操作") -> Callable:
    """统一的页面操作错误处理装饰器，带有 allure step 和操作记录"""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 构建步骤描述
            operation_description = description
            selector = kwargs.get("selector", None)

            # 记录操作内容
            if selector:
                operation_description += f"，操作元素: {selector}"

            try:
                with allure.step(f"{operation_description}"):
                    return func(self, *args, **kwargs)
            except TimeoutError as e:
                info = "元素等待超时"
                if selector:
                    info += f"，元素: {selector}"

                # 截图
                screenshot = self.page.screenshot()
                allure.attach(
                    screenshot,
                    name="超时截图",
                    attachment_type=allure.attachment_type.PNG,
                )

                logger.error(info)
                setattr(e, "_logged", True)
                raise TimeoutError(info)

            except Exception as e:
                # 构建错误信息
                error_msg = f"{operation_description} 操作失败: {str(e)}"
                if selector:
                    error_msg += f"，元素: {selector}"

                # 尝试捕获元素截图，如果失败则截取整个页面
                screenshot = self.page.screenshot()

                # 添加截图到报告
                allure.attach(
                    screenshot,
                    name="错误截图",
                    attachment_type=allure.attachment_type.PNG,
                )
                raise Exception(error_msg)

        return wrapper

    return decorator


def attach_screenshot(page: Page, name="screenshot"):
    """将屏幕截图添加到 Allure 报告，并处理可能出现的异常."""
    screenshot = page.screenshot()
    allure.attach(screenshot, name=name, attachment_type=allure.attachment_type.PNG)


def check_and_screenshot(description="Assertion"):
    """
    装饰器，用于捕获断言失败并进行截图。
    Args:
        description: 断言的描述，用于 Allure 报告。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            step_description = description
            selector = kwargs.get("selector", None)
            expected = kwargs.get("expected", None)
            if selector:
                step_description += f"，断言元素: {selector}"
            if expected:
                step_description += f"，断言值: {expected}"

            try:
                with allure.step(f"{step_description}"):
                    try:
                        return func(self, *args, **kwargs)  # 执行被装饰的函数（断言）
                    except AssertionError as e:
                        # 提取断言错误信息中的关键部分

                        simplified_error = f"断言失败：{step_description}"
                        error_str = str(e)
                        # 尝试提取 Locator expected 和 Actual value 部分
                        import re

                        actual_pattern = r"Actual value: ([^\n]*)"

                        actual_match = re.search(actual_pattern, error_str)

                        actual_text = actual_match.group(1) if actual_match else ""
                        if actual_text:
                            # 构建简化的错误信息 \n实际值: '{actual_text}'"
                            simplified_error += f"，实际值: '{actual_text}'"
                        # 截图
                        if selector and hasattr(self.page, "locator"):
                            try:
                                # 尝试截取元素截图
                                screenshot = self.page.locator(
                                    selector
                                ).first.screenshot()
                            except Exception:
                                screenshot = self.page.screenshot()
                        else:
                            # 如果没有选择器，直接截取页面
                            screenshot = self.page.screenshot()

                        # 添加截图到报告
                        allure.attach(
                            screenshot,
                            name="失败截图",
                            attachment_type=allure.attachment_type.PNG,
                        )

                        # 使用check.fail来记录失败，但不抛出异常

                        # 创建并抛出异常对象，而不是字符串
                        raise AssertionError(simplified_error)
            except Exception as e:
                check.fail(e)

                if not hasattr(e, "_logged"):
                    logger.error(e)
                    setattr(e, "_logged", True)

                    from src.step_actions.step_executor import StepExecutor
                    import gc

                    # 搜索内存中的所有 StepExecutor 实例
                    for obj in gc.get_objects():
                        if isinstance(obj, StepExecutor):
                            obj.step_has_error = True
                            obj.has_error = True
                            setattr(obj, "_last_assertion_error", str(e))
                            break

                # 返回空值，不抛出异常，允许继续执行
                raise AssertionError(e)

            # except Exception as e:
            #     if not hasattr(e, "_logged"):
            #         logger.error(f"断言过程中发生异常: {e}")
            #         setattr(e, "_logged", True)
            #     raise

        return wrapper

    return decorator


def base_url():
    return os.environ.get("BASE_URL")


class BasePage:
    def __init__(self, page: Page):
        self.page = page
        self.pages = [self.page]
        self.variable_manager = VariableManager()
        self._setup_page_handlers()

    def _setup_page_handlers(self):
        """设置页面事件处理器"""
        self.page.on("pageerror", lambda exc: logger.error(f"页面错误: {exc}"))
        self.page.on("crash", lambda: logger.error("页面崩溃"))

    def _locator(
        self,
        selector: str,
        state: Literal["attached", "detached", "hidden", "visible"] = "visible",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """统一的元素定位与等待方法"""
        try:
            # 首先等待元素达到指定状态
            self.page.wait_for_selector(selector, state=state, timeout=timeout)
            # 然后返回定位器
            return self.page.locator(selector)
        except Exception as e:
            raise Exception(f"定位或等待元素 {selector} 失败 (state={state}): {str(e)}")

    @handle_page_error(description="导航到")
    def navigate(self, url: str):
        """导航到指定URL"""
        self.page.goto(url)
        self.page.wait_for_load_state()

    @handle_page_error(description="暂停")
    def pause(self):
        self.page.pause()

    @handle_page_error(description="点击元素")
    def click(self, selector: str):
        """点击元素"""
        self._locator(selector).first.click(timeout=DEFAULT_TIMEOUT)

    @handle_page_error(description="上传文件")
    def upload_file(self, selector: str, file_path: str):
        """上传文件"""
        self.page.set_input_files(selector, file_path)

    @handle_page_error(description="输入文本")
    def fill(self, selector: str, value: Any):
        """在输入框中填写文本"""
        resolved_text = self.variable_manager.replace_variables_refactored(value)
        # 确保传递给 Playwright 的 fill 方法的值是字符串类型
        if resolved_text is not None:
            resolved_text = str(resolved_text)
        self._locator(selector).fill(resolved_text)

    @handle_page_error(description="按键")
    def press_key(self, selector: str, key: str):
        """在元素上按键"""
        self._locator(selector).press(key)

    @handle_page_error(description="获取元素文本")
    def get_text(self, selector: str) -> str:
        """获取元素文本"""
        return self._locator(selector).first.inner_text()

    @check_and_screenshot("断言URL")
    def assert_url(self, expected: str):
        """断言当前URL"""
        expect(self.page).to_have_url(expected)

    @check_and_screenshot("断言文本")
    def assert_text(self, selector: str, expected: str):
        """断言元素文本"""
        resolved_expected = self.variable_manager.replace_variables_refactored(expected)
        expect(self._locator(selector).first).to_have_text(resolved_expected)

    @allure.step("硬断言元素文本")
    def hard_assert_text(self, selector: str, expected: str):
        """硬断言元素文本，失败时会终止测试执行"""
        try:
            resolved_expected = self.variable_manager.replace_variables_refactored(
                expected
            )
            expect(self._locator(selector).first).to_have_text(resolved_expected)
        except AssertionError as e:
            # 截图
            if selector and hasattr(self.page, "locator"):
                try:
                    # 尝试截取元素截图
                    screenshot = self.page.locator(selector).first.screenshot()
                except Exception:
                    screenshot = self.page.screenshot()
            else:
                # 如果没有选择器，直接截取页面
                screenshot = self.page.screenshot()

            # 添加截图到报告
            allure.attach(
                screenshot,
                name="硬断言失败截图",
                attachment_type=allure.attachment_type.PNG,
            )

            # 标记为硬断言异常
            setattr(e, "_hard_assert", True)
            logger.error(f"硬断言失败: {e}")
            raise

    @check_and_screenshot("断言页面标题")
    def assert_title(self, expected: str):
        """断言页面标题"""
        expect(self.page).to_have_title(expected)

    @check_and_screenshot("断言元素数量")
    def assert_element_count(self, selector: str, expected: int):
        """断言元素数量"""
        try:
            expected = int(expected)
        except Exception:
            logger.error(f"断言元素数量失败: 期望数量 '{expected}' 不是有效的整数")
            raise

        expect(self._locator(selector)).to_have_count(expected)

    @check_and_screenshot("断言元素包含文本")
    def assert_text_contains(self, selector: str, expected: str):
        """断言元素文本包含指定内容"""
        resolved_expected = self.variable_manager.replace_variables_refactored(expected)
        expect(self._locator(selector)).to_contain_text(resolved_expected)

    @check_and_screenshot("断言URL包含")
    def assert_url_contains(self, expected: str):
        """断言当前URL包含指定内容"""
        resolved_expected = self.variable_manager.replace_variables_refactored(expected)
        expect(self.page).to_have_url(re.compile(f".*{re.escape(resolved_expected)}.*"))

    @check_and_screenshot("断言元素存在")
    def assert_exists(self, selector: str):
        """断言元素存在于DOM中"""
        expect(self._locator(selector)).to_be_attached()

    @check_and_screenshot("断言元素不存在")
    def assert_not_exists(self, selector: str):
        """断言元素不存在于DOM中"""
        expect(self._locator(selector)).not_to_be_attached()

    @check_and_screenshot("断言元素启用状态")
    def assert_element_enabled(self, selector: str):
        """断言元素处于启用状态（非禁用）"""
        expect(self._locator(selector)).to_be_enabled()

    @check_and_screenshot("断言元素禁用状态")
    def assert_element_disabled(self, selector: str):
        """断言元素处于禁用状态"""
        expect(self._locator(selector)).to_be_disabled()

    @check_and_screenshot("断言元素可见性")
    def assert_visible(self, selector: str):
        """断言元素可见"""
        expect(self._locator(selector)).to_be_visible()

    @check_and_screenshot("断言元素不可见")
    def assert_not_visible(self, selector: str):
        """断言元素不可见"""
        expect(self._locator(selector)).not_to_be_visible()

    @check_and_screenshot("断言元素隐藏")
    def assert_be_hidden(self, selector: str):
        """断言元素隐藏"""
        expect(self.page.locator(selector)).to_be_hidden()

    @check_and_screenshot("断言元素属性值")
    def assert_attribute(self, selector: str, attribute: str, expected: str):
        """断言元素属性值"""
        expect(self._locator(selector)).to_have_attribute(attribute, expected)

    @check_and_screenshot("断言元素值")
    def assert_value(self, selector: str, expected: str):
        """断言元素值"""
        resolved_expected = self.variable_manager.replace_variables_refactored(expected)
        expect(self._locator(selector)).to_have_value(resolved_expected)

    @check_and_screenshot("断言元素已选中")
    def assert_checked(self, selector: str):
        """断言元素已选择"""
        expect(self._locator(selector)).to_be_checked()

    @handle_page_error(description="存储变量")
    def store_variable(self, name: str, value: str, scope: str = "global"):
        """存储变量"""
        self.variable_manager.set_variable(name, value, scope)

    @handle_page_error(description="存储元素文本")
    def store_text(self, selector: str, variable_name: str, scope: str = "global"):
        """存储元素文本到变量"""
        text = self.get_text(selector)
        logger.debug(f"存储变量 {variable_name}: {text}")
        self.store_variable(variable_name, text, scope)

    @handle_page_error(description="存储元素属性")
    def store_attribute(
        self, selector: str, attribute: str, variable_name: str, scope: str = "global"
    ):
        """存储元素属性到变量"""
        value = self._locator(selector).get_attribute(attribute)
        self.store_variable(variable_name, value, scope)

    @handle_page_error(description="刷新页面")
    def refresh(self):
        """刷新页面"""
        self.page.reload()
        self.page.wait_for_load_state("networkidle")

    @handle_page_error(description="等待指定时间")
    def wait_for_timeout(self, timeout: int):
        """等待指定时间"""
        self.page.wait_for_timeout(timeout)

    @handle_page_error(description="等待页面加载完成")
    def wait_for_load_state(
        self, state: Literal["domcontentloaded", "load", "networkidle"] | None = None
    ):
        """等待页面加载状态"""
        self.page.wait_for_load_state(state)

    def wait_and_click(self, selector: str, timeout: Optional[int] = DEFAULT_TIMEOUT):
        """等待元素可点击并点击"""
        self._locator(selector, timeout=timeout).click()

    def wait_and_fill(
        self, selector: str, text: Any, timeout: Optional[int] = DEFAULT_TIMEOUT
    ):
        """等待元素可见并输入文本"""
        # 确保传递给 Playwright 的 fill 方法的值是字符串类型
        if text is not None:
            text = str(text)
        self._locator(selector, timeout=timeout).fill(text)

    def close(self):
        self.page.close()

    @staticmethod
    def reuse_browser_context(context):
        """复用浏览器上下文"""
        return context.new_page()

    @handle_page_error(description="鼠标悬停在元素")
    def hover(self, selector: str):
        """鼠标悬停在元素上"""
        self._locator(selector).hover()

    @handle_page_error(description="双击元素")
    def double_click(self, selector: str):
        """双击元素"""
        self._locator(selector).dblclick()

    @handle_page_error(description="右键点击元素")
    def right_click(self, selector: str):
        """右键点击元素"""
        self._locator(selector).click(button="right")

    @handle_page_error(description="选择下拉框选项")
    def select_option(self, selector: str, value: str):
        """选择下拉框选项"""
        self._locator(selector).select_option(value=value)

    @handle_page_error(description="拖拽元素")
    def drag_and_drop(self, source: str, target: str):
        """拖拽元素"""
        source_element = self._locator(source)
        target_element = self._locator(target)
        source_element.drag_to(target_element)

    @handle_page_error(description="获取元素值")
    def get_value(self, selector: str) -> str:
        """获取元素的value属性值"""
        return self._locator(selector).input_value()

    @handle_page_error(description="滚动到元素")
    def scroll_into_view(self, selector: str):
        """将元素滚动到可视区域"""
        self._locator(selector).scroll_into_view_if_needed()

    @handle_page_error(description="滚动到指定位置")
    def scroll_to(self, x: int = 0, y: int = 0):
        """滚动到指定坐标"""
        self.page.evaluate(f"window.scrollTo({x}, {y})")

    @handle_page_error(description="聚焦元素")
    def focus(self, selector: str):
        """聚焦到指定元素"""
        self._locator(selector).focus()

    @handle_page_error(description="使元素失焦")
    def blur(self, selector: str):
        """使元素失去焦点"""
        self.page.evaluate("element => element.blur()", self._locator(selector))

    @handle_page_error(description="键入文本")
    def type(self, selector: str, text: str, delay: int = DEFAULT_TYPE_DELAY):
        """模拟人工输入文字，带输入延迟"""
        self._locator(selector).type(text, delay=delay)

    @handle_page_error(description="清空输入框")
    def clear(self, selector: str):
        """清空输入框内容"""
        self._locator(selector).clear()

    @handle_page_error(description="进入iframe")
    def enter_frame(self, selector: str):
        """进入iframe"""
        self._locator(selector, state="attached")
        return self.page.frame_locator(selector)

    @handle_page_error(description="接受弹窗")
    def accept_alert(self, selector, value=None):
        dialog_message = None  # 用于存储弹框内容
        if not value:
            value = {}

        def handle_dialog(dialog):
            nonlocal dialog_message  # 声明为外部变量
            if message := value.get("message"):
                dialog.accept(message)
            else:
                dialog.accept()
            dialog_message = dialog.message

        self.page.once("dialog", handle_dialog)
        self._locator(selector).click()
        return dialog_message

    @handle_page_error(description="拒绝弹窗")
    def dismiss_alert(self, selector, value=None):
        dialog_message = None  # 用于存储弹框内容

        if not value:
            value = {}

        def handle_dialog(dialog):
            nonlocal dialog_message  # 声明为外部变量

            if message := value.get("message"):
                dialog.dismiss(message)
            else:
                dialog.dismiss()
            dialog_message = dialog.message

        self.page.once("dialog", handle_dialog)
        self._locator(selector).click()
        return dialog_message

    @handle_page_error(description="弹出tab")
    def expect_popup(self, action, selector, variable_name, scope="global"):
        with self.page.expect_popup() as popup_info:
            # 这里需要执行触发弹出的操作，可以递归调用
            if action == "click":
                self.click(selector)
        new_page = popup_info.value
        self.pages.append(new_page)
        self.page = new_page
        self.store_variable(variable_name, len(self.pages) - 1, scope)

    @handle_page_error(description="切换窗口")
    def switch_window(self, value=0):
        """切换到指定窗口"""
        if value < 0 or value >= len(self.pages):
            raise ValueError("无效的窗口索引")
        """切换到指定窗口"""
        self.page = self.pages[value]
        raise ValueError("未找到匹配的窗口")

    @handle_page_error(description="关闭当前窗口")
    def close_window(self):
        """关闭当前窗口"""
        if len(self.page.context.pages) == 1:
            raise RuntimeError("无法关闭最后一个窗口")
        self.page.close()
        self.page = self.page.context.pages[-1]

    @handle_page_error(description="等待新窗口打开")
    def wait_for_new_window(self) -> Page:
        """等待新窗口打开并返回新窗口"""
        with self.page.context.expect_page() as new_page_info:
            new_page = new_page_info.value
            new_page.wait_for_load_state()
            return new_page

    @handle_page_error(description="等待元素消失")
    def wait_for_element_hidden(
        self, selector: str, timeout: Optional[int] = DEFAULT_TIMEOUT
    ):
        """等待元素消失"""
        return self._locator(selector, state="hidden", timeout=timeout)

    def get_element_count(self, selector: str) -> int:
        """获取元素数量"""
        locator = self._locator(selector, state="attached")
        logger.debug(f"获取元素数量: {selector}{locator}")
        return locator.count()

    @handle_page_error(description="执行JavaScript")
    def execute_script(self, script: str):
        """执行JavaScript代码"""
        return self.page.evaluate(script)

    @handle_page_error(description="保存当前页面截图")
    def capture_screenshot(self, path: str):
        """主动保存页面截图"""
        self.page.screenshot(path=path)

    @handle_page_error(description="操作Cookie")
    def manage_cookies(self, action: str, **kwargs):
        """管理Cookie"""
        if action == "add":
            required = {"name", "value", "url"}
            if not required.issubset(kwargs):
                raise ValueError("添加Cookie缺少必要参数: name, value, url")
            self.page.context.add_cookies(**kwargs)

        elif action == "get":
            return self.page.context.cookies()
        elif action == "delete":
            self.page.context.clear_cookies()
        else:
            raise ValueError("无效的Cookie操作")
        return True

    @handle_page_error(description="获取元素属性")
    def get_element_attribute(self, selector: str, attribute: str) -> str:
        """获取元素属性"""
        self._locator(selector, state="attached")
        return self._locator(selector).get_attribute(attribute)

    @handle_page_error(description="获取当前页面URL")
    def get_current_url(self) -> str:
        """获取当前页面URL"""
        return self.page.url

    @handle_page_error(description="获取页面标题")
    def get_page_title(self) -> str:
        """获取页面标题"""
        return self.page.title()

    @handle_page_error(description="等待网络请求完成")
    def wait_for_network_idle(self, timeout: Optional[int] = DEFAULT_TIMEOUT):
        """等待网络请求完成"""
        self.page.wait_for_load_state("networkidle", timeout=timeout)

    @handle_page_error(description="等待元素可点击")
    def wait_for_element_clickable(
        self, selector: str, timeout: Optional[int] = DEFAULT_TIMEOUT
    ):
        """等待元素可点击"""
        locator = self._locator(selector, state="visible", timeout=timeout)
        is_enabled = not locator.is_disabled()
        if not is_enabled:
            raise TimeoutError(f"元素 {selector} 在 {timeout}ms 内未变为可点击状态")
        return locator

    @handle_page_error(description="等待元素包含文本")
    def wait_for_element_text(
        self,
        selector: str,
        expected: str,
        timeout: Optional[int] = DEFAULT_TIMEOUT,
    ):
        """等待元素包含指定文本"""
        start_time = time.time()
        while time.time() - start_time < timeout / 1000:
            locator = self._locator(selector, timeout=timeout)
            actual_text = locator.inner_text()
            if expected in actual_text:
                return True
            time.sleep(0.1)
        raise TimeoutError(f"元素 {selector} 在 {timeout}ms 内未包含文本 '{expected}'")

    @handle_page_error(description="获取所有匹配元素")
    def get_all_elements(self, selector: str) -> List[Any]:
        """获取所有匹配的元素"""
        elements = self._locator(selector, state="attached").all()
        return elements

    @handle_page_error(description="等待元素数量")
    def wait_for_element_count(
        self,
        selector: str,
        expected_count: int,
        timeout: Optional[int] = DEFAULT_TIMEOUT,
    ):
        """等待元素数量达到预期值"""
        start_time = self.page.evaluate("() => Date.now()")
        while True:
            actual_count = self.get_element_count(selector)
            if actual_count == expected_count:
                return True

            current_time = self.page.evaluate("() => Date.now()")
            elapsed = (current_time - start_time) / 1000  # 转换为秒

            if elapsed > timeout / 1000:
                logger.error(
                    f"等待元素 {selector} 数量为 {expected_count} 超时，当前数量为 {actual_count}"
                )
                raise TimeoutError(f"等待元素 {selector} 数量为 {expected_count} 超时")

            self.page.wait_for_timeout(100)  # 等待100毫秒再检查

    @handle_page_error(description="下载文件")
    def download_file(self, selector: str, save_path: Optional[str] = None) -> str:
        """点击下载按钮并获取下载的文件路径"""
        with self.page.expect_download() as download_info:
            self.click(selector)

        download = download_info.value
        logger.info(f"下载文件: {download.suggested_filename}")

        if save_path:
            download.save_as(save_path)
            return save_path
        else:
            # 使用默认路径
            path = download.path()
            return str(path)

    @handle_page_error(description="验证文件下载")
    def verify_download(
        self, file_pattern: str, timeout: int = DEFAULT_TIMEOUT
    ) -> bool:
        """验证文件是否已下载（通过文件名模式匹配）"""
        import glob
        import time
        import os

        # 获取下载目录
        download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(download_dir):
            # 尝试使用浏览器类型作为备用
            download_dir = os.path.join(
                "./downloads", self.page.context.browser.browser_type.name
            )
            os.makedirs(download_dir, exist_ok=True)

        logger.debug(f"检查下载目录: {download_dir}")
        start_time = time.time()

        while time.time() - start_time < timeout / 1000:
            # 检查下载目录中是否有匹配的文件
            matching_files = glob.glob(os.path.join(download_dir, file_pattern))
            if matching_files:
                logger.info(f"找到下载文件: {matching_files[0]}")
                return True
            time.sleep(0.5)

        logger.error(f"未找到下载文件: {file_pattern}")
        return False

    @handle_page_error(description="按下键盘快捷键")
    def press_keyboard_shortcut(self, key_combination: str):
        """
        按下键盘快捷键组合
        例如: "Control+A", "Shift+ArrowDown", "Control+Shift+V"
        """
        keys = key_combination.split("+")
        # 按下所有修饰键
        for i in range(len(keys) - 1):
            self.page.keyboard.down(keys[i])

        # 按下最后一个键
        self.page.keyboard.press(keys[-1])

        # 释放所有修饰键（从后往前）
        for i in range(len(keys) - 2, -1, -1):
            self.page.keyboard.up(keys[i])

        logger.debug(f"按下键盘快捷键: {key_combination}")

    @handle_page_error(description="全局按键")
    def keyboard_press(self, key: str):
        """全局按键，不针对特定元素"""
        self.page.keyboard.press(key)
        logger.debug(f"全局按键: {key}")

    @handle_page_error(description="全局输入文本")
    def keyboard_type(self, text: str, delay: int = DEFAULT_TYPE_DELAY):
        """全局输入文本，不针对特定元素"""
        resolved_text = self.variable_manager.replace_variables_refactored(text)
        self.page.keyboard.type(resolved_text, delay=delay)
        logger.debug(f"全局输入文本: {resolved_text}")

    @handle_page_error(description="监测操作触发的请求")
    def monitor_action_request(
        self,
        url_pattern: str,
        selector: str,
        action: str = "click",
        assert_params: Dict[str, Any] = None,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """
        监测操作触发的请求并验证参数

        Args:
            url_pattern: URL匹配模式，如 "**/api/user/**"
            selector: 要操作的元素选择器
            action: 要执行的操作，如 "click", "goto" 等
            assert_params: 要验证的参数列表，格式为 [{"$.path.to.field": expected_value}, ...]
            timeout: 等待超时时间(毫秒)
            **kwargs: 其他操作参数，如 goto 操作的 value

        Returns:
            捕获的请求数据
        """
        logger.info(f"开始监测请求: {url_pattern}, 操作: {action} 元素: {selector}")

        try:
            with self.page.expect_request(url_pattern, timeout=timeout) as request_info:
                # 执行操作
                if action == "click":
                    self.click(selector)
                elif action == "fill":
                    self.fill(selector)
                elif action == "press_key":
                    self.press_key(selector)
                elif action == "select":
                    self.select_option(selector)
                elif action == "goto":
                    self.navigate(kwargs.get("value"))
                else:
                    logger.warning(f"不支持的操作类型: {action}，将执行默认点击操作")
                    self.click(selector)

                # 等待请求完成
                request = request_info.value
                logger.info(f"捕获到请求: {request.url}")

                # 获取请求数据
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        request_data = request.post_data_json()
                    except Exception:
                        request_data = json.loads(request.post_data)
                    logger.info(f"请求数据 (解析为JSON): {request_data}")
                else:
                    # 对于GET请求，获取URL参数
                    from urllib.parse import urlparse, parse_qs

                    parsed_url = urlparse(request.url)
                    request_data = parse_qs(parsed_url.query)

                    # 将单项列表值转换为单个值
                    for key, value in request_data.items():
                        if isinstance(value, list) and len(value) == 1:
                            request_data[key] = value[0]

                    logger.info(f"请求参数: {request_data}")

                # 构建完整的请求信息
                captured_data = {
                    "url": request.url,
                    "method": request.method,
                    "data": request_data,
                    "headers": {k: v for k, v in request.headers.items()},
                }

                # 验证参数（如果需要）
                logger.info(f"捕获到请求: {assert_params}")
                if assert_params and request_data:
                    # 处理断言参数
                    for jsonpath_expr, expected_value in assert_params.items():
                        self._verify_jsonpath(
                            request_data, jsonpath_expr, expected_value
                        )

            return captured_data

        except Exception as e:
            logger.error(f"监测请求失败: {e}")
            screenshot = self.page.screenshot()
            allure.attach(
                screenshot,
                name="请求捕获失败截图",
                attachment_type=allure.attachment_type.PNG,
            )
            raise

    @handle_page_error(description="监测操作触发的响应")
    def monitor_action_response(
        self,
        url_pattern: str,
        selector: str,
        action: str = "click",
        assert_params: Dict[str, Any] = None,
        save_params: Dict[str, Any] = None,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """
        监测操作触发的响应并验证参数

        Args:
            url_pattern: URL匹配模式，如 "**/api/user/**"
            selector: 要操作的元素选择器
            action: 要执行的操作，如 "click", "fill" 等
            assert_params: 要验证的参数列表，格式为 [{"$.path.to.field": expected_value}, ...]
            save_params: 要保存的参数列表，格式为 [{"$.path.to.field": viable_name}, ...]
            timeout: 等待超时时间(毫秒)
            **kwargs: 其他操作参数，如 fill 操作的 value

        Returns:
            捕获的响应数据
        """
        logger.info(f"开始监测响应: {url_pattern}, 操作: {action} 元素: {selector}")

        try:
            with self.page.expect_response(
                url_pattern, timeout=timeout
            ) as response_info:
                # 执行操作
                if action == "click":
                    self.click(selector)
                elif action == "fill":
                    self.fill(selector)
                elif action == "press_key":
                    self.press_key(selector)
                elif action == "select":
                    self.select_option(selector)
                elif action == "goto":
                    self.navigate(kwargs.get("value"))
                else:
                    logger.warning(f"不支持的操作类型: {action}，将执行默认点击操作")
                    self.click(selector)

                # 等待响应完成
                response = response_info.value
                logger.info(f"捕获到响应: {response.url}, 状态码: {response.status}")

                # 获取响应数据
                try:
                    response_data = response.json()
                    logger.info(f"响应数据: {response_data}")

                    # 验证参数（如果需要）
                    if response_data:
                        if assert_params:
                            # 处理断言参数
                            for jsonpath_expr, expected_value in assert_params.items():
                                self._verify_jsonpath(
                                    response_data, jsonpath_expr, expected_value
                                )

                        if save_params:
                            # 处理保存参数
                            for jsonpath_expr, viable_name in save_params.items():
                                self._save_jsonpath(
                                    response_data, jsonpath_expr, viable_name
                                )
                    return response_data

                except Exception as e:
                    logger.error(f"处理响应数据失败: {e}")
                    raise

        except Exception as e:
            logger.error(f"监测响应失败: {e}")
            screenshot = self.page.screenshot()
            allure.attach(
                screenshot,
                name="响应捕获失败截图",
                attachment_type=allure.attachment_type.PNG,
            )
            raise

    def _save_jsonpath(self, data, jsonpath_expr, viable_name):
        """
        保存JSONPath表达式的值到变量

        Args:
            data: 要保存的数据
            jsonpath_expr: JSONPath表达式
            viable_name: 变量名称
        """

        # 解析 jsonpath 表达式
        jsonpath_expr = jsonpath_expr.strip()
        expr = parse(jsonpath_expr)
        logger.debug(f"JSONPath表达式: {jsonpath_expr}")
        logger.debug(f"变量名称: {viable_name}")

        # 查找匹配的值
        matches = [value.value for value in expr.find(data)][0]
        logger.debug(f"匹配的值: {matches}")
        self.store_variable(viable_name, matches)

    def _verify_jsonpath(self, data, jsonpath_expr, expected):
        """
        验证JSONPath表达式的值是否符合预期

        Args:
            data: 要验证的数据
            jsonpath_expr: JSONPath表达式
            expected: 期望值
        """

        # 解析 jsonpath 表达式
        jsonpath_expr = jsonpath_expr.strip()
        expr = parse(jsonpath_expr)

        # 查找匹配的值
        matches = [value.value for value in expr.find(data)][0]
        expected_value = self.variable_manager.replace_variables_refactored(expected)
        if expected_value and not matches:
            logger.error(f"JSONPath {jsonpath_expr} 未找到匹配项")
            raise ValueError(f"JSONPath {jsonpath_expr} 未找到匹配项，当前数据: {data}")

        # 处理变量替换
        resolved_expected = self.variable_manager.replace_variables_refactored(
            expected_value
        )

        # 执行断言
        with check, allure.step(f"验证参数 {jsonpath_expr}"):
            if isinstance(matches, list) and isinstance(resolved_expected, list):
                # 列表比较
                assert sorted([str(x) for x in matches]) == sorted(
                    [str(x) for x in resolved_expected]
                ), f"断言失败: 参数 {jsonpath_expr} 期望值为 '{resolved_expected}', 实际值为 '{matches}'"
            elif isinstance(matches, list):
                # 检查列表中是否包含期望值
                expected_str = str(resolved_expected)
                found = any(str(item) == expected_str for item in matches)
                assert (
                    found
                ), f"断言失败: 参数 {jsonpath_expr} 期望包含值 '{resolved_expected}', 实际值为 '{matches}'"
            else:
                # 单值比较
                assert str(matches) == str(
                    resolved_expected
                ), f"断言失败: 参数 {jsonpath_expr} 期望值为 '{resolved_expected}', 实际值为 '{matches}'"

        allure.attach(
            f"断言成功: 参数 {jsonpath_expr} 匹配期望值 {resolved_expected}",
            name="断言结果",
            attachment_type=allure.attachment_type.TEXT,
        )

        logger.info(f"参数验证成功: {jsonpath_expr} 匹配期望值 {resolved_expected}")

    @check_and_screenshot("断言元素值列表")
    def assert_values(self, selector: str, expected: List[str]):
        """断言元素有多个值（适用于多选框等）"""
        resolved_values = [
            self.variable_manager.replace_variables_refactored(val) for val in expected
        ]
        # actual_values = self.page.locator(selector).evaluate(
        #     "el => Array.from(el.selectedOptions).map(o => o.value)"
        # )
        expect(self.page.locator(selector)).to_have_values(resolved_values)
        # allure.attach(
        #     f"断言成功: 元素 {selector} 的值\n期望: {resolved_values}\n实际: {actual_values}",
        #     name="断言结果",
        #     attachment_type=allure.attachment_type.TEXT,
        # )

    @check_and_screenshot("断言元素有精确文本")
    def assert_exact_text(self, selector: str, expected: str):
        """断言元素有精确的文本（不包括子元素文本）"""
        resolved_expected = self.variable_manager.replace_variables_refactored(expected)
        # actual_text = self.page.locator(selector).inner_text()
        expect(self._locator(selector)).to_have_text(
            resolved_expected, use_inner_text=True
        )
        # allure.attach(
        #     f"断言成功: 元素 {selector} 的精确文本\n期望: '{resolved_expected}'\n实际: '{actual_text}'",
        #     name="断言结果",
        #     attachment_type=allure.attachment_type.TEXT,
        # )

    @check_and_screenshot("断言元素匹配文本正则")
    def assert_text_matches(self, selector: str, pattern: str):
        """断言元素文本匹配正则表达式"""
        # actual_text = self.get_text(selector)
        expect(self._locator(selector)).to_have_text(re.compile(pattern))
        # allure.attach(
        #     f"断言成功: 元素 {selector} 的文本匹配正则\n正则模式: '{pattern}'\n实际文本: '{actual_text}'",
        #     name="断言结果",
        #     attachment_type=allure.attachment_type.TEXT,
        # )
