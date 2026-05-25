import types
from inspect import Parameter, Signature

import pytest
from playwright.sync_api import Page

from src.test_case_executor import CaseExecutor
from utils.variable_manager import VariableManager

_DEFAULT_FIXTURES = [
    "page",
    "ui_helper",
    "get_test_name",
    "value",
]


def build_test_signature(fixtures: list) -> Signature:
    if not isinstance(fixtures, list):
        raise ValueError("fixtures 必须是列表类型")  # 中文错误信息
    conflict = set(fixtures) & set(_DEFAULT_FIXTURES)
    if conflict:
        conflict_fixtures_str = ", ".join(conflict)  # 将冲突的 fixtures 转换为字符串
        raise ValueError(
            f"禁止覆盖默认 fixtures: {conflict_fixtures_str}。 默认 fixtures 包括: {', '.join(_DEFAULT_FIXTURES)}"
        )  # 更详细的中文错误信息，列出冲突的 fixtures 和默认 fixtures
    parameters = [
        Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
        for name in _DEFAULT_FIXTURES + fixtures
    ]
    return Signature(parameters)


class TestCaseGenerator(pytest.Item):

    def __init__(
        self,
        module: types.ModuleType,
        name,
        test_cases,
        datas,
        **kw,
    ):
        super().__init__(name, **kw)
        self.datas = datas
        self.test_cases = test_cases
        self.test_datas = self.datas.get("test_datas", {})
        self.elements = self.datas.get("elements", {})
        self.module: types.ModuleType = module  # 动态创建的 module 模型
        self.module_variable = {}  # 模块变量
        self.variable_manager = VariableManager()  # 初始化变量管理器

    def generate(self) -> None:
        """主生成方法"""
        if not isinstance(self.test_cases, list):  # 检查 test_cases 类型
            raise ValueError(
                f"'test_cases' 数据格式错误，期望列表类型，但实际为: {type(self.test_cases)}"
            )

        for case in self.test_cases:
            if not isinstance(case, dict):  # 循环内部增加 case 类型检查
                print(
                    f"警告: 发现非字典类型的用例数据: {case}，已跳过该用例。请检查数据文件。"
                )  # 警告非字典类型，而不是直接 raise 错误，更友好
                continue  # 跳过当前 case，继续处理下一个
            try:
                test_func = self._create_test_function(case)
                setattr(self.module, test_func.__name__, test_func)
            except (
                ValueError
            ) as e:  # 捕获 _create_test_function 中可能抛出的 ValueError
                print(
                    f"警告: 生成用例函数失败，原因: {e}。 用例数据: {case}，已跳过该用例。"
                )  # 警告生成失败原因，并打印用例数据方便排查
                continue  # 跳过当前 case，继续处理下一个

    def _create_test_function(self, case: dict):
        """动态生成测试函数（优化版）"""
        if not isinstance(case, dict):  # 增加 case 类型检查
            raise ValueError(f"用例数据格式错误，期望字典类型，但实际为: {type(case)}")
        try:
            case_name = case["name"]
        except KeyError:
            raise ValueError(
                "用例数据缺少 'name' 字段，请检查用例定义"
            )  # 更友好的错误提示

        depends = case.get("depends_on", [])
        fixtures = case.get("fixtures", [])
        data_name = case.get("data_name", case_name)
        case_data = self.test_datas.get(data_name, {})

        case_data = (
            [case_data]
            if isinstance(case_data, dict) and case_data
            else case_data if isinstance(case_data, list) else []
        )
        # 设置测试数据
        setattr(self.module, f"{case_name}_data", case_data)

        # 使用闭包绑定当前 case 数据
        def _test_function_wrapper_for_case(
            page, ui_helper, **kwargs
        ):  # 重命名闭包函数
            self.execute_test(
                case_data=kwargs.get("value", {}),
                case_metadata=case,
                elements=self.elements,
                page=page,
                ui_helper=ui_helper,
                **kwargs,
            )

        # 添加依赖标记
        marked_func = pytest.mark.dependency(name=case_name, depends=depends)(
            _test_function_wrapper_for_case
        )

        # 设置函数元数据
        marked_func.__name__ = case_name
        marked_func.__doc__ = case.get("description", "")
        marked_func.__signature__ = build_test_signature(fixtures)
        return marked_func

    def execute_test(
        self, case_data, page: Page, ui_helper, case_metadata=None, **kwargs
    ) -> None:
        executor = CaseExecutor(case_data, self.elements, case_metadata=case_metadata)
        executor.execute_test_case(page, ui_helper)

    def runtest(self):
        """
        运行测试用例
        """
        pass
