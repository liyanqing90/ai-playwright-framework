"""
步骤执行器的工具函数
"""

import importlib.util
import sys
from pathlib import Path

from faker import Faker


def generate_faker_data(data_type, **kwargs):
    """
    生成Faker数据的辅助函数
    这个函数只是为了兼容旧代码，实际调用BasePage中的方法
    """
    # 兼容旧的简单数据类型
    if data_type == "name":
        faker = Faker()
        return "新零售" + faker.uuid4().replace("-", "")[:6]
    elif data_type == "mobile":
        return "18210233933"
    elif data_type == "datetime":
        import datetime

        today = datetime.date.today()
        return today.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def run_dynamic_script_from_path(file_path: Path):
    """
    从 Path 对象表示的文件路径动态地导入和执行一个 Python 模块。
    Args:
        file_path:  A pathlib.Path object pointing to the Python file.
    """
    file_path = Path(file_path)
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"文件 {file_path} 不存在。")
        module_name = file_path.stem  # 获取不带扩展名的文件名 (模块名)
        spec = importlib.util.spec_from_file_location(
            module_name, str(file_path)
        )  # 创建模块规范, 需要字符串路径
        if spec is None:
            print(f"无法从文件路径 {file_path} 创建模块规范。")
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        # 检查模块是否定义了一个 `run()` 函数，如果有，则调用它
        if hasattr(module, "run"):
            module.run()
        elif hasattr(module, "main"):
            module.main()
        else:
            print(f"模块 {module_name} 没有 'run' 或 'main' 函数。")
    except FileNotFoundError as e:
        print(e)  # 直接打印 FileNotFoundError 异常信息
    except Exception as e:
        print(f"导入或执行模块 {file_path} 时发生错误：{e}")
