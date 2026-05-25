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


def _resolve_allowed_script_path(file_path: Path) -> Path:
    project_root = Path.cwd().resolve()
    scripts_root = (project_root / "files").resolve()
    resolved_path = Path(file_path)
    if not resolved_path.is_absolute():
        resolved_path = (project_root / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()
    if resolved_path.suffix != ".py":
        raise ValueError(f"动态脚本必须是 .py 文件: {resolved_path}")
    if not resolved_path.is_relative_to(scripts_root):
        raise ValueError(f"动态脚本只允许位于 files 目录下: {resolved_path}")
    return resolved_path


def run_dynamic_script_from_path(file_path: Path):
    """
    从 Path 对象表示的文件路径动态地导入和执行一个 Python 模块。
    Args:
        file_path:  A pathlib.Path object pointing to the Python file.
    """
    file_path = _resolve_allowed_script_path(Path(file_path))
    if not file_path.exists():
        raise FileNotFoundError(f"文件 {file_path} 不存在。")
    module_name = f"dynamic_script_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法从文件路径 {file_path} 创建模块规范。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "run"):
        return module.run()
    if hasattr(module, "main"):
        return module.main()
    raise AttributeError(f"模块 {module_name} 没有 'run' 或 'main' 函数。")
