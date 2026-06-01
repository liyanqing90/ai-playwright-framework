from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

from ai_playwright.utils.yaml_handler import YamlHandler

yaml_handler = YamlHandler()


def format_duplicate_locations(locations: List[Tuple[Path, int]]) -> str:
    """格式化重复位置信息"""
    # 按文件分组
    file_groups = defaultdict(list)
    for file_path, line in locations:
        file_groups[file_path].append(line)

    # 格式化输出
    parts = []
    for file_path, lines in file_groups.items():
        if len(lines) == 1:
            parts.append(f"{file_path.name} 第{lines[0]}行")
        else:
            # 去重并排序行号
            unique_lines = sorted(set(lines))
            if len(unique_lines) == 1:
                parts.append(f"{file_path.name} 第{unique_lines[0]}行")
            else:
                parts.append(
                    f"{file_path.name} 第{'行和第'.join(map(str, unique_lines))}行"
                )

    return "和".join(parts) + "重复"


def check_cases_duplicates(cases_dir: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """检查cases目录下的用例名称重复"""
    duplicates = defaultdict(list)

    for yaml_file in _yaml_files(cases_dir):
        content = yaml_handler.load_yaml(yaml_file)
        if isinstance(content, dict) and "test_cases" in content:
            for name, locations in _case_name_locations(yaml_file).items():
                duplicates[name].extend(locations)

    # 只保留真正的重复项
    return {
        name: locations for name, locations in duplicates.items() if len(locations) > 1
    }


def check_data_duplicates(data_dir: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """检查data目录下的测试数据用例名称重复"""
    duplicates = defaultdict(list)

    for yaml_file in _yaml_files(data_dir):
        content = yaml_handler.load_yaml(yaml_file)
        if isinstance(content, dict) and "test_data" in content:
            keys = set(content.get("test_data", {}))
            for name, locations in _top_level_key_locations(yaml_file, keys).items():
                duplicates[name].extend(locations)

    # 只保留真正的重复项
    return {
        name: locations for name, locations in duplicates.items() if len(locations) > 1
    }


def check_elements_duplicates(elements_dir: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """检查elements目录下的元素名称重复"""
    duplicates = defaultdict(list)
    element_names = defaultdict(list)

    for yaml_file in _yaml_files(elements_dir):
        content = yaml_handler.load_yaml(yaml_file)
        if isinstance(content, dict) and "elements" in content:
            keys = set(content.get("elements", {}))
            for name, locations in _top_level_key_locations(yaml_file, keys).items():
                element_names[name].extend(locations)

    # 找出重复的元素名称（只检查key）
    for name, locations in element_names.items():
        if len(locations) > 1:
            duplicates[name] = locations

    return duplicates


def check_project_duplicates(project_dir: Path) -> bool:
    """检查单个项目内的重复项，返回是否有重复项"""
    project_name = project_dir.name
    has_duplicates = False
    project_has_duplicates = False

    # 检查cases目录
    cases_dir = project_dir / "cases"
    if cases_dir.exists():
        case_duplicates = check_cases_duplicates(cases_dir)
        if case_duplicates:
            has_duplicates = True
            project_has_duplicates = True
            print(f"\n[ERROR] {project_name} 项目中发现重复的用例名称：")
            for name, locations in case_duplicates.items():
                print(f'  "{name}" 在 {format_duplicate_locations(locations)}')

    # 检查data目录
    data_dir = project_dir / "data"
    if data_dir.exists():
        data_duplicates = check_data_duplicates(data_dir)
        if data_duplicates:
            has_duplicates = True
            project_has_duplicates = True
            print(f"\n[ERROR] {project_name} 项目中发现重复的测试数据名称：")
            for name, locations in data_duplicates.items():
                print(f'  "{name}" 在 {format_duplicate_locations(locations)}')

    # 检查elements目录
    elements_dir = project_dir / "elements"
    if elements_dir.exists():
        element_duplicates = check_elements_duplicates(elements_dir)
        if element_duplicates:
            has_duplicates = True
            project_has_duplicates = True
            print(f"\n[ERROR] {project_name} 项目中发现重复的元素名称：")
            for name, locations in element_duplicates.items():
                print(f'  "{name}" 在 {format_duplicate_locations(locations)}')

    return project_has_duplicates


def main():
    test_data_dir = Path("test_data")
    has_any_duplicates = False

    print(f"开始检查 {test_data_dir} 目录下的重复项...")

    # 遍历test_data下的所有项目
    project_count = 0
    for project_dir in test_data_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_count += 1
        print(f"正在检查项目: {project_dir.name}")

        if check_project_duplicates(project_dir):
            has_any_duplicates = True

    print(f"共检查了 {project_count} 个项目")

    if not has_any_duplicates:
        print("[OK] 所有项目中均未发现重复项")
        return 0
    return 1


def _yaml_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("**/*.yaml", "**/*.yml")
        for path in directory.glob(pattern)
    )


def _case_name_locations(yaml_file: Path) -> Dict[str, List[Tuple[Path, int]]]:
    locations = defaultdict(list)
    for line_no, line in enumerate(
        yaml_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.lstrip().startswith("#"):
            continue
        match = re.search(r"\bname:\s*['\"]?([^#'\"]+?)['\"]?\s*(?:#.*)?$", line)
        if match:
            locations[match.group(1).strip()].append((yaml_file, line_no))
    return locations


def _top_level_key_locations(
    yaml_file: Path,
    keys: set[str],
) -> Dict[str, List[Tuple[Path, int]]]:
    locations = defaultdict(list)
    if not keys:
        return locations
    for line_no, line in enumerate(
        yaml_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if not stripped.endswith(":"):
            continue
        key = stripped[:-1].strip("'\"")
        if key in keys:
            locations[key].append((yaml_file, line_no))
    return locations


if __name__ == "__main__":
    sys.exit(main())
