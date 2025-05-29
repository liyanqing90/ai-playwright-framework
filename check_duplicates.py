from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from utils.logger import logger
from utils.yaml_handler import YamlHandler


def get_line_numbers(content: str, target: str) -> List[int]:
    """获取目标字符串在内容中的所有行号"""
    line_numbers = []
    for i, line in enumerate(content.splitlines(), 1):
        if target in line:
            line_numbers.append(i)
    return line_numbers


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
    yaml_handler = YamlHandler()

    for yaml_file in cases_dir.glob("**/*.yaml"):
        content = yaml_handler.load_yaml(yaml_file)
        with open(yaml_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        if isinstance(content, dict) and "test_cases" in content:
            # 记录每个用例名在当前文件中出现的所有行号
            name_lines = defaultdict(list)
            for i, line in enumerate(file_content.splitlines(), 1):
                if "name:" in line:
                    for case in content["test_cases"]:
                        if "name" in case:
                            case_name = case["name"]
                            # 使用更灵活的匹配，确保匹配到 "name: xxx" 这种格式
                            # 确保匹配包含完整用例名，避免匹配到子字符串
                            pattern = f"name: {case_name}"
                            if pattern in line and (
                                line.strip() == pattern
                                or line.strip().startswith(pattern + " ")
                                or line.strip().endswith(pattern)
                            ):
                                name_lines[case_name].append((yaml_file, i))

            # 找出当前文件中的重复用例名
            for name, locations in name_lines.items():
                if len(locations) > 1:  # 如果同一个用例名出现多次
                    duplicates[name].extend(locations)
                else:  # 如果只出现一次，也记录下来，以便后续检查跨文件重复
                    duplicates[name].extend(locations)

    # 只保留真正的重复项
    return {
        name: locations for name, locations in duplicates.items() if len(locations) > 1
    }


def check_data_duplicates(data_dir: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """检查data目录下的测试数据用例名称重复"""
    duplicates = defaultdict(list)
    yaml_handler = YamlHandler()

    for yaml_file in data_dir.glob("**/*.yaml"):
        content = yaml_handler.load_yaml(yaml_file)
        with open(yaml_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        if isinstance(content, dict) and "test_data" in content:
            # 记录每个测试数据名在当前文件中出现的所有行号
            name_lines = defaultdict(list)
            for i, line in enumerate(file_content.splitlines(), 1):
                for test_name in content["test_data"].keys():
                    # 使用精确匹配，确保完全匹配key
                    if line.strip() == f"{test_name}:":
                        name_lines[test_name].append((yaml_file, i))

            # 找出当前文件中的重复数据名
            for name, locations in name_lines.items():
                if len(locations) > 1:  # 如果同一个数据名出现多次
                    duplicates[name].extend(locations)
                else:  # 如果只出现一次，也记录下来，以便后续检查跨文件重复
                    duplicates[name].extend(locations)

    # 只保留真正的重复项
    return {
        name: locations for name, locations in duplicates.items() if len(locations) > 1
    }


def check_elements_duplicates(elements_dir: Path) -> Dict[str, List[Tuple[Path, int]]]:
    """检查elements目录下的元素名称重复"""
    duplicates = defaultdict(list)
    element_names = defaultdict(list)
    yaml_handler = YamlHandler()

    for yaml_file in elements_dir.glob("**/*.yaml"):
        content = yaml_handler.load_yaml(yaml_file)
        with open(yaml_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        if isinstance(content, dict) and "elements" in content:
            for element_name in content["elements"].keys():
                # 查找元素key所在的行号
                for i, line in enumerate(file_content.splitlines(), 1):
                    if f"{element_name}:" in line:  # 使用冒号来匹配key
                        element_names[element_name].append((yaml_file, i))
                        break

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
            print(f"\n❌ {project_name} 项目中发现重复的用例名称：")
            logger.info(f"\n{project_name} 项目中发现重复的用例名称：")
            for name, locations in case_duplicates.items():
                print(f'  "{name}" 在 {format_duplicate_locations(locations)}')
                logger.info(f'"{name}" 在 {format_duplicate_locations(locations)}')

    # 检查data目录
    data_dir = project_dir / "data"
    if data_dir.exists():
        data_duplicates = check_data_duplicates(data_dir)
        if data_duplicates:
            has_duplicates = True
            project_has_duplicates = True
            print(f"\n❌ {project_name} 项目中发现重复的测试数据名称：")
            logger.info(f"\n{project_name} 项目中发现重复的测试数据名称：")
            for name, locations in data_duplicates.items():
                print(f'  "{name}" 在 {format_duplicate_locations(locations)}')
                logger.info(f'"{name}" 在 {format_duplicate_locations(locations)}')

    # 检查elements目录
    elements_dir = project_dir / "elements"
    if elements_dir.exists():
        element_duplicates = check_elements_duplicates(elements_dir)
        if element_duplicates:
            has_duplicates = True
            project_has_duplicates = True
            print(f"\n❌ {project_name} 项目中发现重复的元素名称：")
            logger.info(f"\n{project_name} 项目中发现重复的元素名称：")
            for name, locations in element_duplicates.items():
                print(f'  "{name}" 在 {format_duplicate_locations(locations)}')
                logger.info(f'"{name}" 在 {format_duplicate_locations(locations)}')

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
        print("✅ 所有项目中均未发现重复项")
        logger.info("\n所有项目中均未发现重复项")


if __name__ == "__main__":
    main()
