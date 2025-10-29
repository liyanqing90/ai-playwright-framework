import os
from pathlib import Path
from typing import Dict, Any

from ruamel.yaml import YAML

from utils.logger import logger


def get_yaml_files(directory: str) -> list[Any] | None:
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error(f"指定的路径 {directory} 不是一个有效的目录")
        return None
    yaml_files = [file for file in list(dir_path.glob("**/*.yaml"))]
    return yaml_files


class YamlHandler:
    def __init__(self):
        self.yaml = YAML(typ="safe")
        # 禁用日期自动解析，避免 2025-12-09 被解析为 datetime.date 对象
        # 将 timestamp 标签映射到字符串构造器，这样所有日期都会保持为字符串类型
        if 'tag:yaml.org,2002:timestamp' in self.yaml.constructor.yaml_constructors:
            self.yaml.constructor.yaml_constructors['tag:yaml.org,2002:timestamp'] = \
                self.yaml.constructor.yaml_constructors['tag:yaml.org,2002:str']

    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:

                return self.yaml.load(f)
            except Exception as e:
                logger.error(f"YAML文件解析错误: {file_path}, 错误详情: {str(e)}")
                raise Exception(f"YAML文件解析错误: {file_path}, 错误详情: {str(e)}")

    def merge_yaml_files(self, dir_path: Path):
        yaml_datas = self.load_yaml_dir(dir_path)
        result = {}
        for key in set(k for d in yaml_datas for k in d.keys()):
            value = None
            for yaml_data in yaml_datas:
                if not isinstance(yaml_data, dict):
                    raise ValueError(f"YAML文件内容不是字典格式")
                if key in yaml_data:
                    temp = yaml_data[key]
                    if not value:
                        value = temp
                    else:
                        if isinstance(temp, dict):
                            value.update(temp)
                        else:
                            value += temp
            result[key] = value

        return result

    def load_yaml_dir(self, file_path):
        yaml_files = get_yaml_files(file_path)
        result = []
        if not yaml_files:
            return result
        # 直接顺序加载并合并（后面的文件覆盖前面的）
        for yaml_file in yaml_files:
            if data := self.load_yaml(yaml_file):
                if not isinstance(data, dict):
                    raise ValueError(f"YAML文件 {yaml_file} 内容不是字典格式")
                result.append(data)
        return result

    def save_to_yaml(self, data_dict, output_dir, filename):
        """
        将字典写入 YAML 文件。

        Args:
            data_dict: 要写入的字典。
            output_dir: 输出目录的路径。
            filename: YAML 文件的名称（不含扩展名）。
        """

        self.yaml.indent(mapping=2, sequence=4, offset=2)  # 可选：设置 YAML 缩进样式
        self.yaml.preserve_quotes = True  # 可选: 保留引号
        self.yaml.default_flow_style = False  # 关键设置：禁用流式风格

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(output_dir) / f"{filename}.yaml"
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(filepath, "w", encoding="utf-8") as f:
            self.yaml.dump(data_dict, f)
