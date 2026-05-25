from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from utils.config import BaseInfo
from utils.logger import logger
from utils.variable_manager import VariableManager
from utils.yaml_handler import YamlHandler

yaml_handler = YamlHandler()


class TestCaseDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    data_name: str | None = None
    fixtures: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class TestCaseFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    test_cases: list[TestCaseDefinition] = Field(default_factory=list)


class LoadData:
    def __init__(self, base_info: BaseInfo):
        self.yaml = yaml_handler
        self.test_data_dir = base_info.test_dir
        self.test_env = base_info.env
        self.vars = self._load_vars()

    def _load_test_datas(self) -> Dict[str, Any]:
        """加载现有的YAML配置文件"""
        return self.yaml.merge_yaml_files(self.test_data_dir + "/data").get(
            "test_data", {}
        )

    def _load_elements(self) -> Dict[str, Any]:
        """加载现有的YAML配置文件"""
        elements_dir = Path(self.test_data_dir) / "elements"
        if not elements_dir.exists():
            return {}
        return self.yaml.merge_yaml_files(elements_dir).get("elements", {})

    def _load_vars(self) -> Dict[str, Any]:
        return self._merge_vars(self.yaml.load_yaml_dir(self.test_data_dir + "/vars"))

    def _merge_vars(self, vars_datas) -> Dict[str, Any]:
        """合并vars数据到test_data"""
        result = {}
        for vars_data in vars_datas:
            if vars_data:
                for env in ["dev", "test", "stage", "prod"]:
                    if env in vars_data.keys():
                        if env == self.test_env:
                            vars_data.update(vars_data.pop(env))
                        else:
                            vars_data.pop(env)
            result.update(vars_data)
        return result

    def return_data(self):
        return {
            "elements": self._load_elements(),
            "test_datas": self._load_test_datas(),
        }

    def return_vars(self):
        return self.vars

    def return_modules(self):
        return self.yaml.merge_yaml_files(Path(self.test_data_dir) / "modules")


@lru_cache(maxsize=16)
def _load_data_for(test_dir: str, env: str) -> LoadData:
    base_info = BaseInfo()
    if base_info.test_dir != test_dir or base_info.env != env:
        base_info.test_dir = test_dir
        base_info.env = env
    return LoadData(base_info)


def run_test_data():
    base_info = BaseInfo()
    load_data = _load_data_for(base_info.test_dir, base_info.env)
    test_data = load_data.return_data()
    set_global_variables(load_data, base_info)
    return test_data


def set_global_variables(
    load_data: LoadData | None = None, base_info: BaseInfo | None = None
):
    base_info = base_info or BaseInfo()
    load_data = load_data or _load_data_for(base_info.test_dir, base_info.env)
    logger.info("当前测试环境: " + base_info.env)
    variable_manager = VariableManager()
    if variables := load_data.return_vars():
        for var_name, var_value in variables.items():
            variable_manager.set_variable(var_name, var_value, "temp")


def load_test_cases(file_path):
    payload = yaml_handler.load_yaml(file_path) or {}
    try:
        case_file = TestCaseFile.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"测试用例文件结构不合法: {file_path}\n{exc}") from exc
    return [case.model_dump(exclude_none=True) for case in case_file.test_cases]


def load_modules():
    base_info = BaseInfo()
    return _load_data_for(base_info.test_dir, base_info.env).return_modules()
