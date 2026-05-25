# config.py
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict
from pydantic_settings import BaseSettings

from src.utils import singleton
from utils.logger import logger
from utils.yaml_handler import YamlHandler


class Browser(str, Enum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class Environment(str, Enum):
    DEV = "dev"
    TEST = "test"
    STAGE = "stage"
    PROD = "prod"


class Project(str, Enum):
    DEMO = "demo"
    HOLO_LIVE = "hololive"  # 确保枚举值使用小写
    ASSISTANT = "assistant"
    OTHER_PROJECT = "other_project"
    MARKETING = "marketing"
    CHANNEL_PAGE = "channel_page"
    CRM_STORE_BACKEND = "crm_store_backend"
    AHOH = "ahoh"
    DA_PING = "da_ping"
    CRM_BACKEND = "crm_backend"
    CAR_STORE_C = "car_store_c"


@singleton
class Config(BaseSettings):
    marker: Optional[str] = None
    keyword: Optional[str] = None
    headed: bool = True  # 将默认值改为 True
    browser: Browser = Browser.CHROMIUM
    env: Environment = Environment.PROD
    project: Project = Project.DEMO
    base_url: str = ""
    test_dir: str = ""
    browser_config: Optional[dict] = None
    test_file: str = ""

    model_config = SettingsConfigDict(case_sensitive=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_config()
        self._update_config_based_on_env_and_project()

    def reconfigure(self, **kwargs):
        if ("env" in kwargs or "project" in kwargs) and "base_url" not in kwargs:
            self.base_url = ""
        for key, value in kwargs.items():
            if value is None or not hasattr(self, key):
                continue
            setattr(self, key, value)
        self.validate_config()
        self._update_config_based_on_env_and_project()

    def validate_config(self):
        """验证配置是否有效"""
        self.browser = self._coerce_enum(self.browser, Browser, "browser")
        self.env = self._coerce_enum(self.env, Environment, "env")
        if not isinstance(self.project, Project):
            try:
                self.project = Project(self.project.lower())  # 转换为小写再验证
            except ValueError:
                valid_projects = ", ".join([p.value for p in Project])
                raise ValueError(
                    f"Invalid project: {self.project}. Valid projects are: {valid_projects}"
                )

    @staticmethod
    def _coerce_enum(value, enum_type, field_name: str):
        if isinstance(value, enum_type):
            return value
        try:
            return enum_type(str(value).lower())
        except ValueError:
            valid_values = ", ".join([item.value for item in enum_type])
            raise ValueError(
                f"Invalid {field_name}: {value}. Valid values are: {valid_values}"
            )

    def _update_config_based_on_env_and_project(self):
        """根据环境和项目更新配置"""
        explicit_base_url = self.base_url
        try:
            env_config = YamlHandler().load_yaml(Path("config/env_config.yaml"))
            if not env_config or not isinstance(env_config, dict):
                raise ValueError("Invalid YAML configuration")

            projects = env_config.get("projects", {})
            if not projects or not isinstance(projects, dict):
                raise ValueError("Missing or invalid projects configuration")

            # 从 projects 字典中获取项目配置
            project_config = projects.get(self.project.value)
            if not project_config:
                raise ValueError(f"Project {self.project.value} not found in config")

            # 获取环境URL
            environments = project_config.get("environments", {})
            self.base_url = explicit_base_url or environments.get(self.env.value)
            if not self.base_url:
                raise ValueError(
                    f"Environment {self.env.value} not found for project {self.project.value}"
                )

            # 设置测试数据目录
            self.test_dir = project_config.get("test_dir")
            self.browser_config = project_config.get(
                "browser_config", {"viewport": {"width": 1920, "height": 1080}}
            )

            self._publish_environment()

        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise ValueError(f"Configuration error: {str(e)}")

    def configure_environment(self):
        """配置运行环境"""
        self._publish_environment()

    def _publish_environment(self):
        os.environ["PWHEADED"] = "1" if self.headed else "0"
        os.environ["BROWSER"] = self.browser.value
        os.environ["TEST_ENV"] = self.env.value
        os.environ["TEST_PROJECT"] = self.project.value
        os.environ["BASE_URL"] = self.base_url
        os.environ["TEST_DIR"] = str(self.test_dir)


class BaseInfo:
    def __init__(self):
        config = Config()
        if not os.environ.get("TEST_DIR"):
            config.configure_environment()
        self.test_dir = os.environ.get("TEST_DIR", config.test_dir)
        self.base_dir = Path.cwd()
        self.env = os.environ.get("TEST_ENV", config.env.value)
        self.project = os.environ.get("TEST_PROJECT", config.project.value)
