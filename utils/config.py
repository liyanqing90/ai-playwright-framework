# config.py
import os
from enum import Enum
from pathlib import Path
from typing import Optional

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

    class Config:
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_config_based_on_env_and_project()
        self.validate_config()

    def validate_config(self):
        """验证配置是否有效"""
        if not isinstance(self.project, Project):
            try:
                self.project = Project(self.project.lower())  # 转换为小写再验证
            except ValueError:
                valid_projects = ", ".join([p.value for p in Project])
                raise ValueError(
                    f"Invalid project: {self.project}. Valid projects are: {valid_projects}"
                )

    def _update_config_based_on_env_and_project(self):
        """根据环境和项目更新配置"""
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
            self.base_url = environments.get(self.env.value)
            if not self.base_url:
                raise ValueError(
                    f"Environment {self.env.value} not found for project {self.project.value}"
                )

            # 设置测试数据目录
            self.test_dir = project_config.get("test_dir")
            self.browser_config = project_config.get(
                "browser_config", {"viewport": {"width": 1920, "height": 1080}}
            )

            # 设置环境变量
            os.environ["BASE_URL"] = self.base_url
            os.environ["TEST_DIR"] = str(self.test_dir)
            # os.environ['BROWSER_CONFIG'] = str(browser_config)

        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise ValueError(f"Configuration error: {str(e)}")

    def configure_environment(self):
        """配置运行环境"""
        os.environ["PWHEADED"] = "1" if self.headed else "0"
        os.environ["BROWSER"] = self.browser.value
        os.environ["TEST_ENV"] = self.env.value
        os.environ["TEST_PROJECT"] = self.project.value


class BaseInfo:
    def __init__(self):
        self.test_dir = os.environ["TEST_DIR"]
        self.base_dir = Path.cwd()
        self.env = os.environ["TEST_ENV"]
        self.project = os.environ["TEST_PROJECT"]
