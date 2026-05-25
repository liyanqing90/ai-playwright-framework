import socket
import subprocess

from configobj import ConfigObj


def singleton(cls):
    _instance = {}

    def inner(**kwargs):  # 修改这里，接收关键字参数
        if cls not in _instance:
            _instance[cls] = cls(**kwargs)  # 修改这里，传递关键字参数给 cls 的构造函数
        elif kwargs and hasattr(_instance[cls], "reconfigure"):
            _instance[cls].reconfigure(**kwargs)
        return _instance[cls]

    return inner


def set_ini(key, value):
    config = ConfigObj("pytest.ini", encoding="UTF8")
    config["pytest"][key] = value
    config.write()


def get_ini(key):
    config = ConfigObj("pytest.ini", encoding="UTF8")
    return config["pytest"][key]


def save_info(project, env, reporter):
    reporter = (
        "Jenkins"
        if reporter == "jenkins"
        else subprocess.getoutput("git config user.name") or socket.gethostname()
    )
    config = ConfigObj("pytest.ini", encoding="UTF8")
    config["pytest"]["reporter"] = reporter
    config["pytest"]["project"] = project
    config["pytest"]["base_env"] = env
    config["pytest"]["testpaths"] = f"tests/{project}/case/"
    if env == "test":
        url = "magicdoor-test.com/"
    elif env == "online":
        url = "magicdoor.com/"
    config.write()
