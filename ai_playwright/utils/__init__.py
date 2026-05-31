"""Shared framework utilities."""


def singleton(cls):
    _instance = {}

    def inner(**kwargs):
        if cls not in _instance:
            _instance[cls] = cls(**kwargs)
        elif kwargs and hasattr(_instance[cls], "reconfigure"):
            _instance[cls].reconfigure(**kwargs)
        return _instance[cls]

    return inner
