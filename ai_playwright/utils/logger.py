from datetime import datetime
from pathlib import Path

from loguru import logger

logger.remove()
_FILE_SINK_ID: int | None = None

log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> [<level>{level}</level>] "
    "<white>{message}</white> (<cyan>{file}:{line}</cyan>)"
)
logger.add(
    sink=lambda msg: print(msg),
    format=log_format,
    level="ERROR",
    colorize=True,
)


def configure_file_logger(log_dir: str | Path = "logs") -> None:
    global _FILE_SINK_ID
    if _FILE_SINK_ID is not None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    _FILE_SINK_ID = logger.add(
        sink=log_file,
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8",
        delay=True,
    )


__all__ = ["logger", "configure_file_logger"]
