from datetime import datetime
from pathlib import Path

from loguru import logger

# 移除默认的 sink
logger.remove()

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 生成日志文件名
log_file = log_dir / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
# 通用格式化字符串
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> [<level>{level}</level>] <white>{message}</white> (<cyan>{file}:{line}</cyan>)"
logger.add(
    sink=lambda msg: print(msg),
    format=log_format,  # 使用通用格式
    level="ERROR",
    colorize=True,
)
# 添加文件输出 (DEBUG 级别及以上)
try:
    logger.add(
        sink=log_file,
        format=log_format,  # 使用通用格式
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8",
        delay=True,
    )
except Exception as e:
    print(f"Error adding file logger: {e}")

# 导出 logger 实例供其他模块使用
__all__ = ["logger"]
