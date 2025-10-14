"""框架常量定义"""

# 超时配置
DEFAULT_TIMEOUT = 10000  # 默认超时时间(毫秒)
DEFAULT_TYPE_DELAY = 100  # 默认输入延迟(毫秒)
DEFAULT_POLLING = 500  # 默认轮询间隔(毫秒)

# 路径配置
SCREENSHOT_DIR = "./evidence/screenshots"  # 截图保存目录
LOGS_DIR = "./logs"  # 日志目录
REPORTS_DIR = "./reports"  # 报告目录

# 浏览器配置
DEFAULT_BROWSER = "chromium"
DEFAULT_VIEWPORT = {"width": 1920, "height": 1080}

# 性能配置
MAX_SCREENSHOT_SIZE_MB = 5  # 最大截图大小(MB)
PERFORMANCE_MONITORING_INTERVAL = 5.0  # 性能监控间隔(秒)
