"""
定义所有支持的步骤操作类型
"""


class StepAction:
    """操作类型定义"""

    # 基础操作
    NAVIGATE = ["navigate", "goto", "打开", "访问"]
    CLICK = ["click", "点击"]
    FILL = ["fill", "输入"]
    PRESS_KEY = ["press_key", "按键"]
    WAIT = ["wait", "等待"]

    # 执行Python文件
    EXECUTE_PYTHON = ["execute_python", "执行Python"]
    # 断言相关
    HARD_ASSERT_TEXT = ["hard_assert", "硬断言"]
    ASSERT_VISIBLE = ["assert_visible", "验证可见"]
    ASSERT_BE_HIDDEN = ["assert_be_hidden", "验证隐藏"]
    ASSERT_TEXT = ["assert_text", "assertion", "验证文本", "验证", "verify"]
    ASSERT_ATTRIBUTE = ["assert_attribute", "验证属性"]
    ASSERT_URL = ["assert_url", "验证URL"]
    ASSERT_TITLE = ["assert_title", "验证标题"]
    ASSERT_TITLE_CONTAINS = ["assert_title_contains", "验证标题包含"]
    ASSERT_ELEMENT_COUNT = ["assert_element_count", "验证元素数量"]
    ASSERT_TEXT_CONTAINS = ["assert_text_contains", "验证包含文本"]
    ASSERT_URL_CONTAINS = ["assert_url_contains", "验证URL包含"]
    ASSERT_EXISTS = ["assert_exists", "验证存在"]
    ASSERT_NOT_EXISTS = ["assert_not_exists", "验证不存在"]
    ASSERT_ENABLED = ["assert_enabled", "验证启用"]
    ASSERT_DISABLED = ["assert_disabled", "验证禁用"]
    ASSERT_VALUE = ["assert_value", "验证值"]

    ASSERT_HAVE_VALUES = ["assert_have_values", "验证多个值"]

    # 存储相关
    STORE_VARIABLE = ["store_variable", "存储变量"]
    STORE_TEXT = ["store_text", "存储文本"]
    STORE_ATTRIBUTE = ["store_attribute", "存储属性"]
    STORE_INPUT_VALUE = ["store_input_value", "存储输入框值"]

    # 等待相关
    WAIT_FOR_ELEMENT_HIDDEN = ["wait_for_element_hidden", "等待元素消失"]
    WAIT_FOR_NETWORK_IDLE = ["wait_for_network_idle", "等待网络空闲"]
    WAIT_FOR_ELEMENT_CLICKABLE = ["wait_for_element_clickable", "等待元素可点击"]
    WAIT_FOR_ELEMENT_TEXT = ["wait_for_element_text", "等待元素文本"]
    WAIT_FOR_ELEMENT_COUNT = ["wait_for_element_count", "等待元素数量"]

    # 其他操作
    REFRESH = ["refresh", "刷新"]
    PAUSE = ["pause", "暂停"]
    UPLOAD = ["upload", "上传", "上传文件"]
    HOVER = ["hover", "悬停"]
    DOUBLE_CLICK = ["double_click", "双击"]
    RIGHT_CLICK = ["right_click", "右键点击"]
    SELECT = ["select", "选择"]
    DRAG_AND_DROP = ["drag_and_drop", "拖拽"]
    GET_VALUE = ["get_value", "获取值"]
    SCROLL_INTO_VIEW = ["scroll_into_view", "滚动到元素", "向下滑动到"]
    SCROLL_TO = ["scroll_to", "滚动到位置", "滑到顶部", "滑到底部"]
    FOCUS = ["focus", "聚焦"]
    BLUR = ["blur", "失焦"]
    TYPE = ["type", "模拟输入"]
    CLEAR = ["clear", "清空"]
    ENTER_FRAME = ["enter_frame", "进入框架"]
    ACCEPT_ALERT = ["accept_alert", "接受弹框"]
    DISMISS_ALERT = ["dismiss_alert", "取消弹框"]
    EXPECT_POPUP = ["expect_popup", "弹出tab"]
    SWITCH_WINDOW = ["switch_window", "切换窗口"]
    CLOSE_WINDOW = ["close_window", "关闭窗口"]
    WAIT_FOR_NEW_WINDOW = ["wait_for_new_window", "等待新窗口"]
    SAVE_ELEMENT_COUNT = ["save_ele_count", "存储元素数量"]
    EXECUTE_SCRIPT = ["execute_script", "执行脚本"]
    CAPTURE_SCREENSHOT = ["capture", "截图"]
    MANAGE_COOKIES = ["cookies", "Cookie操作"]
    TAB_SWITCH = ["switch_tab", "切换标签页"]
    DOWNLOAD_FILE = ["download", "下载文件"]
    DOWNLOAD_VERIFY = ["verify_download", "验证下载"]
    FAKER = ["faker", "生成数据"]
    GET_ALL_ELEMENTS = ["get_all_elements", "获取所有元素"]
    KEYBOARD_SHORTCUT = ["keyboard_shortcut", "键盘快捷键"]
    KEYBOARD_PRESS = ["keyboard_press", "全局按键"]
    KEYBOARD_TYPE = ["keyboard_type", "全局输入"]
    AI_STEP = ["ai_step", "observe", "AI操作", "智能操作"]

    # 流程控制操作
    USE_MODULE = ["use_module", "使用模块"]
    IF_CONDITION = ["if", "如果"]
    FOR_EACH = ["for_each", "循环"]

    # 接口监测相关
    MONITOR_REQUEST = ["monitor_request", "监测请求"]
    MONITOR_RESPONSE = ["monitor_response", "监测响应"]

    # 不需要selector的操作
    NO_SELECTOR_ACTIONS = (
        NAVIGATE
        + ASSERT_URL
        + ASSERT_TITLE
        + ASSERT_TITLE_CONTAINS
        + ASSERT_URL_CONTAINS
        + EXECUTE_PYTHON
        + WAIT
        + WAIT_FOR_NETWORK_IDLE
        + REFRESH
        + PAUSE
        + CLOSE_WINDOW
        + WAIT_FOR_NEW_WINDOW
        + SAVE_ELEMENT_COUNT
        + EXECUTE_SCRIPT
        + CAPTURE_SCREENSHOT
        + MANAGE_COOKIES
        + TAB_SWITCH
        + DOWNLOAD_VERIFY
        + FAKER
        + KEYBOARD_SHORTCUT
        + KEYBOARD_PRESS
        + KEYBOARD_TYPE
        + AI_STEP
        + USE_MODULE
        + IF_CONDITION
        + FOR_EACH
        + SWITCH_WINDOW
    )
