"""
元素检查相关的混入类
"""

from utils.logger import logger


class ElementCheckerMixin:
    """元素检查混入类"""
    
    def check_element_exists(self, selector: str) -> bool:
        """检查元素是否存在"""
        try:
            return self._locator(selector).first.is_attached()
        except Exception as e:
            logger.debug(f"元素存在性检查失败 {selector}: {e}")
            return False

    def check_element_visible(self, selector: str) -> bool:
        """检查元素是否可见"""
        try:
            return self._locator(selector).first.is_visible()
        except Exception as e:
            logger.debug(f"元素可见性检查失败 {selector}: {e}")
            return False

    def check_element_enabled(self, selector: str) -> bool:
        """检查元素是否启用"""
        try:
            return self._locator(selector).first.is_enabled()
        except Exception as e:
            logger.debug(f"元素启用状态检查失败 {selector}: {e}")
            return False

    def get_element_text_content(self, selector: str) -> str:
        """获取元素文本内容"""
        try:
            return self._locator(selector).first.inner_text()
        except Exception as e:
            logger.debug(f"元素文本获取失败 {selector}: {e}")
            return ""

    def get_element_attribute_value(self, selector: str, attr_name: str):
        """获取元素属性值"""
        try:
            return self._locator(selector).first.get_attribute(attr_name)
        except Exception as e:
            logger.debug(f"元素属性获取失败 {selector}.{attr_name}: {e}")
            return None

    def get_element_count(self, selector: str) -> int:
        """获取匹配元素的数量"""
        try:
            return self._locator(selector).count()
        except Exception as e:
            logger.debug(f"元素计数失败 {selector}: {e}")
            return 0