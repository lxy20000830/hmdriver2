# -*- coding: utf-8 -*-

import re
from typing import Dict
from lxml import etree
from functools import cached_property

from . import logger
from .proto import Bounds
from .driver import Driver
from .utils import delay, parse_bounds
from .exception import XmlElementNotFoundError


class _XPath:
    def __init__(self, d: Driver):
        self._d = d

    def __call__(self, xpath: str) -> '_XMLElement':

        hierarchy: Dict = self._d.dump_hierarchy()
        if not hierarchy:
            raise RuntimeError("hierarchy is empty")

        xml = _XPath._json2xml(hierarchy)
        result = xml.xpath(xpath)

        if len(result) > 0:
            node = result[0]
            raw_bounds: str = node.attrib.get("bounds")  # [832,1282][1125,1412]
            bounds: Bounds = parse_bounds(raw_bounds)
            logger.debug(f"{xpath} Bounds: {bounds}")
            _xe = _XMLElement(bounds, self._d)
            setattr(_xe, "attrib_info", node.attrib)
            setattr(_xe, "_xml_node", node)
            return _xe

        return _XMLElement(None, self._d)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove XML-incompatible control characters."""
        return re.sub(r'[\x00-\x1F\x7F]', '', text)

    @staticmethod
    def _json2xml(hierarchy: Dict) -> etree.Element:
        """Convert JSON-like hierarchy to XML."""
        attributes = hierarchy.get("attributes", {})

        # 过滤所有属性的值，确保无非法字符
        cleaned_attributes = {k: _XPath._sanitize_text(str(v)) for k, v in attributes.items()}

        tag = cleaned_attributes.get("type", "orgRoot") or "orgRoot"
        xml = etree.Element(tag, attrib=cleaned_attributes)

        children = hierarchy.get("children", [])
        for item in children:
            xml.append(_XPath._json2xml(item))

        return xml


class _XMLElement:
    def __init__(self, bounds: Bounds, d: Driver):
        self.bounds = bounds
        self._d = d
        self.attrib_info = {}
        self._xml_node = None

    def _verify(self):
        if not self.bounds:
            raise XmlElementNotFoundError("xpath not found")

    @cached_property
    def center(self):
        self._verify()
        return self.bounds.get_center()

    def exists(self) -> bool:
        return self.bounds is not None

    def get_text(self) -> str:
        """
        显式获取文本的方法，优先级：
        1. 从attrib_info的text字段获取
        2. 从XML节点的text()方法获取（兼容部分特殊节点）
        3. 返回空字符串
        """
        if not self.exists():
            logger.warning("元素不存在，无法获取文本")
            return ""

        # 优先从属性中获取文本
        text = self.attrib_info.get("text", "").strip()
        if text:
            return text

        # 备用：从XML节点直接获取文本（兼容特殊场景）
        if self._xml_node is not None:
            try:
                node_text = self._xml_node.text or ""
                return node_text.strip()
            except Exception as e:
                logger.warning(f"从XML节点获取文本失败: {e}")

        return ""

    @property
    @delay
    def text(self) -> str:
        """
        简化的text属性，直接调用get_text()，和UiObject的text属性用法保持一致
        """
        return self.get_text()

    @delay
    def click(self):
        x, y = self.center.x, self.center.y
        self._d.click(x, y)

    @delay
    def click_if_exists(self):

        if not self.exists():
            logger.debug("click_exist: xpath not found")
            return

        x, y = self.center.x, self.center.y
        self._d.click(x, y)

    @delay
    def double_click(self):
        x, y = self.center.x, self.center.y
        self._d.double_click(x, y)

    @delay
    def long_click(self):
        x, y = self.center.x, self.center.y
        self._d.long_click(x, y)

    @delay
    def input_text(self, text):
        self.click()
        self._d.input_text(text)

    @property
    @delay
    def info(self) -> dict:
        if hasattr(self, 'attrib_info'):
            return getattr(self, 'attrib_info')
        else:
            logger.warning("the attribute <attrib_info> does not exists！")
            return {}

    @property
    @delay
    def text(self) -> str:
        return self.info.get("text")
