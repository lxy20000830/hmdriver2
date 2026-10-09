# -*- coding: utf-8 -*-

import enum
import re
import time
from typing import List, Union, TYPE_CHECKING

from . import logger
from .utils import delay, parse_bounds
from ._client import HmClient
from .exception import ElementNotFoundError
from .proto import ComponentData, ByData, HypiumResponse, Point, Bounds, ElementInfo, NodeData

if TYPE_CHECKING:
    from .driver import Driver


class ByType(enum.Enum):
    id = "id"
    key = "key"
    text = "text"
    type = "type"
    description = "description"
    clickable = "clickable"
    longClickable = "longClickable"
    scrollable = "scrollable"
    enabled = "enabled"
    focused = "focused"
    selected = "selected"
    checked = "checked"
    checkable = "checkable"
    isBefore = "isBefore"
    isAfter = "isAfter"

    # fuzzy matching (implemented client-side via dump_hierarchy)
    textContains = "textContains"
    textStartsWith = "textStartsWith"
    textEndsWith = "textEndsWith"
    textMatches = "textMatches"
    descriptionContains = "descriptionContains"
    descriptionStartsWith = "descriptionStartsWith"
    descriptionEndsWith = "descriptionEndsWith"
    descriptionMatches = "descriptionMatches"
    idContains = "idContains"
    idStartsWith = "idStartsWith"
    idEndsWith = "idEndsWith"
    idMatches = "idMatches"
    keyContains = "keyContains"
    keyStartsWith = "keyStartsWith"
    keyEndsWith = "keyEndsWith"
    keyMatches = "keyMatches"
    typeContains = "typeContains"
    typeStartsWith = "typeStartsWith"
    typeEndsWith = "typeEndsWith"
    typeMatches = "typeMatches"

    @classmethod
    def verify(cls, value):
        return any(value == item.value for item in cls)


# all fuzzy matching keys, handled by client-side hierarchy dump
FUZZY_KEYS = frozenset({
    "textContains", "textStartsWith", "textEndsWith", "textMatches",
    "descriptionContains", "descriptionStartsWith", "descriptionEndsWith", "descriptionMatches",
    "idContains", "idStartsWith", "idEndsWith", "idMatches",
    "keyContains", "keyStartsWith", "keyEndsWith", "keyMatches",
    "typeContains", "typeStartsWith", "typeEndsWith", "typeMatches",
})

# boolean attributes in hierarchy dump (values are "true"/"false"/"" strings)
_BOOL_ATTRS = frozenset({
    "clickable", "longClickable", "scrollable", "enabled",
    "focused", "selected", "checked", "checkable",
})

_FUZZY_SUFFIXES = ("Contains", "StartsWith", "EndsWith", "Matches")


def _norm_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


class UiObject:
    DEFAULT_TIMEOUT = 2

    def __init__(self, client: HmClient, driver: "Driver" = None, **kwargs) -> None:
        self._client = client
        self._driver = driver
        self._raw_kwargs = kwargs

        self._index = kwargs.pop("index", 0)
        self._isBefore = kwargs.pop("isBefore", False)
        self._isAfter = kwargs.pop("isAfter", False)

        self._kwargs = kwargs
        self.__verify()

        self._component: Union[ComponentData, NodeData, None] = None  # cache

    def __str__(self) -> str:
        return f"UiObject [{self._raw_kwargs}]"

    def __verify(self):
        for k, v in self._kwargs.items():
            if not ByType.verify(k):
                raise ReferenceError(f"{k} is not allowed.")

        if (self._isBefore or self._isAfter) and self.__is_client_match():
            raise ValueError("isBefore/isAfter is not supported with fuzzy matching keys")

    def __is_client_match(self) -> bool:
        return any(k in FUZZY_KEYS for k in self._kwargs)

    @property
    def count(self) -> int:
        eleements = self.__find_components()
        return len(eleements) if eleements else 0

    def __len__(self):
        return self.count

    def exists(self, retries: int = 2, wait_time=1) -> bool:
        obj = self.find_component(retries, wait_time)
        return True if obj else False

    def __set_component(self, component: Union[ComponentData, NodeData]):
        self._component = component

    def find_component(self, retries: int = 1, wait_time=1) -> Union[ComponentData, NodeData]:
        for attempt in range(retries):
            components = self.__find_components()
            if components and self._index < len(components):
                self.__set_component(components[self._index])
                return self._component

            if attempt < retries:
                time.sleep(wait_time)
                logger.info(f"Retry found element {self}")

        return None

    # useless
    def __find_component(self) -> Union[ComponentData, None]:
        by: ByData = self.__get_by()
        resp: HypiumResponse = self._client.invoke("Driver.findComponent", args=[by.value])
        if not resp.result:
            return None
        return ComponentData(resp.result)

    def __find_components(self) -> Union[List[ComponentData], List[NodeData], None]:
        if self.__is_client_match():
            return self.__find_components_client_side()

        by: ByData = self.__get_by()
        resp: HypiumResponse = self._client.invoke("Driver.findComponents", args=[by.value])
        if not resp.result:
            return None
        components: List[ComponentData] = []
        for item in resp.result:
            components.append(ComponentData(item))

        return components

    # ------------------------------------------------------------------
    # client-side fuzzy matching (dump_hierarchy based)
    # ------------------------------------------------------------------
    def __find_components_client_side(self) -> List[NodeData]:
        if not self._driver:
            raise RuntimeError("fuzzy matching requires a Driver instance, please use d(...) instead")

        hierarchy = self._driver.dump_hierarchy()
        if not hierarchy:
            return None

        results: List[NodeData] = []
        self.__walk_hierarchy(hierarchy, results)
        return results

    def __walk_hierarchy(self, node: dict, results: List[NodeData]):
        attrs = node.get("attributes", {})
        if attrs and self.__match_attributes(attrs):
            bounds = parse_bounds(attrs.get("bounds", ""))
            if bounds:
                results.append(NodeData(attrs, bounds))
        for child in node.get("children", []):
            self.__walk_hierarchy(child, results)

    def __match_attributes(self, attrs: dict) -> bool:
        for k, v in self._kwargs.items():
            if not self.__match_one(k, v, attrs):
                return False
        return True

    @staticmethod
    def __match_one(key: str, value, attrs: dict) -> bool:
        for suffix in _FUZZY_SUFFIXES:
            if key.endswith(suffix):
                attr = key[: -len(suffix)]
                raw = str(attrs.get(attr, ""))
                value = str(value)
                if suffix == "Contains":
                    return value in raw
                if suffix == "StartsWith":
                    return raw.startswith(value)
                if suffix == "EndsWith":
                    return raw.endswith(value)
                # Matches
                return re.search(value, raw) is not None

        raw = attrs.get(key, "")
        if key in _BOOL_ATTRS:
            return _norm_bool(raw) == bool(value)
        return str(raw) == str(value)

    # ------------------------------------------------------------------
    # protocol based matching (unchanged)
    # ------------------------------------------------------------------
    def __get_by(self) -> ByData:
        for k, v in self._kwargs.items():
            api = f"On.{k}"
            this = "On#seed"
            resp: HypiumResponse = self._client.invoke(api, this, args=[v])
            this = resp.result

        if self._isBefore:
            resp: HypiumResponse = self._client.invoke("On.isBefore", this="On#seed", args=[resp.result])

        if self._isAfter:
            resp: HypiumResponse = self._client.invoke("On.isAfter", this="On#seed", args=[resp.result])

        return ByData(resp.result)

    def __operate(self, api, args=[], retries: int = 2):
        if self.__is_client_match():
            return self.__client_operate(api, args, retries)

        if not self._component:
            if not self.find_component(retries):
                raise ElementNotFoundError(f"Element({self}) not found after {retries} retries")

        resp: HypiumResponse = self._client.invoke(api, this=self._component.value, args=args)
        return resp.result

    # ------------------------------------------------------------------
    # client-side matched element operations
    # ------------------------------------------------------------------
    def __client_operate(self, api, args=[], retries: int = 2):
        if not self._component:
            if not self.find_component(retries):
                raise ElementNotFoundError(f"Element({self}) not found after {retries} retries")

        node: NodeData = self._component
        attrs = node.attributes
        name = api.split(".", 1)[-1]  # "Component.click" -> "click"

        if name == "getId":
            return attrs.get("id", "")
        if name == "getKey":
            return attrs.get("key", "")
        if name == "getType":
            return attrs.get("type", "")
        if name == "getText":
            return attrs.get("text", "")
        if name == "getDescription":
            return attrs.get("description", "")
        if name in ("isSelected", "isChecked", "isEnabled", "isFocused",
                    "isCheckable", "isClickable", "isLongClickable", "isScrollable"):
            return _norm_bool(attrs.get(name[2:].lower(), ""))
        if name == "getBounds":
            return {"bottom": node.bounds.bottom, "left": node.bounds.left,
                    "right": node.bounds.right, "top": node.bounds.top}
        if name == "getBoundsCenter":
            center: Point = node.bounds.get_center()
            return {"x": center.x, "y": center.y}
        if name == "click":
            self._driver.click(*node.bounds.get_center().to_tuple())
            return None
        if name == "doubleClick":
            self._driver.double_click(*node.bounds.get_center().to_tuple())
            return None
        if name == "longClick":
            self._driver.long_click(*node.bounds.get_center().to_tuple())
            return None
        if name == "inputText":
            # click to focus the input field, then type
            self._driver.click(*node.bounds.get_center().to_tuple())
            self._driver.input_text(args[0])
            return None
        if name in ("clearText", "pinchIn", "pinchOut", "dragTo"):
            raise NotImplementedError(
                f"`{api}` is not supported for fuzzy matched elements, use exact matching instead")

        raise RuntimeError(f"unknown api for client-side matched element: {api}")

    @property
    def id(self) -> str:
        return self.__operate("Component.getId")

    @property
    def key(self) -> str:
        return self.__operate("Component.getId")

    @property
    def type(self) -> str:
        return self.__operate("Component.getType")

    @property
    def text(self) -> str:
        return self.__operate("Component.getText")

    @property
    def description(self) -> str:
        return self.__operate("Component.getDescription")

    @property
    def isSelected(self) -> bool:
        return self.__operate("Component.isSelected")

    @property
    def isChecked(self) -> bool:
        return self.__operate("Component.isChecked")

    @property
    def isEnabled(self) -> bool:
        return self.__operate("Component.isEnabled")

    @property
    def isFocused(self) -> bool:
        return self.__operate("Component.isFocused")

    @property
    def isCheckable(self) -> bool:
        return self.__operate("Component.isCheckable")

    @property
    def isClickable(self) -> bool:
        return self.__operate("Component.isClickable")

    @property
    def isLongClickable(self) -> bool:
        return self.__operate("Component.isLongClickable")

    @property
    def isScrollable(self) -> bool:
        return self.__operate("Component.isScrollable")

    @property
    def bounds(self) -> Bounds:
        _raw = self.__operate("Component.getBounds")
        _raw = {k: v for k, v in _raw.items() if k in {"bottom", "left", "right", "top"}}
        return Bounds(**_raw)

    @property
    def boundsCenter(self) -> Point:
        _raw = self.__operate("Component.getBoundsCenter")
        _raw = {k: v for k, v in _raw.items() if k in {"x", "y"}}
        return Point(**_raw)

    @property
    def info(self) -> ElementInfo:
        return ElementInfo(
            id=self.id,
            key=self.key,
            type=self.type,
            text=self.text,
            description=self.description,
            isSelected=self.isSelected,
            isChecked=self.isChecked,
            isEnabled=self.isEnabled,
            isFocused=self.isFocused,
            isCheckable=self.isCheckable,
            isClickable=self.isClickable,
            isLongClickable=self.isLongClickable,
            isScrollable=self.isScrollable,
            bounds=self.bounds,
            boundsCenter=self.boundsCenter)

    @delay
    def click(self):
        return self.__operate("Component.click")

    @delay
    def click_if_exists(self):
        try:
            return self.__operate("Component.click")
        except ElementNotFoundError:
            pass

    @delay
    def double_click(self):
        return self.__operate("Component.doubleClick")

    @delay
    def long_click(self):
        return self.__operate("Component.longClick")

    @delay
    def drag_to(self, component: ComponentData):
        return self.__operate("Component.dragTo", [component.value])

    @delay
    def input_text(self, text: str):
        return self.__operate("Component.inputText", [text])

    @delay
    def clear_text(self):
        return self.__operate("Component.clearText")

    @delay
    def pinch_in(self, scale: float = 0.5):
        return self.__operate("Component.pinchIn", [scale])

    @delay
    def pinch_out(self, scale: float = 2):
        return self.__operate("Component.pinchOut", [scale])
