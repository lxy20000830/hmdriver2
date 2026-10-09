# -*- coding: utf-8 -*-
"""
Offline unit tests for client-side fuzzy matching (textContains/textMatches/...).
No real device required: a fake driver serves a synthetic hierarchy dump.
"""

import pytest

import re

from hmdriver2._uiobject import UiObject, ByType, FUZZY_KEYS
from hmdriver2.exception import ElementNotFoundError
from hmdriver2.proto import Bounds, Point


def _node(text="", id="", key="", type="Text", description="", bounds="[0,0][1,1]",
          clickable="false", enabled="true", selected="false", checked="false",
          focused="false", checkable="false", scrollable="false", longClickable="false",
          children=None):
    return {
        "attributes": {
            "text": text, "id": id, "key": key, "type": type,
            "description": description, "bounds": bounds,
            "clickable": clickable, "enabled": enabled, "selected": selected,
            "checked": checked, "focused": focused, "checkable": checkable,
            "scrollable": scrollable, "longClickable": longClickable,
        },
        "children": children or [],
    }


HIERARCHY = _node(
    type="root", bounds="[0,0][1260,2720]", children=[
        _node(text="showToast", id="btn_toast", key="btn_toast", type="Button",
              description="show toast", bounds="[100,200][300,280]", clickable="true"),
        _node(text="精选推荐", id="txt_1", key="txt_1", type="Text",
              bounds="[0,300][200,350]", clickable="true"),
        _node(text="精选视频", id="txt_2", key="txt_2", type="Text",
              bounds="[0,400][200,450]"),
        _node(text="视频精选", id="txt_3", key="txt_3", type="Text",
              bounds="[0,500][200,550]"),
        _node(text="btn_disabled", id="btn_2", key="btn_2", type="Button",
              bounds="[0,600][300,680]"),
    ])


class FakeDriver:
    def __init__(self, hierarchy=None):
        self._hierarchy = hierarchy if hierarchy is not None else HIERARCHY
        self.clicks = []
        self.double_clicks = []
        self.long_clicks = []
        self.inputs = []

    def dump_hierarchy(self):
        return self._hierarchy

    def click(self, x, y):
        self.clicks.append((x, y))

    def double_click(self, x, y):
        self.double_clicks.append((x, y))

    def long_click(self, x, y):
        self.long_clicks.append((x, y))

    def input_text(self, text):
        self.inputs.append(text)


@pytest.fixture
def d():
    return FakeDriver()


def ui(driver, **kwargs):
    return UiObject(None, driver, **kwargs)


# ------------------------------------------------------------------
# ByType / verification
# ------------------------------------------------------------------
def test_by_type_new_fuzzy_keys():
    for k in ["textContains", "textStartsWith", "textEndsWith", "textMatches",
              "descriptionContains", "idContains", "keyStartsWith",
              "typeEndsWith", "typeMatches"]:
        assert ByType.verify(k)
    assert set(FUZZY_KEYS) <= {item.value for item in ByType}


def test_invalid_key_rejected(d):
    with pytest.raises(ReferenceError):
        ui(d, fooContains="x")


def test_is_before_after_with_fuzzy_rejected(d):
    with pytest.raises(ValueError):
        ui(d, textContains="xx", isBefore=True)
    with pytest.raises(ValueError):
        ui(d, textMatches="xx", isAfter=True)


def test_fuzzy_without_driver_raises():
    obj = UiObject(None, None, textContains="xx")
    with pytest.raises(RuntimeError):
        obj.exists()


# ------------------------------------------------------------------
# text fuzzy matching
# ------------------------------------------------------------------
def test_text_contains(d):
    assert ui(d, textContains="精选").count == 3  # 精选推荐 / 精选视频 / 视频精选
    assert ui(d, textContains="不存在").count == 0


def test_text_starts_with(d):
    assert ui(d, textStartsWith="精选").count == 2


def test_text_ends_with(d):
    assert ui(d, textEndsWith="精选").count == 1
    assert ui(d, textEndsWith="精选").text == "视频精选"


def test_text_matches_regex(d):
    assert ui(d, textMatches=r"^精选.*$").count == 2
    assert ui(d, textMatches=r"^showToast$").count == 1
    assert ui(d, textMatches=r"^视频").text == "视频精选"


def test_text_matches_invalid_regex(d):
    with pytest.raises(re.error):
        ui(d, textMatches=r"(").exists()


# ------------------------------------------------------------------
# id/key/type/description fuzzy matching
# ------------------------------------------------------------------
def test_id_fuzzy(d):
    assert ui(d, idContains="btn").count == 2
    assert ui(d, idStartsWith="txt").count == 3
    assert ui(d, idEndsWith="_1").count == 1
    assert ui(d, idMatches=r"^txt_\d$").count == 3


def test_key_fuzzy(d):
    assert ui(d, keyContains="btn").count == 2


def test_type_fuzzy(d):
    assert ui(d, typeContains="But").count == 2
    assert ui(d, typeMatches=r"^But.*n$").count == 2


def test_description_fuzzy(d):
    assert ui(d, descriptionContains="toast").count == 1
    assert ui(d, descriptionMatches=r"^show.*$").text == "showToast"


# ------------------------------------------------------------------
# combined conditions & bool normalization
# ------------------------------------------------------------------
def test_combine_fuzzy_and_exact(d):
    assert ui(d, textContains="精选", type="Text").count == 3
    assert ui(d, textContains="精选", type="Button").count == 0


def test_combine_fuzzy_and_bool(d):
    # bool accepts both python bool and hierarchy "true" string
    assert ui(d, textContains="精选", clickable=True).count == 1
    assert ui(d, textContains="精选", clickable=False).count == 2


def test_index(d):
    obj = ui(d, textContains="精选", index=2)
    assert obj.text == "视频精选"
    assert ui(d, textContains="精选", index=99).exists() is False


def test_empty_hierarchy():
    d = FakeDriver(hierarchy={})
    assert ui(d, textContains="xx").count == 0
    assert ui(d, textContains="xx").exists() is False


# ------------------------------------------------------------------
# properties reading (client-side matched node)
# ------------------------------------------------------------------
def test_properties(d):
    obj = ui(d, textContains="show")
    assert obj.text == "showToast"
    assert obj.type == "Button"
    assert obj.id == "btn_toast"
    assert obj.key == "btn_toast"
    assert obj.description == "show toast"
    assert obj.isClickable is True
    assert obj.isEnabled is True
    assert obj.isSelected is False
    assert obj.bounds == Bounds(left=100, top=200, right=300, bottom=280)
    assert obj.boundsCenter == Point(x=200, y=240)


def test_info(d):
    info = ui(d, textContains="show").info
    assert info.text == "showToast"
    assert info.type == "Button"
    assert info.isClickable is True
    assert info.bounds.left == 100
    assert info.boundsCenter.x == 200


# ------------------------------------------------------------------
# operations (coordinate based)
# ------------------------------------------------------------------
def test_click(d):
    ui(d, textContains="show").click()
    assert d.clicks == [(200, 240)]


def test_double_click(d):
    ui(d, textContains="show").double_click()
    assert d.double_clicks == [(200, 240)]


def test_long_click(d):
    ui(d, textContains="show").long_click()
    assert d.long_clicks == [(200, 240)]


def test_click_if_exists(d):
    ui(d, textContains="不存在xx").click_if_exists()  # no raise
    assert d.clicks == []


def test_click_not_found_raises(d):
    with pytest.raises(ElementNotFoundError):
        ui(d, textContains="不存在xx").click()


def test_input_text(d):
    ui(d, textContains="show").input_text("abc")
    assert d.clicks == [(200, 240)]  # click to focus first
    assert d.inputs == ["abc"]


def test_unsupported_operations(d):
    from hmdriver2.proto import ComponentData
    with pytest.raises(NotImplementedError):
        ui(d, textContains="show").clear_text()
    with pytest.raises(NotImplementedError):
        ui(d, textContains="show").pinch_in()
    with pytest.raises(NotImplementedError):
        ui(d, textContains="show").pinch_out()
    with pytest.raises(NotImplementedError):
        ui(d, textContains="show").drag_to(ComponentData("Component#0"))
