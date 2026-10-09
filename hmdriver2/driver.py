# -*- coding: utf-8 -*-

import json
import uuid
import re
from typing import Type, Any, Tuple, Dict, Union, List, Optional
from functools import cached_property  # python3.8+
import os
from . import logger
from .utils import delay
from ._client import HmClient
from ._uiobject import UiObject
from .hdc import list_devices
from .exception import DeviceNotFoundError, ElementNotFoundError, HdcError
from .proto import HypiumResponse, KeyCode, Point, DisplayRotation, DeviceInfo, CommandResult
from dataclasses import dataclass
import cv2
import numpy as np



@dataclass
class Rect:
    left: int
    right: int
    top: int
    bottom: int

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


class Driver:
    _instance: Dict[str, "Driver"] = {}

    def __new__(cls: Type["Driver"], serial: Optional[str] = None) -> "Driver":
        """
        Ensure that only one instance of Driver exists per serial.
        If serial is None, use the first serial from list_devices().
        """
        serial = cls._prepare_serial(serial)

        if serial not in cls._instance:
            instance = super().__new__(cls)
            cls._instance[serial] = instance
            # Temporarily store the serial in the instance for initialization
            instance._serial_for_init = serial
        return cls._instance[serial]

    def __init__(self, serial: Optional[str] = None):
        """
        Initialize the Driver instance. Only initialize if `_initialized` is not set.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Use the serial prepared in `__new__`
        serial = getattr(self, "_serial_for_init", serial)
        if serial is None:
            raise ValueError("Serial number is required for initialization.")

        self.serial = serial
        self._client = HmClient(self.serial)
        self.hdc = self._client.hdc
        self._init_hmclient()
        self._initialized = True  # Mark the instance as initialized
        del self._serial_for_init  # Clean up temporary attribute

    @classmethod
    def _prepare_serial(cls, serial: str = None) -> str:
        """
        Prepare the serial. Use the first available device if serial is None.
        """
        devices = list_devices()
        if not devices:
            raise DeviceNotFoundError("No devices found. Please connect a device.")

        if serial is None:
            logger.info(f"No serial provided, using the first device: {devices[0]}")
            return devices[0]
        if serial not in devices:
            raise DeviceNotFoundError(f"Device [{serial}] not found")
        return serial

    def __call__(self, **kwargs) -> UiObject:

        return UiObject(self._client, self, **kwargs)

    def __del__(self):
        Driver._instance.clear()
        if hasattr(self, '_client') and self._client:
            self._client.release()

    def _init_hmclient(self):
        self._client.start()

    def _invoke(self, api: str, args: List = []) -> HypiumResponse:
        return self._client.invoke(api, this="Driver#0", args=args)

    @delay
    def start_app(self, package_name: str, page_name: Optional[str] = None):
        """
        Start an application on the device.
        If the `package_name` is empty, it will retrieve main ability using `get_app_main_ability`.

        Args:
            package_name (str): The package name of the application.
            page_name (Optional[str]): Ability Name within the application to start.
        """
        if not page_name:
            page_name = self.get_app_main_ability(package_name).get('name', 'MainAbility')
        self.hdc.start_app(package_name, page_name)

    def force_start_app(self, package_name: str, page_name: Optional[str] = None):
        self.go_home()
        self.stop_app(package_name)
        self.start_app(package_name, page_name)

    def stop_app(self, package_name: str):
        self.hdc.stop_app(package_name)

    def clear_app(self, package_name: str):
        """
        Clear the application's cache and data.
        """
        self.hdc.shell(f"bm clean -n {package_name} -c")  # clear cache
        self.hdc.shell(f"bm clean -n {package_name} -d")  # clear data

    def install_app(self, apk_path: str):
        self.hdc.install(apk_path)

    def uninstall_app(self, package_name: str):
        self.hdc.uninstall(package_name)

    def list_apps(self, include_system_apps: bool = False) -> List:
        return self.hdc.list_apps(include_system_apps)

    def app_version(self, bundle_name) -> Dict:
        return self.hdc.app_version(bundle_name)

    def has_app(self, package_name: str) -> bool:
        return self.hdc.has_app(package_name)

    def current_app(self) -> Tuple[str, str]:
        """
        Get the current foreground application information.

        Returns:
            Tuple[str, str]: A tuple contain the package_name andpage_name of the foreground application.
                             If no foreground application is found, returns (None, None).
        """

        return self.hdc.current_app()

    def get_app_info(self, package_name: str) -> Dict:
        """
        Get detailed information about a specific application.

        Args:
            package_name (str): The package name of the application to retrieve information for.

        Returns:
            Dict: A dictionary containing the application information. If an error occurs during parsing,
                  an empty dictionary is returned.
        """
        app_info = {}
        data: CommandResult = self.hdc.shell(f"bm dump -n {package_name}")
        output = data.output
        try:
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            json_output = output[json_start:json_end]

            app_info = json.loads(json_output)
        except Exception as e:
            logger.error(f"An error occurred:{e}")
        return app_info

    def get_app_abilities(self, package_name: str) -> List[Dict]:
        """
        Get the abilities of an application.

        Args:
            package_name (str): The package name of the application.

        Returns:
            List[Dict]: A list of dictionaries containing the abilities of the application.
        """
        result = []
        app_info = self.get_app_info(package_name)
        hap_module_infos = app_info.get("hapModuleInfos")
        main_entry = app_info.get("mainEntry")
        for hap_module_info in hap_module_infos:
            # 尝试读取moduleInfo
            try:
                ability_infos = hap_module_info.get("abilityInfos")
                module_main = hap_module_info["mainAbility"]
            except Exception as e:
                logger.warning(f"Fail to parse moduleInfo item, {repr(e)}")
                continue
            # 尝试读取abilityInfo
            for ability_info in ability_infos:
                try:
                    is_launcher_ability = False
                    skills = ability_info['skills']
                    if len(skills) > 0 or "action.system.home" in skills[0]["actions"]:
                        is_launcher_ability = True
                    icon_ability_info = {
                        "name": ability_info["name"],
                        "moduleName": ability_info["moduleName"],
                        "moduleMainAbility": module_main,
                        "mainModule": main_entry,
                        "isLauncherAbility": is_launcher_ability
                    }
                    result.append(icon_ability_info)
                except Exception as e:
                    logger.warning(f"Fail to parse ability_info item, {repr(e)}")
                    continue
        logger.debug(f"all abilities: {result}")
        return result

    def get_app_main_ability(self, package_name: str) -> Dict:
        """
        Get the main ability of an application.

        Args:
            package_name (str): The package name of the application to retrieve information for.

        Returns:
            Dict: A dictionary containing the main ability of the application.

        """
        if not (abilities := self.get_app_abilities(package_name)):
            return {}
        for item in abilities:
            score = 0
            if (name := item["name"]) and name == item["moduleMainAbility"]:
                score += 1
            if (module_name := item["moduleName"]) and module_name == item["mainModule"]:
                score += 1
            item["score"] = score
        abilities.sort(key=lambda x: (not x["isLauncherAbility"], -x["score"]))
        logger.debug(f"main ability: {abilities[0]}")
        return abilities[0]

    @cached_property
    def toast_watcher(self):

        obj = self

        class _Watcher:
            def start(self) -> bool:
                api = "Driver.uiEventObserverOnce"
                resp: HypiumResponse = obj._invoke(api, args=["toastShow"])
                return resp.result

            def get_toast(self, timeout: int = 3) -> str:
                api = "Driver.getRecentUiEvent"
                resp: HypiumResponse = obj._invoke(api, args=[timeout])
                if resp.result:
                    return resp.result.get("text")
                return None

        return _Watcher()

    @delay
    def go_back(self):
        self.hdc.send_key(KeyCode.BACK)

    @delay
    def go_home(self):
        self.hdc.send_key(KeyCode.HOME)

    @delay
    def press_key(self, key_code: Union[KeyCode, int]):
        self.hdc.send_key(key_code)

    def screen_on(self):
        self.hdc.wakeup()

    def screen_off(self):
        self.hdc.wakeup()
        self.press_key(KeyCode.POWER)

    @delay
    def unlock(self):
        self.screen_on()
        w, h = self.display_size
        self.swipe(0.5 * w, 0.8 * h, 0.5 * w, 0.2 * h, speed=6000)

    @cached_property
    def display_size(self) -> Tuple[int, int]:
        api = "Driver.getDisplaySize"
        resp: HypiumResponse = self._invoke(api)
        w, h = resp.result.get("x"), resp.result.get("y")
        return w, h

    @cached_property
    def display_rotation(self) -> DisplayRotation:
        api = "Driver.getDisplayRotation"
        value = self._invoke(api).result
        return DisplayRotation.from_value(value)

    def set_display_rotation(self, rotation: DisplayRotation):
        """
        Sets the display rotation to the specified orientation.

        Args:
            rotation (DisplayRotation): display rotation.
        """
        api = "Driver.setDisplayRotation"
        self._invoke(api, args=[rotation.value])

    @cached_property
    def device_info(self) -> DeviceInfo:
        """
        Get detailed information about the device.

        Returns:
            DeviceInfo: An object containing various properties of the device.
        """
        hdc = self.hdc
        return DeviceInfo(
            productName=hdc.product_name(),
            model=hdc.model(),
            sdkVersion=hdc.sdk_version(),
            sysVersion=hdc.sys_version(),
            cpuAbi=hdc.cpu_abi(),
            wlanIp=hdc.wlan_ip(),
            displaySize=self.display_size,
            displayRotation=self.display_rotation
        )

    @delay
    def open_url(self, url: str, system_browser: bool = True):
        if system_browser:
            # Use the system browser
            self.hdc.shell(f"aa start -A ohos.want.action.viewData -e entity.system.browsable -U {url}")
        else:
            # Default method
            self.hdc.shell(f"aa start -U {url}")

    def pull_file(self, rpath: str, lpath: str):
        """
        Pull a file from the device to the local machine.

        Args:
            rpath (str): The remote path of the file on the device.
            lpath (str): The local path where the file should be saved.
        """
        self.hdc.recv_file(rpath, lpath)

    def push_file(self, lpath: str, rpath: str):
        """
        Push a file from the local machine to the device.

        Args:
            lpath (str): The local path of the file.
            rpath (str): The remote path where the file should be saved on the device.
        """
        self.hdc.send_file(lpath, rpath)

    def screenshot(self, path: str, method: str = "snapshot_display") -> str:
        """
        Take a screenshot of the device display.

        Args:
            path (str): The local path to save the screenshot.
            method (str): The screenshot method to use. Options are:
                          - "snapshot_display" (default, recommended for better performance)
                          - "screenCap" (alternative method, higher quality but slower).

        Returns:
            str: The path where the screenshot is saved.
        """
        return self.hdc.screenshot(path, method=method)

    def shell(self, cmd) -> CommandResult:
        return self.hdc.shell(cmd)

    def _to_abs_pos(self, x: Union[int, float], y: Union[int, float]) -> Point:
        """
        Convert percentages to absolute screen coordinates.

        Args:
            x (Union[int, float]): X coordinate as a percentage or absolute value.
            y (Union[int, float]): Y coordinate as a percentage or absolute value.

        Returns:
            Point: A Point object with absolute screen coordinates.
        """
        assert x >= 0
        assert y >= 0

        w, h = self.display_size

        if x < 1:
            x = int(w * x)
        if y < 1:
            y = int(h * y)
        return Point(int(x), int(y))

    @delay
    def click(self, x: Union[int, float], y: Union[int, float]):

        # self.hdc.tap(point.x, point.y)
        point = self._to_abs_pos(x, y)
        api = "Driver.click"
        self._invoke(api, args=[point.x, point.y])

    @delay
    def double_click(self, x: Union[int, float], y: Union[int, float]):
        point = self._to_abs_pos(x, y)
        api = "Driver.doubleClick"
        self._invoke(api, args=[point.x, point.y])

    @delay
    def long_click(self, x: Union[int, float], y: Union[int, float]):
        point = self._to_abs_pos(x, y)
        api = "Driver.longClick"
        self._invoke(api, args=[point.x, point.y])

    @delay
    def swipe(self, x1, y1, x2, y2, speed=2000):
        """
        Perform a swipe action on the device screen.

        Args:
            x1 (float): The start X coordinate as a percentage or absolute value.
            y1 (float): The start Y coordinate as a percentage or absolute value.
            x2 (float): The end X coordinate as a percentage or absolute value.
            y2 (float): The end Y coordinate as a percentage or absolute value.
            speed (int, optional): The swipe speed in pixels per second. Default is 2000. Range: 200-40000,
            If not within the range, set to default value of 2000.
        """

        point1 = self._to_abs_pos(x1, y1)
        point2 = self._to_abs_pos(x2, y2)

        if speed < 200 or speed > 40000:
            logger.warning("`speed` is not in the range[200-40000], Set to default value of 2000.")
            speed = 2000

        api = "Driver.swipe"
        self._invoke(api, args=[point1.x, point1.y, point2.x, point2.y, speed])

    @cached_property
    def swipe_ext(self):
        """
        d.swipe_ext("up")
        d.swipe_ext("up", box=(0.2, 0.2, 0.8, 0.8))
        """
        from ._swipe import SwipeExt
        return SwipeExt(self)

    @delay
    def input_text(self, text: str):
        """
        Inputs text into the currently focused input field.

        Note: The input field must have focus before calling this method.

        Args:
            text (str): input value
        """
        return self._invoke("Driver.inputText", args=[{"x": 1, "y": 1}, text])

    def dump_hierarchy(self) -> Dict:
        """
        Dump the UI hierarchy of the device screen.

        Returns:
            Dict: The dumped UI hierarchy as a dictionary.
        """
        # return self._client.invoke_captures("captureLayout").result
        return self.hdc.dump_hierarchy()

    @cached_property
    def gesture(self):
        from ._gesture import _Gesture
        return _Gesture(self)

    @cached_property
    def screenrecord(self):
        from ._screenrecord import RecordClient
        return RecordClient(self.serial, self)

    def _invalidate_cache(self, attribute_name):
        """
        Invalidate the cached property.

        Args:
            attribute_name (str): The name of the attribute to invalidate.
        """
        if attribute_name in self.__dict__:
            del self.__dict__[attribute_name]

    @cached_property
    def xpath(self):
        """
        d.xpath("//*[@text='Hello']").click()
        """
        from ._xpath import _XPath
        return _XPath(self)

    @cached_property
    def image(self):
        """
        Image matching & comparison.
        d.image.find("btn.png")        # template matching on current screen
        d.image.compare("a.png", "b.png")  # compare two local images
        """
        from ._image import _Image
        return _Image(self)

    def capture_screen(self, save_path: str, in_pc: bool = True,
                       area: Union[Rect, None] = None) -> str:
        """
        @func 直接在PC端生成截图（绕开设备端文件拉取），支持全屏/指定Rect区域截图
        @param save_path: 截图保存路径
        @param in_pc: 保存到PC端（True，固定为True，绕开设备端逻辑）
        @param area: 指定区域截图（仅支持Rect/None）
        @return: 截图保存路径
        """
        # 1. 参数校验
        if not save_path:
            raise ValueError("保存路径save_path不能为空")
        # 强制in_pc=True，避免设备端交互（根源解决拉取失败）
        in_pc = True

        # 2. 生成PC端全屏临时截图路径
        pc_full_snap = os.path.abspath(f"./hmdriver_full_snap_{uuid.uuid4()}.jpeg")

        try:
            # 3. 直接生成PC端全屏截图（核心：绕开设备端文件）
            # 调用已封装的screenshot方法，直接保存到PC，无需设备端中转
            self.screenshot(pc_full_snap, method="snapshot_display")
            if not os.path.exists(pc_full_snap):
                # 降级：尝试screenCap方法（若snapshot_display失败）
                logger.warning("snapshot_display生成PC截图失败，尝试screenCap")
                # 若self.screenshot不支持screenCap，直接调用hdc screenCap并拉取（最后尝试）
                device_temp = f"/data/local/tmp/last_try_snap_{uuid.uuid4()}.jpeg"
                self.hdc.shell(f"screenCap -f {device_temp}")
                # 最后一次尝试拉取（若仍失败则抛明确错误）
                try:
                    self.pull_file(device_temp, pc_full_snap)
                    self.hdc.shell(f"rm -f {device_temp}")
                except Exception as e:
                    raise RuntimeError(
                        f"所有截图方式均失败！\n"
                        f"1. self.screenshot直接生成PC截图失败\n"
                        f"2. screenCap+拉取失败：{str(e)}\n"
                        f"请检查：\n"
                        f"- HDC连接是否正常（hdc devices）\n"
                        f"- 设备是否解锁/开发者模式已开启\n"
                        f"- 设备是否授予HDC文件访问权限"
                    )

            # 4. 处理截图区域：全屏/裁剪（仅保留Rect逻辑）
            capture_rect = area  # 直接使用传入的Rect，无需解析控件
            if capture_rect is not None and not isinstance(capture_rect, Rect):
                raise TypeError(f"area仅支持Rect/None类型，当前传入类型：{type(capture_rect)}")

            # 5. 生成最终截图（全屏/裁剪）
            if capture_rect is None:
                # 全屏截图：直接重命名
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(pc_full_snap, save_path)
            else:
                # 指定Rect区域裁剪
                full_img = cv2.imread(pc_full_snap)
                if full_img is None:
                    raise RuntimeError(f"读取全屏截图{pc_full_snap}失败，文件损坏或格式不支持")
                # 校验裁剪区域
                img_h, img_w = full_img.shape[:2]
                if (capture_rect.left < 0 or capture_rect.top < 0 or
                        capture_rect.right > img_w or capture_rect.bottom > img_h):
                    raise RuntimeError(
                        f"裁剪区域越界！图片尺寸({img_w}x{img_h})，裁剪区域({capture_rect.left},{capture_rect.top})-({capture_rect.right},{capture_rect.bottom})"
                    )
                # 裁剪并保存
                crop_img = full_img[capture_rect.top:capture_rect.bottom, capture_rect.left:capture_rect.right]
                cv2.imwrite(save_path, crop_img)
                # 删除全屏临时文件
                os.remove(pc_full_snap)

            logger.info(f"截图成功保存到PC端：{save_path}")
            return save_path

        except Exception as e:
            logger.error(f"截图失败：{str(e)}")
            # 清理临时文件
            if os.path.exists(pc_full_snap):
                os.remove(pc_full_snap)
            raise
    

