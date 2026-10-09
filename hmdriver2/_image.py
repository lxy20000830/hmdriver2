# -*- coding: utf-8 -*-

"""
Image matching (template matching on device screen)
and image comparison (similarity between two local images).

Usage:
    d.image.find("btn.png", threshold=0.8)      # -> MatchResult or None
    d.image.exists("btn.png")                  # -> bool
    d.image.click("btn.png")                   # click by image
    d.image.compare("a.png", "b.png")          # -> similarity score (0~1)
    d.image.assert_same("a.png", "b.png", threshold=0.9)
"""

import os
import tempfile
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from . import logger
from .driver import Driver, Rect
from .exception import ImageCompareError, ImageNotFoundError
from .proto import Bounds, MatchResult, Point


def _read_image(path: str) -> np.ndarray:
    """
    Read an image file (supports non-ascii paths on Windows).
    """
    if not os.path.exists(path):
        raise ImageCompareError(f"image file not found: {path}")

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageCompareError(f"failed to read image (corrupted or unsupported format): {path}")
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


class _Image:
    def __init__(self, d: Driver):
        self._d = d

    # ------------------------------------------------------------------
    # template matching (find an image on the current screen)
    # ------------------------------------------------------------------
    def find(self, template: str,
             region: Union[Rect, Tuple[int, int, int, int], None] = None,
             threshold: float = 0.8,
             grayscale: bool = True) -> Optional[MatchResult]:
        """
        Find `template` image on the current device screen.

        Args:
            template: path of the template image (local file).
            region: search region, Rect or (left, top, right, bottom). None = full screen.
            threshold: similarity threshold in [0, 1], default 0.8.
            grayscale: match on grayscale images (more robust to color noise), default True.

        Returns:
            MatchResult(similarity, rect, center) if found, else None.
        """
        results = self.__find_template_all(template, region, threshold, grayscale, max_results=1)
        return results[0] if results else None

    def find_all(self, template: str,
                 region: Union[Rect, Tuple[int, int, int, int], None] = None,
                 threshold: float = 0.8,
                 grayscale: bool = True,
                 max_results: int = 5) -> List[MatchResult]:
        """
        Find all occurrences of `template` image on the current screen.
        Overlapped matches are removed (keep the highest similarity one).

        Returns:
            List[MatchResult], sorted by similarity desc. Empty list if none found.
        """
        return self.__find_template_all(template, region, threshold, grayscale, max_results)

    def exists(self, template: str,
               region: Union[Rect, Tuple[int, int, int, int], None] = None,
               threshold: float = 0.8,
               grayscale: bool = True) -> bool:
        """
        Whether `template` image exists on the current screen.
        """
        return self.find(template, region=region, threshold=threshold, grayscale=grayscale) is not None

    def click(self, template: str,
              region: Union[Rect, Tuple[int, int, int, int], None] = None,
              threshold: float = 0.8,
              grayscale: bool = True) -> MatchResult:
        """
        Find `template` image on the screen and click its center.

        Raises:
            ImageNotFoundError: if the template is not found.
        """
        result = self.find(template, region=region, threshold=threshold, grayscale=grayscale)
        if not result:
            raise ImageNotFoundError(
                f"image [{template}] not found on screen (threshold={threshold})")
        self._d.click(result.center.x, result.center.y)
        return result

    def click_if_exists(self, template: str,
                        region: Union[Rect, Tuple[int, int, int, int], None] = None,
                        threshold: float = 0.8,
                        grayscale: bool = True):
        """
        Click the `template` image center if it exists, otherwise skip silently.
        """
        try:
            return self.click(template, region=region, threshold=threshold, grayscale=grayscale)
        except ImageNotFoundError:
            return None

    def __find_template_all(self, template: str,
                            region, threshold: float, grayscale: bool,
                            max_results: int) -> List[MatchResult]:
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        tpl = _read_image(template)

        # take a screenshot of the current screen into a temp file
        fd, screen_path = tempfile.mkstemp(suffix=".jpeg", prefix="hmdriver_snap_")
        os.close(fd)
        try:
            self._d.screenshot(screen_path, method="snapshot_display")
            screen = _read_image(screen_path)
        finally:
            if os.path.exists(screen_path):
                os.remove(screen_path)

        # crop search region
        offset_x, offset_y = 0, 0
        if region is not None:
            if isinstance(region, Rect):
                left, top, right, bottom = region.left, region.top, region.right, region.bottom
            else:
                left, top, right, bottom = region
            offset_x, offset_y = left, top
            screen = screen[top:bottom, left:right]

        if grayscale:
            screen, tpl = _to_gray(screen), _to_gray(tpl)

        th, tw = tpl.shape[:2]
        sh, sw = screen.shape[:2]
        if tw > sw or th > sh:
            raise ImageCompareError(
                f"template({tw}x{th}) is larger than search region({sw}x{sh})")

        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res)  # avoid NaN on zero-variance areas

        ys, xs = np.where(res >= threshold)
        if len(xs) == 0:
            return []

        # sort by similarity desc
        candidates = sorted(zip(res[ys, xs], xs, ys), key=lambda c: -c[0])

        results: List[MatchResult] = []
        for score, x, y in candidates:
            rect = Bounds(left=int(x) + offset_x, top=int(y) + offset_y,
                          right=int(x) + tw + offset_x, bottom=int(y) + th + offset_y)
            # remove overlapped matches (keep the best one)
            if any(self.__overlap_ratio(rect, r.rect) > 0.5 for r in results):
                continue
            results.append(MatchResult(
                similarity=float(score),
                rect=rect,
                center=rect.get_center()))
            if len(results) >= max_results:
                break

        logger.debug(f"image find [{template}]: {len(results)} match(es)")
        return results

    @staticmethod
    def __overlap_ratio(a: Bounds, b: Bounds) -> float:
        """intersection area ratio relative to `a`"""
        inter_left, inter_top = max(a.left, b.left), max(a.top, b.top)
        inter_right, inter_bottom = min(a.right, b.right), min(a.bottom, b.bottom)
        inter_w, inter_h = inter_right - inter_left, inter_bottom - inter_top
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        inter_area = inter_w * inter_h
        a_area = (a.right - a.left) * (a.bottom - a.top)
        return inter_area / a_area if a_area else 0.0

    # ------------------------------------------------------------------
    # image comparison (similarity between two local images)
    # ------------------------------------------------------------------
    def compare(self, img1: str, img2: str, method: str = "ssim") -> float:
        """
        Compare two local images and return a similarity score in [0, 1].
        (1.0 means identical)

        Args:
            img1, img2: local image paths.
            method: "ssim" (structural similarity, default, best for UI regression),
                    "histogram" (color distribution, robust to position shift),
                    "mse" (normalized mean squared error).

        Notes:
            If the two images have different sizes, `img2` is resized to match `img1`.
        """
        a = _read_image(img1)
        b = _read_image(img2)

        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))

        if method == "ssim":
            return self.__ssim(a, b)
        if method == "histogram":
            return self.__histogram_score(a, b)
        if method == "mse":
            return self.__mse_score(a, b)
        raise ValueError(f"invalid method: {method}, use 'ssim' | 'histogram' | 'mse'")

    def assert_same(self, img1: str, img2: str, threshold: float = 0.9, method: str = "ssim") -> float:
        """
        Assert that two images are similar enough.

        Raises:
            ImageCompareError: if similarity < threshold.
        """
        score = self.compare(img1, img2, method=method)
        if score < threshold:
            raise ImageCompareError(
                f"image compare failed: score={score:.4f} < threshold={threshold} (method={method})")
        return score

    @staticmethod
    def __ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Structural Similarity (grayscale, gaussian window)."""
        a = _to_gray(a).astype(np.float64)
        b = _to_gray(b).astype(np.float64)

        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

        sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
        sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
        sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

        ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / \
                   ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))

        # ssim range is [-1, 1], clamp to [0, 1]
        return float(np.clip(ssim_map.mean(), 0.0, 1.0))

    @staticmethod
    def __histogram_score(a: np.ndarray, b: np.ndarray) -> float:
        """Histogram correlation (BGR joint histogram)."""
        hist_a = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_b = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        # HISTCMP_CORREL range is [-1, 1]
        return float(np.clip(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL), 0.0, 1.0))

    @staticmethod
    def __mse_score(a: np.ndarray, b: np.ndarray) -> float:
        """1 - normalized MSE, 1.0 means identical."""
        diff = a.astype(np.float64) - b.astype(np.float64)
        mse = float(np.mean(diff * diff))
        return float(np.clip(1.0 - mse / (255.0 ** 2), 0.0, 1.0))
