# -*- coding: utf-8 -*-
"""
Offline unit tests for d.image (template matching + image comparison).
No real device required: a fake driver serves synthetic screenshots.
"""

import cv2
import numpy as np
import pytest

from hmdriver2._image import _Image
from hmdriver2.driver import Rect
from hmdriver2.exception import ImageCompareError, ImageNotFoundError
from hmdriver2.proto import Bounds


def _gradient(w=200, h=200):
    """Synthetic screen: horizontal gradient background."""
    img = np.tile(np.linspace(30, 225, w, dtype=np.uint8), (h, 1))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _pattern(seed=0):
    """Synthetic 30x40 template pattern."""
    rng = np.random.RandomState(seed)
    gray = rng.randint(0, 256, (40, 30), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _screen_with_patterns(positions, size=(200, 200)):
    """Gradient screen with the same pattern pasted at each (x, y) position."""
    screen = _gradient(*size)
    pattern = _pattern()
    ph, pw = pattern.shape[:2]
    for (x, y) in positions:
        screen[y:y + ph, x:x + pw] = pattern
    return screen


class FakeDriver:
    def __init__(self, screen):
        self._screen = screen
        self.clicks = []

    def screenshot(self, path, method="snapshot_display"):
        cv2.imwrite(path, self._screen)
        return path

    def click(self, x, y):
        self.clicks.append((x, y))


PATTERN_SIZE = (30, 40)  # (w, h)


@pytest.fixture
def paths(tmp_path):
    """Create screen (one pattern at (60,70)) + template files."""
    screen = _screen_with_patterns([(60, 70)])
    screen_path = str(tmp_path / "screen.png")
    cv2.imwrite(screen_path, screen)

    template_path = str(tmp_path / "template.png")
    cv2.imwrite(template_path, _pattern())

    return screen_path, template_path


@pytest.fixture
def d(paths):
    screen_path, _ = paths
    screen = cv2.imread(screen_path)
    return FakeDriver(screen)


@pytest.fixture
def image(d):
    return _Image(d)


# ------------------------------------------------------------------
# template matching: find / exists / click
# ------------------------------------------------------------------
def test_find_hit(image, paths):
    _, template_path = paths
    result = image.find(template_path)
    assert result is not None
    assert result.similarity > 0.99
    assert result.rect == Bounds(left=60, top=70, right=90, bottom=110)
    assert result.center.x == 75
    assert result.center.y == 90


def test_find_miss(image, paths, tmp_path):
    # a random noise template is not on the gradient screen
    assert image.find(_write_noise(tmp_path), threshold=0.95) is None


def _write_noise(tmp_path, name="noise.png"):
    """Write a random noise template that is NOT on the gradient screen."""
    noise_path = str(tmp_path / name)
    rng = np.random.RandomState(123)
    noise = rng.randint(0, 256, (40, 30), dtype=np.uint8)
    cv2.imwrite(noise_path, cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR))
    return noise_path


def test_exists(image, paths, tmp_path):
    _, template_path = paths
    assert image.exists(template_path) is True
    assert image.exists(_write_noise(tmp_path), threshold=0.95) is False
    # a missing template file is a script error, not a "not found"
    with pytest.raises(ImageCompareError):
        image.exists(str(tmp_path / "not_exist_file.png"))


def test_click(image, paths):
    _, template_path = paths
    image.click(template_path)
    assert image._d.clicks == [(75, 90)]


def test_click_not_found_raises(image, paths, tmp_path):
    with pytest.raises(ImageNotFoundError):
        image.click(_write_noise(tmp_path), threshold=0.95)


def test_click_if_exists(image, paths, tmp_path):
    _, template_path = paths
    image.click_if_exists(template_path)  # no raise, clicks
    assert image._d.clicks == [(75, 90)]
    image.click_if_exists(_write_noise(tmp_path, "noise2.png"), threshold=0.95)  # no raise, no click
    assert image._d.clicks == [(75, 90)]


def test_find_all_and_dedup(image, paths, tmp_path):
    # screen with 3 identical patterns
    screen = _screen_with_patterns([(10, 20), (100, 20), (10, 120)])
    screen_path = str(tmp_path / "screen3.png")
    cv2.imwrite(screen_path, screen)
    image._d._screen = screen

    _, template_path = paths
    results = image.find_all(template_path, threshold=0.9, max_results=5)
    assert len(results) == 3  # dedup removes overlapped candidates
    centers = sorted((r.center.x, r.center.y) for r in results)
    assert centers == [(25, 40), (25, 140), (115, 40)]


def test_find_with_region(image, paths, tmp_path):
    # two patterns: A(10,20) inside region, B(120,130) outside
    screen = _screen_with_patterns([(10, 20), (120, 130)])
    image._d._screen = screen
    _, template_path = paths

    # Rect region containing only A (note: Rect field order is left, right, top, bottom)
    results = image.find_all(template_path, region=Rect(left=0, right=100, top=0, bottom=100), threshold=0.9)
    assert len(results) == 1
    assert (results[0].center.x, results[0].center.y) == (25, 40)

    # tuple region (left, top, right, bottom) containing only B: verify region offset
    results = image.find_all(template_path, region=(100, 100, 200, 200), threshold=0.9)
    assert len(results) == 1
    assert (results[0].center.x, results[0].center.y) == (135, 150)


def test_template_larger_than_screen_raises(image, paths, tmp_path):
    big_path = str(tmp_path / "big.png")
    cv2.imwrite(big_path, _gradient(300, 300))
    with pytest.raises(ImageCompareError):
        image.find(big_path)


def test_invalid_threshold_raises(image, paths):
    _, template_path = paths
    with pytest.raises(ValueError):
        image.find(template_path, threshold=1.5)


# ------------------------------------------------------------------
# image comparison
# ------------------------------------------------------------------
def test_compare_same_image(image, paths):
    _, template_path = paths
    assert image.compare(template_path, template_path, method="ssim") > 0.99
    assert image.compare(template_path, template_path, method="histogram") > 0.99
    assert image.compare(template_path, template_path, method="mse") == pytest.approx(1.0)


def test_compare_different_image(image, tmp_path):
    w, h = 200, 200
    # A: left black / right white;  B: left white / right black
    a = np.zeros((h, w), dtype=np.uint8)
    a[:, w // 2:] = 255
    b = 255 - a

    a_path = str(tmp_path / "a.png")
    b_path = str(tmp_path / "b.png")
    cv2.imwrite(a_path, cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(b_path, cv2.cvtColor(b, cv2.COLOR_GRAY2BGR))

    # note: histogram is insensitive to position/color reversal, so only
    # ssim/mse are asserted to be low here
    assert image.compare(a_path, b_path, method="ssim") < 0.5
    assert image.compare(a_path, b_path, method="mse") < 0.5


def test_compare_resize_different_size(image, tmp_path):
    img = _gradient(200, 200)
    a_path = str(tmp_path / "a.png")
    b_path = str(tmp_path / "b.png")
    cv2.imwrite(a_path, img)
    cv2.imwrite(b_path, cv2.resize(img, (100, 100)))  # different size, same content

    score = image.compare(a_path, b_path, method="ssim")
    assert 0.0 <= score <= 1.0
    assert score > 0.7


def test_compare_invalid_method(image, paths):
    _, template_path = paths
    with pytest.raises(ValueError):
        image.compare(template_path, template_path, method="unknown")


def test_compare_file_not_found(image):
    with pytest.raises(ImageCompareError):
        image.compare("not_exist_a.png", "not_exist_b.png")


def test_assert_same(image, paths):
    _, template_path = paths
    score = image.assert_same(template_path, template_path, threshold=0.99)
    assert score >= 0.99


def test_assert_same_failed(image, tmp_path):
    w, h = 200, 200
    a = np.zeros((h, w), dtype=np.uint8)
    a[:, w // 2:] = 255
    b = 255 - a
    a_path = str(tmp_path / "a.png")
    b_path = str(tmp_path / "b.png")
    cv2.imwrite(a_path, cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(b_path, cv2.cvtColor(b, cv2.COLOR_GRAY2BGR))

    with pytest.raises(ImageCompareError):
        image.assert_same(a_path, b_path, threshold=0.5, method="ssim")
