"""Tests for shared math utility helpers."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import MIN_TIME_DELTA
from src.utils import safe_divide, calculate_fps


def test_safe_divide_zero_uses_default():
    assert safe_divide(10, 0) == pytest.approx(0.0)


def test_safe_divide_zero_with_custom_default():
    assert safe_divide(10, 0, default=1.0) == pytest.approx(1.0)


def test_safe_divide_handles_type_error():
    assert safe_divide(10, None, default=2.5) == pytest.approx(2.5)


def test_calculate_fps_zero_time_uses_min_delta():
    fps = calculate_fps(frame_count=30, time_delta=0.0)
    assert fps == pytest.approx(30 / MIN_TIME_DELTA)


def test_calculate_fps_zero_frames_returns_zero():
    assert calculate_fps(frame_count=0, time_delta=0.5) == pytest.approx(0.0)
