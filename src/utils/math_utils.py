"""Common mathematical helpers for telescope camera detection."""
from __future__ import annotations

from typing import Union

from src.constants import MIN_TIME_DELTA

Number = Union[int, float]


def safe_divide(numerator: Number, denominator: Number, default: float = 0.0) -> float:
    """Safely divide, returning ``default`` if denominator is falsy or invalid."""
    try:
        if not denominator:
            return default
        return float(numerator) / float(denominator)
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def calculate_fps(frame_count: Number, time_delta: Number, default: float = 0.0) -> float:
    """Calculate frames-per-second with a minimum time threshold."""
    adjusted_delta = max(float(time_delta or 0.0), MIN_TIME_DELTA)
    return safe_divide(frame_count, adjusted_delta, default=default)
