"""Speed / quality helpers."""
from __future__ import annotations

from typing import Dict, Mapping, TypeVar

T = TypeVar("T", int, float)


def clamp_speed_quality(value: int) -> int:
    """Clamp the UI slider (0-100)."""
    return max(0, min(100, int(value)))


def quality_weight(value: int) -> float:
    """Return [0,1] weighting representing quality preference."""
    return clamp_speed_quality(value) / 100.0


def speed_weight(value: int) -> float:
    """Return [0,1] weighting representing speed preference."""
    return 1.0 - quality_weight(value)


def speed_bucket(value: int) -> str:
    """Three-way bucket: speed / balanced / quality."""
    score = clamp_speed_quality(value)
    if score <= 33:
        return "speed"
    if score <= 66:
        return "balanced"
    return "quality"


def lerp(a: T, b: T, t: float) -> T:
    """Linear interpolation."""
    return a + (b - a) * t  # type: ignore[operator]


def interpolate_three(bands: Mapping[str, T], score: int) -> T:
    """Interpolate between speed -> balanced -> quality anchor values."""
    score = clamp_speed_quality(score)
    speed_val = bands["speed"]
    balanced_val = bands["balanced"]
    quality_val = bands["quality"]

    if score <= 50:
        t = score / 50.0
        return lerp(speed_val, balanced_val, t)

    t = (score - 50) / 50.0
    return lerp(balanced_val, quality_val, t)


def map_range(value: float, min_value: float, max_value: float) -> float:
    """Clamp float value into [min, max]."""
    return max(min_value, min(max_value, value))


def scale_with_quality(speed_value: float, quality_value: float, score: int) -> float:
    """Smoothly move between speed and quality anchors based on slider."""
    q = quality_weight(score)
    return speed_value + (quality_value - speed_value) * q


def weighted_choice(options: Dict[str, float], score: int) -> float:
    """
    Convenience helper: interpret the options dict as three anchors.

    Example::

        weighted_choice({"speed": 0.9, "balanced": 0.95, "quality": 0.99}, score)
    """
    required = {"speed", "balanced", "quality"}
    missing = required.difference(options.keys())
    if missing:
        raise KeyError(f"Missing anchors for weighted_choice: {sorted(missing)}")
    return interpolate_three(options, score)

