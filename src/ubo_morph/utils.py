from __future__ import annotations

import numpy as np


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR array")
    return np.ascontiguousarray(array)


def as_point(value: object) -> np.ndarray:
    point = np.asarray(value, dtype=np.float32)
    if point.shape != (2,):
        raise ValueError("point must have shape (2,)")
    return point


def round_away(x: float) -> int:
    """Round half away from zero: 1.5 -> 2, -1.5 -> -2."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)
