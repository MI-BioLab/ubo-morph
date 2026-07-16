from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    blending_factor: float,
) -> np.ndarray:
    """Numba GPU implementation hook for weighted image blending."""
    raise NotImplementedError("GPU image blending is not implemented yet.")
