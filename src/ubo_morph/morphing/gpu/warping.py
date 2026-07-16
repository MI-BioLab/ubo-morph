from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def warp_image_by_triangles(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    triangles: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Numba GPU implementation hook for piecewise-affine image warping."""
    raise NotImplementedError("GPU image warping is not implemented yet.")
