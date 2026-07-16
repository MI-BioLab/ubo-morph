from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def delaunay_triangles(
    points: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Numba GPU implementation hook for Delaunay triangulation."""
    raise NotImplementedError("GPU Delaunay triangulation is not implemented yet.")
