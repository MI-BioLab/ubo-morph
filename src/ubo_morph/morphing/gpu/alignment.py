from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from ubo_morph.landmarks import Landmarks


def align_face_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
) -> tuple[np.ndarray, np.ndarray, Landmarks, Landmarks]:
    """Numba GPU implementation hook for face alignment."""
    raise NotImplementedError("GPU face alignment is not implemented yet.")
