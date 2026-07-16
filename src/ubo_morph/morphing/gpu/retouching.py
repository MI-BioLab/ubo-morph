from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def equalize_face(
    reference_image: np.ndarray,
    image_to_equalize: np.ndarray,
    reference_points: np.ndarray,
    points_to_equalize: np.ndarray,
    *,
    method: str = "color",
) -> np.ndarray:
    """Numba GPU implementation hook for face histogram matching."""
    raise NotImplementedError("GPU face equalization is not implemented yet.")


def substitute_background(
    image: np.ndarray,
    reference_points: np.ndarray,
    background_image: np.ndarray,
    *,
    blend: bool,
    eye_distance: float,
) -> np.ndarray:
    """Numba GPU implementation hook for face/background compositing."""
    raise NotImplementedError("GPU background substitution is not implemented yet.")
