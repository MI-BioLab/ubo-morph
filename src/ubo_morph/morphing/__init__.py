from __future__ import annotations

from typing import TYPE_CHECKING

from ubo_morph.morphing._backends import _impl
from ubo_morph.morphing.core import MorphResult, morph_images, morph_with_landmarks

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ubo_morph.landmarks import Landmarks


def align_face_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, Landmarks, Landmarks]:
    return _impl(device).align_face_images(image1, image2, landmarks1, landmarks2)


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    blending_factor: float,
    device: str = "cpu",
) -> np.ndarray:
    return _impl(device).blend_images(image1, image2, blending_factor)


def delaunay_triangles(
    points: np.ndarray,
    device: str = "cpu",
) -> list[tuple[int, int, int]]:
    """Compute Delaunay triangle indices on the requested device."""
    return _impl(device).delaunay_triangles(points)


def equalize_face(
    reference_image: np.ndarray,
    image_to_equalize: np.ndarray,
    reference_points: np.ndarray,
    points_to_equalize: np.ndarray,
    *,
    method: str = "color",
    device: str = "cpu",
) -> np.ndarray:
    return _impl(device).equalize_face(
        reference_image,
        image_to_equalize,
        reference_points,
        points_to_equalize,
        method=method,
    )


def substitute_background(
    image: np.ndarray,
    reference_points: np.ndarray,
    background_image: np.ndarray,
    *,
    blend: bool,
    eye_distance: float,
    device: str = "cpu",
) -> np.ndarray:
    return _impl(device).substitute_background(
        image,
        reference_points,
        background_image,
        blend=blend,
        eye_distance=eye_distance,
    )


def warp_image_by_triangles(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    triangles: Sequence[tuple[int, int, int]],
    device: str = "cpu",
) -> np.ndarray:
    return _impl(device).warp_image_by_triangles(image, source_points, target_points, triangles)


__all__ = [
    "MorphResult",
    "align_face_images",
    "blend_images",
    "delaunay_triangles",
    "equalize_face",
    "morph_images",
    "morph_with_landmarks",
    "substitute_background",
    "warp_image_by_triangles",
]
