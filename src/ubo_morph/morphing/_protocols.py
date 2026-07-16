from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ubo_morph.landmarks import Landmarks


class AlignFaceImages(Protocol):
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        landmarks1: Landmarks,
        landmarks2: Landmarks,
    ) -> tuple[np.ndarray, np.ndarray, Landmarks, Landmarks]: ...


class BlendImages(Protocol):
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        blending_factor: float,
    ) -> np.ndarray: ...


class DelaunayTriangles(Protocol):
    def __call__(
        self,
        points: np.ndarray,
    ) -> list[tuple[int, int, int]]: ...


class EqualizeFace(Protocol):
    def __call__(
        self,
        reference_image: np.ndarray,
        image_to_equalize: np.ndarray,
        reference_points: np.ndarray,
        points_to_equalize: np.ndarray,
        *,
        method: str = ...,
    ) -> np.ndarray: ...


class SubstituteBackground(Protocol):
    def __call__(
        self,
        image: np.ndarray,
        reference_points: np.ndarray,
        background_image: np.ndarray,
        *,
        blend: bool,
        eye_distance: float,
    ) -> np.ndarray: ...


class WarpImageByTriangles(Protocol):
    def __call__(
        self,
        image: np.ndarray,
        source_points: np.ndarray,
        target_points: np.ndarray,
        triangles: Sequence[tuple[int, int, int]],
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class MorphingImpl:
    """Typed collection of callables supplied by a device backend."""

    align_face_images: AlignFaceImages
    blend_images: BlendImages
    delaunay_triangles: DelaunayTriangles
    equalize_face: EqualizeFace
    substitute_background: SubstituteBackground
    warp_image_by_triangles: WarpImageByTriangles
