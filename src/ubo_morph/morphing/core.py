from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from ubo_morph.device import validate_device
from ubo_morph.landmarks import LandmarkExtractor, Landmarks
from ubo_morph.morphing._backends import _impl
from ubo_morph.morphing._protocols import MorphingImpl
from ubo_morph.morphing.points import (
    add_border_points as _add_border_points,
    remove_border_points,
    remove_overlapped_points,
)
from ubo_morph.utils import ensure_bgr_uint8


@dataclass(slots=True)
class MorphResult:
    image: np.ndarray
    morphed_points: np.ndarray
    warped_image1: np.ndarray
    warped_image2: np.ndarray
    aligned_image1: np.ndarray
    aligned_image2: np.ndarray
    aligned_landmarks1: Landmarks
    aligned_landmarks2: Landmarks


@overload
def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: Literal[False] = False,
) -> np.ndarray: ...


@overload
def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: Literal[True] = True,
) -> MorphResult: ...


@overload
def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: bool,
) -> np.ndarray | MorphResult: ...


def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: bool = False,
) -> np.ndarray | MorphResult:
    selected_device = validate_device(device)
    image1 = ensure_bgr_uint8(image1)
    image2 = ensure_bgr_uint8(image2)
    landmarks1 = landmark_extractor.extract(image1)
    landmarks2 = landmark_extractor.extract(image2)
    return morph_with_landmarks(
        image1,
        image2,
        landmarks1,
        landmarks2,
        warping_factor=warping_factor,
        blending_factor=blending_factor,
        align_eye_centers=align_eye_centers,
        add_border_points=add_border_points,
        automatic_retouching=automatic_retouching,
        color_equalization=color_equalization,
        equalization_method=equalization_method,
        blend_background=blend_background,
        device=selected_device,
        return_details=return_details,
    )


@overload
def morph_with_landmarks(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: Literal[False] = False,
) -> np.ndarray: ...


@overload
def morph_with_landmarks(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: Literal[True] = True,
) -> MorphResult: ...


@overload
def morph_with_landmarks(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: bool,
) -> np.ndarray | MorphResult: ...


def morph_with_landmarks(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    *,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    add_border_points: bool = True,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    device: str = "cpu",
    return_details: bool = False,
) -> np.ndarray | MorphResult:
    selected_device = validate_device(device)
    image1 = ensure_bgr_uint8(image1)
    image2 = ensure_bgr_uint8(image2)

    warping_factor = float(warping_factor)
    blending_factor = float(blending_factor)
    if not 0.0 <= warping_factor <= 1.0:
        raise ValueError("warping_factor must be in [0, 1]")
    if not 0.0 <= blending_factor <= 1.0:
        raise ValueError("blending_factor must be in [0, 1]")

    aligned_image1 = image1
    aligned_image2 = image2
    aligned_landmarks1 = landmarks1
    aligned_landmarks2 = landmarks2
    impl = _impl(selected_device)
    if align_eye_centers:
        (
            aligned_image1,
            aligned_image2,
            aligned_landmarks1,
            aligned_landmarks2,
        ) = impl.align_face_images(image1, image2, landmarks1, landmarks2)

    if aligned_image1.shape[:2] != aligned_image2.shape[:2]:
        raise ValueError(
            "the two images must have the same size; "
            "enable eye alignment to auto-resize"
        )
    if len(aligned_landmarks1.points) != len(aligned_landmarks2.points):
        raise ValueError(
            "the two extractors must return the same number of landmark points"
        )

    morph_points1 = aligned_landmarks1.points
    morph_points2 = aligned_landmarks2.points
    if add_border_points:
        morph_points1 = _add_border_points(aligned_image1, morph_points1, 5)
        morph_points2 = _add_border_points(aligned_image2, morph_points2, 5)
    morph_points1, morph_points2 = remove_overlapped_points(
        morph_points1,
        morph_points2,
    )

    work_image1 = aligned_image1
    work_image2 = aligned_image2
    if automatic_retouching and color_equalization:
        equalization_points1, equalization_points2 = remove_overlapped_points(
            aligned_landmarks1.points,
            aligned_landmarks2.points,
        )
        if blending_factor <= 0.5:
            work_image2 = impl.equalize_face(
                aligned_image1,
                aligned_image2,
                equalization_points1,
                equalization_points2,
                method=equalization_method,
            )
        else:
            work_image1 = impl.equalize_face(
                aligned_image2,
                aligned_image1,
                equalization_points2,
                equalization_points1,
                method=equalization_method,
            )

    morphed, morphed_points, warped1, warped2 = _morph_image_pair(
        work_image1,
        work_image2,
        morph_points1,
        morph_points2,
        warping_factor,
        blending_factor,
        impl=impl,
    )

    if automatic_retouching:
        background = warped1 if (1.0 - blending_factor) >= blending_factor else warped2
        face_points = remove_border_points(
            morphed_points,
            morphed.shape[1],
            morphed.shape[0],
        )
        eye_distance = float(
            np.linalg.norm(aligned_landmarks1.left_eye - aligned_landmarks1.right_eye)
        )
        morphed = impl.substitute_background(
            morphed,
            face_points,
            background,
            blend=blend_background,
            eye_distance=eye_distance,
        )

    if not return_details:
        return morphed
    return MorphResult(
        image=morphed,
        morphed_points=morphed_points,
        warped_image1=warped1,
        warped_image2=warped2,
        aligned_image1=aligned_image1,
        aligned_image2=aligned_image2,
        aligned_landmarks1=aligned_landmarks1,
        aligned_landmarks2=aligned_landmarks2,
    )


def _morph_image_pair(
    image1: np.ndarray,
    image2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    warping_factor: float,
    blending_factor: float,
    *,
    impl: MorphingImpl,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if image1.shape != image2.shape:
        raise ValueError("input images must have identical shape")
    if len(points1) != len(points2):
        raise ValueError("point arrays must have the same length")
    morphed_points = points1 + (points2 - points1) * warping_factor
    triangles = impl.delaunay_triangles(points1)
    warped1 = impl.warp_image_by_triangles(image1, points1, morphed_points, triangles)
    warped2 = impl.warp_image_by_triangles(image2, points2, morphed_points, triangles)
    morphed = impl.blend_images(warped1, warped2, blending_factor)
    return morphed, morphed_points, warped1, warped2
