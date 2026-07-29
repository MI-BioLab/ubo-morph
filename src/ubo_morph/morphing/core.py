from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, overload

import cv2
import numpy as np

from ubo_morph.landmarks import LandmarkExtractor, Landmarks
from ubo_morph.morphing.alignment import alignment_geometry, scale_landmarks
from ubo_morph.morphing.backend import Backend, BackendName, get_backend
from ubo_morph.morphing.points import (
    add_border_points,
    non_overlapped_point_indices,
    remove_border_points,
    remove_overlapped_points,
)
from ubo_morph.utils import ensure_bgr_uint8, round_away


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
    original_landmarks1: Landmarks
    original_landmarks2: Landmarks
    before_background_substitution: np.ndarray | None = None
    after_equalization_image1: np.ndarray | None = None
    after_equalization_image2: np.ndarray | None = None
    source_points1: np.ndarray | None = None
    source_points2: np.ndarray | None = None
    point_landmark_indices: np.ndarray | None = None
    triangles: list[tuple[int, int, int]] = field(default_factory=list)


@overload
def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    landmark_extraction_short_side: int = 0,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: Literal[False] = False,
    backend: BackendName = "cpu",
) -> np.ndarray: ...


@overload
def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    landmark_extraction_short_side: int = 0,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: Literal[True],
    backend: BackendName = "cpu",
) -> MorphResult: ...


@overload
def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    landmark_extraction_short_side: int = 0,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: bool,
    backend: BackendName = "cpu",
) -> np.ndarray | MorphResult: ...


def morph_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmark_extractor: LandmarkExtractor,
    *,
    landmark_extraction_short_side: int = 0,
    warping_factor: float = 0.5,
    blending_factor: float = 0.5,
    align_eye_centers: bool = True,
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: bool = False,
    backend: BackendName = "cpu",
) -> np.ndarray | MorphResult:
    resolved_backend = get_backend(backend)
    landmarks1 = landmark_extractor.extract(
        image1,
        max_short_side=landmark_extraction_short_side,
    )
    landmarks2 = landmark_extractor.extract(
        image2,
        max_short_side=landmark_extraction_short_side,
    )
    return _morph_pipeline(
        image1,
        image2,
        landmarks1,
        landmarks2,
        backend=resolved_backend,
        warping_factor=warping_factor,
        blending_factor=blending_factor,
        align_eye_centers=align_eye_centers,
        points_per_border=points_per_border,
        automatic_retouching=automatic_retouching,
        color_equalization=color_equalization,
        equalization_method=equalization_method,
        blend_background=blend_background,
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
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: Literal[False] = False,
    backend: BackendName = "cpu",
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
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: Literal[True],
    backend: BackendName = "cpu",
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
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: bool,
    backend: BackendName = "cpu",
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
    points_per_border: int = 5,
    automatic_retouching: bool = True,
    color_equalization: bool = True,
    equalization_method: str = "color",
    blend_background: bool = True,
    return_details: bool = False,
    backend: BackendName = "cpu",
) -> np.ndarray | MorphResult:
    resolved_backend = get_backend(backend)
    return _morph_pipeline(
        image1,
        image2,
        landmarks1,
        landmarks2,
        backend=resolved_backend,
        warping_factor=warping_factor,
        blending_factor=blending_factor,
        align_eye_centers=align_eye_centers,
        points_per_border=points_per_border,
        automatic_retouching=automatic_retouching,
        color_equalization=color_equalization,
        equalization_method=equalization_method,
        blend_background=blend_background,
        return_details=return_details,
    )


def _morph_pipeline(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    *,
    backend: Backend[Any],
    warping_factor: float,
    blending_factor: float,
    align_eye_centers: bool,
    points_per_border: int,
    automatic_retouching: bool,
    color_equalization: bool,
    equalization_method: str,
    blend_background: bool,
    return_details: bool,
) -> np.ndarray | MorphResult:
    image1 = ensure_bgr_uint8(image1)
    image2 = ensure_bgr_uint8(image2)
    warping_factor, blending_factor = _validate_pipeline_options(
        warping_factor,
        blending_factor,
        points_per_border,
    )

    backend_image1 = backend.to_backend(image1)
    backend_image2 = backend.to_backend(image2)
    aligned_image1 = backend_image1
    aligned_image2 = backend_image2
    aligned_landmarks1 = landmarks1
    aligned_landmarks2 = landmarks2
    if align_eye_centers:
        (
            aligned_image1,
            aligned_image2,
            aligned_landmarks1,
            aligned_landmarks2,
        ) = _align_face_images(
            backend,
            backend_image1,
            backend_image2,
            landmarks1,
            landmarks2,
        )

    if aligned_image1.shape[:2] != aligned_image2.shape[:2]:
        raise ValueError(
            "the two images must have the same size; enable eye alignment to auto-resize"
        )
    if len(aligned_landmarks1.points) != len(aligned_landmarks2.points):
        raise ValueError(
            "the two extractors must return the same number of landmark points"
        )

    morph_points1 = aligned_landmarks1.points
    morph_points2 = aligned_landmarks2.points
    point_landmark_indices = np.arange(len(morph_points1), dtype=np.int32)
    if points_per_border:
        morph_points1 = add_border_points(
            aligned_image1,
            morph_points1,
            points_per_border,
        )
        morph_points2 = add_border_points(
            aligned_image2,
            morph_points2,
            points_per_border,
        )
        border_point_count = len(morph_points1) - len(point_landmark_indices)
        point_landmark_indices = np.concatenate(
            (
                point_landmark_indices,
                np.full(border_point_count, -1, dtype=np.int32),
            )
        )
    retained_point_indices = non_overlapped_point_indices(
        morph_points1,
        morph_points2,
        point_priorities=point_landmark_indices >= 0,
    )
    morph_points1 = morph_points1[retained_point_indices].copy()
    morph_points2 = morph_points2[retained_point_indices].copy()
    point_landmark_indices = point_landmark_indices[retained_point_indices].copy()

    work_image1 = aligned_image1
    work_image2 = aligned_image2
    after_equalization_image1: Any | None = None
    after_equalization_image2: Any | None = None
    if automatic_retouching and color_equalization:
        equalization_points1, equalization_points2 = remove_overlapped_points(
            aligned_landmarks1.points,
            aligned_landmarks2.points,
        )
        if blending_factor <= 0.5:
            work_image2 = _equalize_face(
                backend,
                aligned_image1,
                aligned_image2,
                equalization_points1,
                equalization_points2,
                method=equalization_method,
            )
            after_equalization_image2 = work_image2
        else:
            work_image1 = _equalize_face(
                backend,
                aligned_image2,
                aligned_image1,
                equalization_points2,
                equalization_points1,
                method=equalization_method,
            )
            after_equalization_image1 = work_image1

    morphed_points = (
        morph_points1 + (morph_points2 - morph_points1) * warping_factor
    )
    triangles = _delaunay_triangles(morph_points1)
    warped_image1 = backend.warp_triangles(
        work_image1,
        morph_points1,
        morphed_points,
        triangles,
    )
    warped_image2 = backend.warp_triangles(
        work_image2,
        morph_points2,
        morphed_points,
        triangles,
    )
    before_background = backend.blend(
        warped_image1,
        warped_image2,
        blending_factor,
    )
    morphed_image = before_background

    if automatic_retouching:
        background = warped_image1 if blending_factor <= 0.5 else warped_image2
        face_points = remove_border_points(
            morphed_points,
            int(morphed_image.shape[1]),
            int(morphed_image.shape[0]),
        )
        eye_distance = float(
            np.linalg.norm(aligned_landmarks1.left_eye - aligned_landmarks1.right_eye)
        )
        morphed_image = _substitute_background(
            backend,
            morphed_image,
            face_points,
            background,
            blend=blend_background,
            eye_distance=eye_distance,
        )

    result_image = backend.to_numpy(morphed_image)
    if not return_details:
        return result_image
    return MorphResult(
        image=result_image,
        morphed_points=morphed_points,
        source_points1=morph_points1,
        source_points2=morph_points2,
        point_landmark_indices=point_landmark_indices,
        triangles=triangles,
        warped_image1=backend.to_numpy(warped_image1),
        warped_image2=backend.to_numpy(warped_image2),
        aligned_image1=backend.to_numpy(aligned_image1),
        aligned_image2=backend.to_numpy(aligned_image2),
        aligned_landmarks1=aligned_landmarks1,
        aligned_landmarks2=aligned_landmarks2,
        original_landmarks1=landmarks1,
        original_landmarks2=landmarks2,
        before_background_substitution=(
            backend.to_numpy(before_background) if automatic_retouching else None
        ),
        after_equalization_image1=(
            None
            if after_equalization_image1 is None
            else backend.to_numpy(after_equalization_image1)
        ),
        after_equalization_image2=(
            None
            if after_equalization_image2 is None
            else backend.to_numpy(after_equalization_image2)
        ),
    )


def _validate_pipeline_options(
    warping_factor: float,
    blending_factor: float,
    points_per_border: int,
) -> tuple[float, float]:
    warping_factor = float(warping_factor)
    blending_factor = float(blending_factor)
    if not 0.0 <= warping_factor <= 1.0:
        raise ValueError("warping_factor must be in [0, 1]")
    if not 0.0 <= blending_factor <= 1.0:
        raise ValueError("blending_factor must be in [0, 1]")
    if points_per_border < 0 or points_per_border == 1:
        raise ValueError("points_per_border must be 0 or at least 2")
    return warping_factor, blending_factor


def _align_face_images(
    backend: Backend[Any],
    image1: Any,
    image2: Any,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
) -> tuple[Any, Any, Landmarks, Landmarks]:
    tokenized1, aligned_landmarks1 = _tokenize_image_and_landmarks(
        backend,
        image1,
        landmarks1,
    )
    tokenized2, aligned_landmarks2 = _tokenize_image_and_landmarks(
        backend,
        image2,
        landmarks2,
    )
    target_shape = (
        max(tokenized1.shape[0], tokenized2.shape[0]),
        max(tokenized1.shape[1], tokenized2.shape[1]),
    )
    output1, aligned_landmarks1 = _resize_image_and_landmarks(
        backend,
        tokenized1,
        aligned_landmarks1,
        target_shape,
    )
    output2, aligned_landmarks2 = _resize_image_and_landmarks(
        backend,
        tokenized2,
        aligned_landmarks2,
        target_shape,
    )
    return output1, output2, aligned_landmarks1, aligned_landmarks2


def _tokenize_image_and_landmarks(
    backend: Backend[Any],
    image: Any,
    landmarks: Landmarks,
) -> tuple[Any, Landmarks]:
    geometry = alignment_geometry(image.shape[:2], landmarks)
    border_x, border_y = geometry.border
    padded = backend.pad_image(
        image,
        (border_y, border_y, border_x, border_x),
    )
    rotated = backend.warp_affine(
        padded,
        geometry.rotation_matrix,
        geometry.padded_shape,
    )
    if any(geometry.crop_padding):
        rotated = backend.pad_image(rotated, geometry.crop_padding)
    crop_left, crop_top = geometry.crop_origin
    crop_height, crop_width = geometry.crop_shape
    cropped = rotated[
        crop_top : crop_top + crop_height,
        crop_left : crop_left + crop_width,
    ].copy()
    return cropped, geometry.landmarks


def _resize_image_and_landmarks(
    backend: Backend[Any],
    image: Any,
    landmarks: Landmarks,
    target_shape: tuple[int, int],
) -> tuple[Any, Landmarks]:
    height, width = image.shape[:2]
    target_height, target_width = target_shape
    if (height, width) == target_shape:
        return image, landmarks
    resized = backend.resize_image(image, target_shape)
    return resized, scale_landmarks(
        landmarks,
        target_width / width,
        target_height / height,
    )


def _equalize_face(
    backend: Backend[Any],
    reference_image: Any,
    image_to_equalize: Any,
    reference_points: np.ndarray,
    points_to_equalize: np.ndarray,
    *,
    method: str,
) -> Any:
    method = method.lower()
    if method not in {"color", "lightness"}:
        raise ValueError('equalization method must be "color" or "lightness"')
    reference_mask = backend.convex_hull_mask(
        reference_image.shape[:2],
        reference_points,
    )
    equalize_mask = backend.convex_hull_mask(
        image_to_equalize.shape[:2],
        points_to_equalize,
    )
    if not equalize_mask.any() or not reference_mask.any():
        return image_to_equalize.copy()
    if method == "lightness":
        return backend.equalize_lightness(
            reference_image,
            image_to_equalize,
            reference_mask,
            equalize_mask,
        )

    return backend.match_histogram_image(
        image_to_equalize,
        equalize_mask,
        reference_image,
        reference_mask,
    )


def _substitute_background(
    backend: Backend[Any],
    image: Any,
    reference_points: np.ndarray,
    background_image: Any,
    *,
    blend: bool,
    eye_distance: float,
) -> Any:
    mask = backend.convex_hull_mask(image.shape[:2], reference_points)
    if not blend:
        return backend.copy_with_mask(image, background_image, mask)
    element_size = max(round_away(0.15 * eye_distance), 1)
    transition = max(element_size // 2, 1)
    return backend.blend_with_feather(
        image,
        background_image,
        mask,
        element_size=element_size,
        transition=transition,
    )


def _delaunay_triangles(points: np.ndarray) -> list[tuple[int, int, int]]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(points) < 3:
        raise ValueError("Less than three points in point set.")

    min_xy = np.floor(points.min(axis=0)).astype(int) - 2
    max_xy = np.ceil(points.max(axis=0)).astype(int) + 2
    rect = (
        int(min_xy[0]),
        int(min_xy[1]),
        int(max_xy[0] - min_xy[0] + 1),
        int(max_xy[1] - min_xy[1] + 1),
    )
    subdivision = cv2.Subdiv2D(rect)
    vertex_indices: dict[int, int] = {}
    for index, point in enumerate(points):
        vertex_id = subdivision.insert((float(point[0]), float(point[1])))
        vertex_indices.setdefault(int(vertex_id), index)

    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    raw_triangles = np.asarray(
        subdivision.getTriangleList(),
        dtype=np.float32,
    ).reshape(-1, 3, 2)
    for triangle_points in raw_triangles:
        vertex_ids = [
            int(subdivision.findNearest((float(point[0]), float(point[1])))[0])
            for point in triangle_points
        ]
        try:
            indices = (
                vertex_indices[vertex_ids[0]],
                vertex_indices[vertex_ids[1]],
                vertex_indices[vertex_ids[2]],
            )
        except KeyError:
            continue
        if len(set(indices)) != 3:
            continue
        sorted_indices = sorted(indices)
        key = (sorted_indices[0], sorted_indices[1], sorted_indices[2])
        if key in seen:
            continue
        seen.add(key)
        triangles.append(indices)
    return triangles
