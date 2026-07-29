from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ubo_morph.utils import ensure_bgr_uint8


def annotate_landmark_mesh(
    image: np.ndarray,
    points: np.ndarray,
    triangles: Sequence[tuple[int, int, int]],
    landmark_indices: np.ndarray,
) -> np.ndarray:
    """Draw indexed facial landmarks and an unindexed border-aware mesh."""
    annotated = ensure_bgr_uint8(image).copy()
    points = np.asarray(points, dtype=np.float32)
    landmark_indices = np.asarray(landmark_indices, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if landmark_indices.shape != (len(points),):
        raise ValueError("landmark_indices must have one entry per point")

    short_side = max(1, min(annotated.shape[:2]))
    line_thickness = max(1, round(short_side / 500))
    point_radius = max(2, round(short_side / 250))
    font_scale = max(0.35, min(0.8, short_side / 900))
    font_thickness = max(1, round(short_side / 600))
    integer_points = np.rint(points).astype(np.int32)

    for triangle in triangles:
        triangle_points = integer_points[list(triangle)].reshape(-1, 1, 2)
        cv2.polylines(
            annotated,
            [triangle_points],
            isClosed=True,
            color=(255, 255, 0),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    for point, landmark_index in zip(
        integer_points,
        landmark_indices,
        strict=True,
    ):
        center = (int(point[0]), int(point[1]))
        point_color = (0, 255, 0) if landmark_index >= 0 else (0, 128, 255)
        cv2.circle(
            annotated,
            center,
            point_radius,
            point_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        if landmark_index < 0:
            continue
        label_origin = (
            center[0] + point_radius + 2,
            center[1] - point_radius - 2,
        )
        label = str(int(landmark_index))
        cv2.putText(
            annotated,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness=font_thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )
    return annotated
