from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ubo_morph.utils import ensure_bgr_uint8


def annotate_landmark_mesh(
    image: np.ndarray,
    mesh_points: np.ndarray,
    triangles: Sequence[tuple[int, int, int]],
    landmark_points: np.ndarray,
) -> np.ndarray:
    """Draw indexed facial landmarks and an unindexed border-aware mesh."""
    annotated = ensure_bgr_uint8(image).copy()
    mesh_points = np.asarray(mesh_points, dtype=np.float32)
    landmark_points = np.asarray(landmark_points, dtype=np.float32)
    if mesh_points.ndim != 2 or mesh_points.shape[1] != 2:
        raise ValueError("mesh_points must have shape (n, 2)")
    if landmark_points.ndim != 2 or landmark_points.shape[1] != 2:
        raise ValueError("landmark_points must have shape (n, 2)")

    short_side = max(1, min(annotated.shape[:2]))
    line_thickness = max(1, round(short_side / 900))
    point_radius = max(1, round(short_side / 500))
    font_scale = max(0.2, min(0.4, short_side / 1800))
    font_thickness = max(1, round(short_side / 1200))
    integer_mesh_points = np.rint(mesh_points).astype(np.int32)
    integer_landmark_points = np.rint(landmark_points).astype(np.int32)

    for triangle in triangles:
        triangle_points = integer_mesh_points[list(triangle)].reshape(-1, 1, 2)
        cv2.polylines(
            annotated,
            [triangle_points],
            isClosed=True,
            color=(255, 255, 0),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    for point in integer_mesh_points:
        center = (int(point[0]), int(point[1]))
        cv2.circle(
            annotated,
            center,
            point_radius,
            (0, 128, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    for landmark_index, point in enumerate(integer_landmark_points):
        center = (int(point[0]), int(point[1]))
        cv2.circle(
            annotated,
            center,
            point_radius,
            (0, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        label = str(int(landmark_index))
        (label_width, label_height), label_baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        label_gap = point_radius + 1
        label_candidates = (
            (center[0] + label_gap, center[1] - label_gap),
            (center[0] - label_gap - label_width, center[1] - label_gap),
            (center[0] + label_gap, center[1] + label_gap + label_height),
            (
                center[0] - label_gap - label_width,
                center[1] + label_gap + label_height,
            ),
        )
        image_height, image_width = annotated.shape[:2]
        label_origin = next(
            (
                candidate
                for candidate in label_candidates
                if 0 <= candidate[0]
                and candidate[0] + label_width < image_width
                and candidate[1] - label_height >= 0
                and candidate[1] + label_baseline < image_height
            ),
            (
                min(max(label_candidates[0][0], 0), max(image_width - label_width, 0)),
                min(
                    max(label_candidates[0][1], label_height),
                    max(image_height - label_baseline - 1, label_height),
                ),
            ),
        )
        cv2.putText(
            annotated,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness=font_thickness + 1,
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
