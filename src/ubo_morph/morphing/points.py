from __future__ import annotations

import cv2
import numpy as np


def add_border_points(
    image: np.ndarray,
    points: np.ndarray,
    points_per_border: int = 5,
) -> np.ndarray:
    if points_per_border < 2:
        raise ValueError("points_per_border must be at least 2")
    height, width = image.shape[:2]
    points = np.asarray(points, dtype=np.float32)
    corners = np.array(
        [
            [0.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    x_positions = np.linspace(
        0.0,
        float(width),
        points_per_border,
        dtype=np.float32,
    )[1:-1]
    y_positions = np.linspace(
        0.0,
        float(height),
        points_per_border,
        dtype=np.float32,
    )[1:-1]
    vertical = np.column_stack(
        (
            np.tile([0.0, width - 1.0], len(y_positions)),
            np.repeat(y_positions, 2),
        )
    )
    horizontal = np.column_stack(
        (
            np.repeat(x_positions, 2),
            np.tile([0.0, height - 1.0], len(x_positions)),
        )
    )
    return np.vstack((points, corners, vertical, horizontal))


def remove_border_points(
    points: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    keep = (
        (points[:, 0] != 0.0)
        & (points[:, 1] != 0.0)
        & (points[:, 0] != float(image_width - 1))
        & (points[:, 1] != float(image_height - 1))
    )
    return points[keep]


def remove_overlapped_points(
    points1: np.ndarray,
    points2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Two-pass duplicate removal: pass 1 drops rows that are duplicates within
    # array1 (keeping each duplicate's *last* occurrence, via the reversed-array
    # unique trick), applying the same row filter to array2 so the two arrays stay
    # aligned. Pass 2 repeats the same process for array2, operating on the
    # already-reduced arrays from pass 1. The net effect is that no row index is
    # duplicated in either array once both passes complete.
    array1 = np.asarray(points1, dtype=np.float32)
    array2 = np.asarray(points2, dtype=np.float32)
    if len(array1) != len(array2):
        raise ValueError("point arrays must have the same length")
    for array_index in range(2):
        array = (array1, array2)[array_index]
        _, reversed_indices = np.unique(array[::-1], axis=0, return_index=True)
        indices = np.sort(len(array) - 1 - reversed_indices)
        array1, array2 = array1[indices], array2[indices]
    return array1.copy(), array2.copy()


def convex_hull_mask(
    shape: tuple[int, int],
    points: np.ndarray,
) -> np.ndarray:
    """Build a boolean mask of the filled convex hull of ``points``.

    ``shape`` is ``(height, width)``. Returns an all-``False`` mask if there
    are fewer than three points (a hull cannot be filled).
    """
    height, width = shape
    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)
    hull = cv2.convexHull(
        np.asarray(points, dtype=np.float32),
        clockwise=False,
        returnPoints=True,
    ).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(hull) < 3:
        return mask.astype(bool)
    cv2.fillConvexPoly(  # ty: ignore[no-matching-overload]
        mask,
        np.int32(hull),
        [1],
        lineType=cv2.LINE_8,
    )
    return mask.astype(bool)
