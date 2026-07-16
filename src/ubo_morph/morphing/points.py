from __future__ import annotations

import numpy as np


def add_border_points(
    image: np.ndarray,
    points: np.ndarray,
    point_per_border_count: int = 5,
) -> np.ndarray:
    if point_per_border_count < 2:
        raise ValueError("point_per_border_count must be at least 2")
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
        point_per_border_count,
        dtype=np.float32,
    )[1:-1]
    y_positions = np.linspace(
        0.0,
        float(height),
        point_per_border_count,
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
