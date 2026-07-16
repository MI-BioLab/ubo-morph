from __future__ import annotations

import cv2
import numpy as np


def delaunay_triangles(points: np.ndarray) -> list[tuple[int, int, int]]:
    """OpenCV implementation of Delaunay triangulation."""
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
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))

    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    raw_triangles = np.asarray(
        subdiv.getTriangleList(),
        dtype=np.float32,
    ).reshape(-1, 3, 2)
    for triangle_points in raw_triangles:
        indices = _nearest_point_indices(points, triangle_points)
        if indices is None:
            continue
        sorted_indices = sorted(indices)
        key = (sorted_indices[0], sorted_indices[1], sorted_indices[2])
        if key in seen:
            continue
        seen.add(key)
        triangles.append(indices)
    return triangles


def _nearest_point_indices(
    points: np.ndarray,
    triangle_points: np.ndarray,
) -> tuple[int, int, int] | None:
    distances = np.linalg.norm(
        triangle_points[:, np.newaxis, :] - points[np.newaxis, :, :],
        axis=2,
    )
    indices = np.argmin(distances, axis=1)
    if np.any(distances[np.arange(3), indices] > 1e-3):
        return None
    result = (int(indices[0]), int(indices[1]), int(indices[2]))
    if len(set(result)) != 3:
        return None
    return result
