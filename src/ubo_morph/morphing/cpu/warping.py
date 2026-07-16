from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def warp_image_by_triangles(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    triangles: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    warped = np.zeros_like(image)
    for triangle in triangles:
        indices = list(triangle)
        source_triangle = source_points[indices].astype(np.float32)
        target_triangle = target_points[indices].astype(np.float32)
        source_x, source_y, source_width, source_height = cv2.boundingRect(source_triangle)
        target_x, target_y, target_width, target_height = cv2.boundingRect(target_triangle)
        if min(source_width, source_height, target_width, target_height) <= 0:
            continue
        source_crop = image[source_y : source_y + source_height, source_x : source_x + source_width]
        if source_crop.size == 0:
            continue
        source_local = source_triangle - np.array([source_x, source_y], dtype=np.float32)
        target_local = target_triangle - np.array([target_x, target_y], dtype=np.float32)
        matrix = cv2.getAffineTransform(source_local, target_local).astype(np.float32)
        warped_crop = cv2.warpAffine(
            source_crop,
            matrix,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = np.zeros((target_height, target_width), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(target_local), [1.0], lineType=cv2.LINE_AA)  # ty: ignore[no-matching-overload]
        region = warped[target_y : target_y + target_height, target_x : target_x + target_width]
        if region.shape != warped_crop.shape:
            continue
        alpha = mask[:, :, None]
        blended = region.astype(np.float32) * (1.0 - alpha) + warped_crop.astype(np.float32) * alpha
        region[:] = np.clip(blended, 0, 255).astype(np.uint8)
    return warped
