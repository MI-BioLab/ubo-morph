from __future__ import annotations

import cv2
import numpy as np

from ubo_morph.utils import round_away


def equalize_face(
    reference_image: np.ndarray,
    image_to_equalize: np.ndarray,
    reference_points: np.ndarray,
    points_to_equalize: np.ndarray,
    *,
    method: str = "color",
) -> np.ndarray:
    method = method.lower()
    if method not in {"color", "lightness"}:
        raise ValueError('equalization method must be "color" or "lightness"')
    reference_mask = _convex_hull_mask(reference_image.shape[:2], reference_points)
    equalize_mask = _convex_hull_mask(image_to_equalize.shape[:2], points_to_equalize)
    if method == "color":
        result = image_to_equalize.copy()
        for channel_index in range(3):
            result[:, :, channel_index] = _equalize_channel_to_histogram(
                image_to_equalize[:, :, channel_index],
                equalize_mask,
                reference_image[:, :, channel_index],
                reference_mask,
            )
        return result

    hls = cv2.cvtColor(image_to_equalize, cv2.COLOR_BGR2HLS)
    reference_hls = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HLS)
    hls[:, :, 1] = _equalize_channel_to_histogram(
        hls[:, :, 1],
        equalize_mask,
        reference_hls[:, :, 1],
        reference_mask,
    )
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


def substitute_background(
    image: np.ndarray,
    reference_points: np.ndarray,
    background_image: np.ndarray,
    *,
    blend: bool,
    eye_distance: float,
) -> np.ndarray:
    mask = _convex_hull_mask(image.shape[:2], reference_points).astype(np.uint8)
    if not blend:
        return np.where(
            mask[:, :, None].astype(bool),
            image,
            background_image,
        ).astype(np.uint8)

    element_size = max(round_away(0.15 * eye_distance), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_3)
    distance[0, :] = 0.0
    distance[-1, :] = 0.0
    distance[:, 0] = 0.0
    distance[:, -1] = 0.0
    transition = max(element_size // 2, 1)
    alpha = np.clip((distance - 1.0) / transition, 0.0, 1.0)
    alpha[eroded > 0] = 1.0
    alpha[mask == 0] = 0.0
    alpha = alpha[:, :, None]
    blended = alpha * image.astype(np.float32) + (1.0 - alpha) * background_image.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _convex_hull_mask(
    shape: tuple[int, int],
    points: np.ndarray,
) -> np.ndarray:
    height, width = shape
    if len(points) == 0:
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


def _equalize_channel_to_histogram(
    channel: np.ndarray,
    mask: np.ndarray,
    reference_channel: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    if not mask.any() or not reference_mask.any():
        return channel.copy()
    mask_uint8 = mask.astype(np.uint8) * 255
    reference_mask_uint8 = reference_mask.astype(np.uint8) * 255
    input_histogram = cv2.calcHist([channel], [0], mask_uint8, [256], [0, 256]).ravel()
    reference_histogram = cv2.calcHist(
        [reference_channel], [0], reference_mask_uint8, [256], [0, 256]
    ).ravel()
    input_cdf = np.cumsum(input_histogram, dtype=np.float64)
    input_cdf /= input_cdf[-1]
    reference_cdf = np.cumsum(reference_histogram, dtype=np.float64)
    reference_cdf /= reference_cdf[-1]
    lookup = np.searchsorted(reference_cdf, input_cdf, side="left")
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    matched = cv2.LUT(channel, lookup)
    equalized = channel.copy()
    equalized[mask] = matched[mask]
    return equalized
