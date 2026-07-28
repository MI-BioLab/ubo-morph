from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import cv2
import numpy as np

from ubo_morph.morphing.backend import Backend, BackendName


class CPUBackend(Backend[np.ndarray]):
    """OpenCV/NumPy implementation of the morphing backend contract."""

    name: ClassVar[BackendName] = "cpu"

    def to_backend(self, array: np.ndarray) -> np.ndarray:
        return array

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        return array

    def pad_image(
        self,
        image: np.ndarray,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        top, bottom, left, right = padding
        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def warp_affine(
        self,
        image: np.ndarray,
        matrix: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = output_shape
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def resize_image(
        self,
        image: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        target_height, target_width = output_shape
        height, width = image.shape[:2]
        scale_x = target_width / width
        scale_y = target_height / height
        interpolation = (
            cv2.INTER_CUBIC if scale_x > 1.0 or scale_y > 1.0 else cv2.INTER_AREA
        )
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation,
        )

    def match_histogram_channel(
        self,
        channel: np.ndarray,
        mask: np.ndarray,
        reference_channel: np.ndarray,
        reference_mask: np.ndarray,
    ) -> np.ndarray:
        mask = mask.astype(bool, copy=False)
        reference_mask = reference_mask.astype(bool, copy=False)
        mask_uint8 = mask.astype(np.uint8) * 255
        reference_mask_uint8 = reference_mask.astype(np.uint8) * 255
        input_histogram = cv2.calcHist(
            [channel],
            [0],
            mask_uint8,
            [256],
            [0, 256],
        ).ravel()
        reference_histogram = cv2.calcHist(
            [reference_channel],
            [0],
            reference_mask_uint8,
            [256],
            [0, 256],
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

    def equalize_lightness(
        self,
        reference_image: np.ndarray,
        image_to_equalize: np.ndarray,
        reference_mask: np.ndarray,
        equalize_mask: np.ndarray,
    ) -> np.ndarray:
        hls = cv2.cvtColor(image_to_equalize, cv2.COLOR_BGR2HLS)
        reference_hls = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HLS)
        hls[:, :, 1] = self.match_histogram_channel(
            hls[:, :, 1],
            equalize_mask,
            reference_hls[:, :, 1],
            reference_mask,
        )
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

    def copy_with_mask(
        self,
        image: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        result = background.copy()
        cv2.copyTo(image, mask.astype(np.uint8, copy=False), result)
        return result

    def feather_fields(
        self,
        mask: np.ndarray,
        element_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask_uint8 = mask.astype(np.uint8, copy=False)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (element_size, element_size),
        )
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_3)
        return eroded, distance

    def blend(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        blending_factor: float = 0.5,
        *,
        foreground_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        if foreground_alpha is not None:
            return cv2.blendLinear(
                image1,
                image2,
                foreground_alpha,
                1.0 - foreground_alpha,
            )
        else:
            return cv2.addWeighted(
                image1,
                1.0 - blending_factor,
                image2,
                blending_factor,
                0.0,
            )

    def warp_triangles(
        self,
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
            source_x, source_y, source_width, source_height = cv2.boundingRect(
                source_triangle
            )
            target_x, target_y, target_width, target_height = cv2.boundingRect(
                target_triangle
            )
            if min(source_width, source_height, target_width, target_height) <= 0:
                continue
            source_crop = image[
                source_y : source_y + source_height,
                source_x : source_x + source_width,
            ]
            if source_crop.size == 0:
                continue
            source_local = source_triangle - np.array(
                [source_x, source_y],
                dtype=np.float32,
            )
            target_local = target_triangle - np.array(
                [target_x, target_y],
                dtype=np.float32,
            )
            matrix = cv2.getAffineTransform(source_local, target_local).astype(
                np.float32
            )
            warped_crop = cv2.warpAffine(
                source_crop,
                matrix,
                (target_width, target_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = np.zeros((target_height, target_width), dtype=np.float32)
            cv2.fillConvexPoly(  # ty: ignore[no-matching-overload]
                mask,
                np.int32(target_local),
                [1.0],
                lineType=cv2.LINE_AA,
            )
            region = warped[
                target_y : target_y + target_height,
                target_x : target_x + target_width,
            ]
            if region.shape != warped_crop.shape:
                continue
            region[:] = cv2.blendLinear(region, warped_crop, 1.0 - mask, mask)
        return warped
