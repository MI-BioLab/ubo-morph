from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import cv2
import cupy as cp
import cupyx.scipy.ndimage as cpnd
import numpy as np

from ubo_morph.morphing.backend import Backend, BackendName
from ubo_morph.morphing.cupy.kernels import load_kernel_source


def _triangles_have_interior_overlap(vertices: np.ndarray) -> bool:
    triangle_count = len(vertices)
    if triangle_count < 2:
        return False
    lower = vertices.min(axis=1)
    upper = vertices.max(axis=1)
    bounding_boxes_overlap = (
        (upper[:, None, 0] > lower[None, :, 0])
        & (lower[:, None, 0] < upper[None, :, 0])
        & (upper[:, None, 1] > lower[None, :, 1])
        & (lower[:, None, 1] < upper[None, :, 1])
    )
    candidate_pairs = np.argwhere(
        np.triu(bounding_boxes_overlap, k=1)
    )
    for first, second in candidate_pairs:
        intersection_area, _ = cv2.intersectConvexConvex(
            vertices[first],
            vertices[second],
        )
        if intersection_area > 1e-3:
            return True
    return False


class CuPyBackend(Backend[cp.ndarray]):
    """CuPy implementation of the morphing backend contract."""

    name: ClassVar[BackendName] = "cupy"

    def __init__(self) -> None:
        module = cp.RawModule(code=load_kernel_source())
        self._affine_warp_kernel = module.get_function("affine_warp_uint8")
        self._area_resize_kernel = module.get_function("area_resize_uint8")
        self._blend_kernel = module.get_function("blend_uint8")
        self._bgr_to_hls_kernel = module.get_function("bgr_to_hls")
        self._hls_to_bgr_kernel = module.get_function("hls_to_bgr")
        self._convex_hull_kernel = module.get_function("rasterize_convex_hull")
        self._histogram_kernel = module.get_function("build_histograms")
        self._lookup_kernel = module.get_function("build_histogram_lookup")
        self._apply_lookup_kernel = module.get_function("apply_histogram_lookup")
        self._feather_blend_kernel = module.get_function("feather_blend")
        self._triangle_warp_kernel = module.get_function("warp_triangle_boxes")
        self._triangle_membership_kernel = module.get_function("build_box_membership")
        self._triangle_sample_kernel = module.get_function("sample_box_membership")

    def to_backend(self, array: np.ndarray) -> cp.ndarray:
        return cp.asarray(array)

    def to_numpy(self, array: cp.ndarray) -> np.ndarray:
        return cp.asnumpy(array)

    def pad_image(
        self,
        image: cp.ndarray,
        padding: tuple[int, int, int, int],
    ) -> cp.ndarray:
        top, bottom, left, right = padding
        return cp.pad(
            image,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def warp_affine(
        self,
        image: cp.ndarray,
        matrix: np.ndarray,
        output_shape: tuple[int, int],
    ) -> cp.ndarray:
        output_height, output_width = output_shape
        height, width = image.shape[:2]
        inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
        contiguous = cp.ascontiguousarray(image)
        output = cp.empty(
            (output_height, output_width, image.shape[2]),
            dtype=cp.uint8,
        )
        pixel_count = output_height * output_width
        threads = 256
        self._affine_warp_kernel(
            ((pixel_count + threads - 1) // threads,),
            (threads,),
            (
                contiguous,
                np.int32(height),
                np.int32(width),
                np.int32(output_height),
                np.int32(output_width),
                *(np.float32(value) for value in inverse.ravel()),
                output,
            ),
        )
        return output

    def resize_image(
        self,
        image: cp.ndarray,
        output_shape: tuple[int, int],
    ) -> cp.ndarray:
        target_height, target_width = output_shape
        height, width = image.shape[:2]
        zoom_y = target_height / height
        zoom_x = target_width / width
        if zoom_y <= 1.0 and zoom_x <= 1.0:
            contiguous = cp.ascontiguousarray(image)
            output = cp.empty(
                (target_height, target_width, image.shape[2]),
                dtype=cp.uint8,
            )
            pixel_count = target_height * target_width
            threads = 256
            self._area_resize_kernel(
                ((pixel_count + threads - 1) // threads,),
                (threads,),
                (
                    contiguous,
                    np.int32(height),
                    np.int32(width),
                    np.int32(target_height),
                    np.int32(target_width),
                    output,
                ),
            )
            return output
        return cpnd.zoom(
            image,
            (zoom_y, zoom_x, 1.0),
            output=cp.uint8,
            order=3,
        )

    def convex_hull_mask(
        self,
        shape: tuple[int, int],
        points: np.ndarray,
    ) -> cp.ndarray:
        height, width = shape
        output = cp.zeros((height, width), dtype=cp.bool_)
        if len(points) < 3:
            return output
        hull = cv2.convexHull(
            np.asarray(points, dtype=np.float32),
            clockwise=False,
            returnPoints=True,
        ).reshape(-1, 2)
        hull = np.asarray(hull, dtype=np.int32)
        if len(hull) < 3:
            return output

        triangle_count = len(hull) - 2
        vertices = np.empty((triangle_count, 3, 2), dtype=np.float32)
        vertices[:, 0] = hull[0]
        vertices[:, 1] = hull[1:-1]
        vertices[:, 2] = hull[2:]
        edge1 = vertices[:, 1] - vertices[:, 0]
        edge2 = vertices[:, 2] - vertices[:, 0]
        signed_areas = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        reversed_triangles = signed_areas < 0.0
        vertices[reversed_triangles, 1:3] = vertices[
            reversed_triangles, 2:0:-1
        ]

        descriptors = np.empty((triangle_count, 10), dtype=np.float32)
        descriptors[:, :6] = vertices.reshape(triangle_count, 6)
        lower = np.floor(vertices.min(axis=1)).astype(np.int64)
        upper = np.ceil(vertices.max(axis=1)).astype(np.int64)
        lower[:, 0] = np.maximum(lower[:, 0], 0)
        lower[:, 1] = np.maximum(lower[:, 1], 0)
        upper[:, 0] = np.minimum(upper[:, 0], width - 1)
        upper[:, 1] = np.minimum(upper[:, 1], height - 1)
        box_sizes = np.maximum(upper - lower + 1, 0)
        box_sizes[np.abs(signed_areas) <= np.finfo(np.float32).eps] = 0
        descriptors[:, 6:8] = lower
        descriptors[:, 8:10] = box_sizes

        work_sizes = box_sizes[:, 0] * box_sizes[:, 1]
        offsets = np.empty(triangle_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(work_sizes, out=offsets[1:])
        work_count = int(offsets[-1])
        if work_count == 0:
            return output

        threads = 256
        self._convex_hull_kernel(
            ((work_count + threads - 1) // threads,),
            (threads,),
            (
                cp.asarray(descriptors),
                cp.asarray(offsets),
                np.int32(triangle_count),
                np.int32(width),
                np.int64(work_count),
                output,
            ),
        )
        return output

    def match_histogram_channel(
        self,
        channel: cp.ndarray,
        mask: cp.ndarray,
        reference_channel: cp.ndarray,
        reference_mask: cp.ndarray,
    ) -> cp.ndarray:
        return self._match_histograms(
            channel,
            mask,
            reference_channel,
            reference_mask,
            channels=1,
        )

    def match_histogram_image(
        self,
        image: cp.ndarray,
        mask: cp.ndarray,
        reference_image: cp.ndarray,
        reference_mask: cp.ndarray,
    ) -> cp.ndarray:
        return self._match_histograms(
            image,
            mask,
            reference_image,
            reference_mask,
            channels=3,
        )

    def _match_histograms(
        self,
        image: cp.ndarray,
        mask: cp.ndarray,
        reference_image: cp.ndarray,
        reference_mask: cp.ndarray,
        *,
        channels: int,
    ) -> cp.ndarray:
        contiguous_image = cp.ascontiguousarray(image)
        contiguous_reference = cp.ascontiguousarray(reference_image)
        contiguous_mask = cp.ascontiguousarray(mask)
        contiguous_reference_mask = cp.ascontiguousarray(reference_mask)
        input_pixel_count = int(mask.size)
        reference_pixel_count = int(reference_mask.size)
        histograms = cp.zeros((channels, 2, 256), dtype=cp.uint32)
        lookup = cp.empty((channels, 256), dtype=cp.uint8)
        output = cp.empty_like(contiguous_image)
        threads = 256
        maximum_pixel_count = max(input_pixel_count, reference_pixel_count)
        blocks = min((maximum_pixel_count + threads - 1) // threads, 256)
        self._histogram_kernel(
            (blocks,),
            (threads,),
            (
                contiguous_image,
                contiguous_mask,
                contiguous_reference,
                contiguous_reference_mask,
                np.int64(input_pixel_count),
                np.int64(reference_pixel_count),
                np.int32(channels),
                histograms,
            ),
            shared_mem=channels * 2 * 256 * np.dtype(np.uint32).itemsize,
        )
        self._lookup_kernel(
            (channels,),
            (1,),
            (
                histograms,
                np.int32(channels),
                lookup,
            ),
        )
        self._apply_lookup_kernel(
            ((input_pixel_count + threads - 1) // threads,),
            (threads,),
            (
                contiguous_image,
                contiguous_mask,
                lookup,
                np.int64(input_pixel_count),
                np.int32(channels),
                output,
            ),
        )
        return output

    def _convert_color(
        self,
        image: cp.ndarray,
        kernel: cp.RawKernel,
    ) -> cp.ndarray:
        contiguous = cp.ascontiguousarray(image)
        output = cp.empty_like(contiguous)
        pixel_count = int(image.shape[0] * image.shape[1])
        threads = 256
        kernel(
            ((pixel_count + threads - 1) // threads,),
            (threads,),
            (
                contiguous,
                np.int64(pixel_count),
                output,
            ),
        )
        return output

    def _bgr_to_hls(self, image: cp.ndarray) -> cp.ndarray:
        return self._convert_color(image, self._bgr_to_hls_kernel)

    def _hls_to_bgr(self, image: cp.ndarray) -> cp.ndarray:
        return self._convert_color(image, self._hls_to_bgr_kernel)

    def equalize_lightness(
        self,
        reference_image: cp.ndarray,
        image_to_equalize: cp.ndarray,
        reference_mask: cp.ndarray,
        equalize_mask: cp.ndarray,
    ) -> cp.ndarray:
        hls = self._bgr_to_hls(image_to_equalize)
        reference_hls = self._bgr_to_hls(reference_image)
        hls[:, :, 1] = self.match_histogram_channel(
            hls[:, :, 1],
            equalize_mask,
            reference_hls[:, :, 1],
            reference_mask,
        )
        return self._hls_to_bgr(hls)

    def copy_with_mask(
        self,
        image: cp.ndarray,
        background: cp.ndarray,
        mask: cp.ndarray,
    ) -> cp.ndarray:
        return cp.where(mask[:, :, None], image, background)

    def feather_fields(
        self,
        mask: cp.ndarray,
        element_size: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        struct = cp.asarray(
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (element_size, element_size),
            ).astype(bool)
        )
        eroded = cpnd.binary_erosion(mask, structure=struct)
        distance = cpnd.distance_transform_edt(mask, float64_distances=False)
        return eroded, distance

    def blend_with_feather(
        self,
        image: cp.ndarray,
        background: cp.ndarray,
        mask: cp.ndarray,
        *,
        element_size: int,
        transition: int,
    ) -> cp.ndarray:
        eroded, distance = self.feather_fields(mask, element_size)
        contiguous_image = cp.ascontiguousarray(image)
        contiguous_background = cp.ascontiguousarray(background)
        contiguous_mask = cp.ascontiguousarray(mask)
        contiguous_eroded = cp.ascontiguousarray(eroded)
        contiguous_distance = cp.ascontiguousarray(distance, dtype=cp.float32)
        output = cp.empty_like(contiguous_image)
        height, width = image.shape[:2]
        pixel_count = int(height * width)
        threads = 256
        self._feather_blend_kernel(
            ((pixel_count + threads - 1) // threads,),
            (threads,),
            (
                contiguous_image,
                contiguous_background,
                contiguous_mask,
                contiguous_eroded,
                contiguous_distance,
                np.int32(height),
                np.int32(width),
                np.float32(transition),
                output,
            ),
        )
        return output

    def blend(
        self,
        image1: cp.ndarray,
        image2: cp.ndarray,
        blending_factor: float = 0.5,
        *,
        foreground_alpha: cp.ndarray | None = None,
    ) -> cp.ndarray:
        contiguous_image1 = cp.ascontiguousarray(image1)
        contiguous_image2 = cp.ascontiguousarray(image2)
        output = cp.empty_like(contiguous_image1)
        pixel_count = int(image1.shape[0] * image1.shape[1])
        alpha = (
            contiguous_image1
            if foreground_alpha is None
            else cp.ascontiguousarray(foreground_alpha, dtype=cp.float32)
        )
        threads = 256
        self._blend_kernel(
            ((pixel_count + threads - 1) // threads,),
            (threads,),
            (
                contiguous_image1,
                contiguous_image2,
                alpha,
                np.int64(pixel_count),
                np.float32(blending_factor),
                np.int32(foreground_alpha is not None),
                output,
            ),
        )
        return output

    def warp_triangles(
        self,
        image: cp.ndarray,
        source_points: np.ndarray,
        target_points: np.ndarray,
        triangles: Sequence[tuple[int, int, int]],
    ) -> cp.ndarray:
        height, width = image.shape[:2]
        triangle_count = len(triangles)
        if triangle_count == 0:
            return cp.zeros_like(image, order="C")

        indices = np.asarray(triangles, dtype=np.intp)
        source_vertices = np.ascontiguousarray(
            source_points[indices],
            dtype=np.float32,
        )
        target_vertices = np.ascontiguousarray(
            target_points[indices],
            dtype=np.float32,
        )
        edge1 = target_vertices[:, 1] - target_vertices[:, 0]
        edge2 = target_vertices[:, 2] - target_vertices[:, 0]
        signed_areas = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        reversed_triangles = signed_areas < 0.0
        source_vertices[reversed_triangles, 1:3] = source_vertices[
            reversed_triangles, 2:0:-1
        ]
        target_vertices[reversed_triangles, 1:3] = target_vertices[
            reversed_triangles, 2:0:-1
        ]

        descriptors = np.empty((triangle_count, 16), dtype=np.float32)
        descriptors[:, :6] = target_vertices.reshape(triangle_count, 6)
        for index in range(triangle_count):
            descriptors[index, 6:12] = cv2.getAffineTransform(
                target_vertices[index],
                source_vertices[index],
            ).reshape(6)

        lower = np.floor(target_vertices.min(axis=1)).astype(np.int64)
        upper = np.ceil(target_vertices.max(axis=1)).astype(np.int64)
        lower[:, 0] = np.maximum(lower[:, 0], 0)
        lower[:, 1] = np.maximum(lower[:, 1], 0)
        upper[:, 0] = np.minimum(upper[:, 0], width - 1)
        upper[:, 1] = np.minimum(upper[:, 1], height - 1)
        box_sizes = np.maximum(upper - lower + 1, 0)
        degenerate = np.abs(signed_areas) <= np.finfo(np.float32).eps
        box_sizes[degenerate] = 0
        descriptors[:, 12:14] = lower
        descriptors[:, 14:16] = box_sizes

        work_sizes = box_sizes[:, 0] * box_sizes[:, 1]
        offsets = np.empty(triangle_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(work_sizes, out=offsets[1:])
        work_count = int(offsets[-1])
        if work_count == 0:
            return cp.zeros_like(image, order="C")

        descriptors_gpu = cp.asarray(descriptors)
        offsets_gpu = cp.asarray(offsets)
        contiguous_image = cp.ascontiguousarray(image)
        threads = 256
        if _triangles_have_interior_overlap(target_vertices):
            membership = cp.full(
                (height, width),
                -1,
                dtype=cp.int32,
            )
            self._triangle_membership_kernel(
                ((work_count + threads - 1) // threads,),
                (threads,),
                (
                    descriptors_gpu,
                    offsets_gpu,
                    np.int32(triangle_count),
                    np.int32(width),
                    np.int64(work_count),
                    membership,
                ),
            )
            output = cp.empty_like(contiguous_image)
            pixel_count = height * width
            self._triangle_sample_kernel(
                ((pixel_count + threads - 1) // threads,),
                (threads,),
                (
                    contiguous_image,
                    descriptors_gpu,
                    membership,
                    np.int32(height),
                    np.int32(width),
                    output,
                ),
            )
            return output

        output = cp.zeros_like(contiguous_image, order="C")
        self._triangle_warp_kernel(
            ((work_count + threads - 1) // threads,),
            (threads,),
            (
                contiguous_image,
                descriptors_gpu,
                offsets_gpu,
                np.int32(triangle_count),
                np.int32(height),
                np.int32(width),
                np.int64(work_count),
                output,
            ),
        )
        return output
