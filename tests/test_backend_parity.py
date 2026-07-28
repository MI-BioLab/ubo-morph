from __future__ import annotations

import warnings
from typing import Any

import cv2
import numpy as np
import pytest

from ubo_morph.morphing.cpu import CPUBackend


cupy: Any = None
CuPyBackendType: Any = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    try:
        import cupy as cupy_module

        from ubo_morph.morphing.cupy import CuPyBackend as ImportedCuPyBackend
    except ModuleNotFoundError as error:
        if error.name != "cupy":
            raise
    else:
        cupy = cupy_module
        CuPyBackendType = ImportedCuPyBackend


EXACT_ATOL = 0.0
BLEND_ATOL = 1.0
AFFINE_ATOL = 2.0
RESIZE_ATOL = 4.0
EQUALIZE_LIGHTNESS_ATOL = 4.0
DISTANCE_FIELD_ATOL = 0.5
TRIANGLE_INTERIOR_ATOL = 2.0


def _cuda_unavailable_reason() -> str | None:
    if cupy is None:
        return "CuPy extra is not installed"
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return "No CUDA device is available"
        cupy.zeros(1, dtype=cupy.uint8)
        cupy.cuda.Stream.null.synchronize()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        return f"CUDA is unavailable: {error}"
    return None


CUDA_UNAVAILABLE_REASON = _cuda_unavailable_reason()


def _smooth_image(height: int, width: int) -> np.ndarray:
    rows, columns = np.mgrid[:height, :width]
    return np.stack(
        (
            10 + 2 * columns + rows,
            20 + columns + 3 * rows,
            30 + 3 * columns + 2 * rows,
        ),
        axis=2,
    ).astype(np.uint8)


@pytest.mark.skipif(
    CUDA_UNAVAILABLE_REASON is not None,
    reason=CUDA_UNAVAILABLE_REASON or "CUDA is unavailable",
)
class TestBackendNumericalParity:
    @pytest.fixture(autouse=True)
    def _backends(self) -> None:
        self.cpu = CPUBackend()
        self.cupy = CuPyBackendType()

    def assert_backend_close(
        self,
        operation: str,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        atol: float,
    ) -> None:
        assert actual.shape == expected.shape, operation
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        maximum_difference = float(np.max(difference, initial=0.0))
        assert maximum_difference <= atol, (
            f"{operation}: maximum absolute difference "
            f"{maximum_difference:g}; tolerance {atol:g}"
        )

    def to_cupy(self, array: np.ndarray) -> Any:
        return self.cupy.to_backend(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        return self.cupy.to_numpy(array)

    def test_array_transfer_matches_cpu(self) -> None:
        image = _smooth_image(9, 11)

        expected = self.cpu.to_numpy(self.cpu.to_backend(image))
        actual = self.to_numpy(self.cupy.to_backend(image))

        self.assert_backend_close(
            "array transfer",
            actual,
            expected,
            atol=EXACT_ATOL,
        )

    def test_padding_matches_cpu(self) -> None:
        image = _smooth_image(9, 11)
        padding = (2, 3, 4, 1)

        expected = self.cpu.pad_image(image=image, padding=padding)
        actual = self.to_numpy(
            self.cupy.pad_image(image=self.to_cupy(image), padding=padding)
        )

        self.assert_backend_close("padding", actual, expected, atol=EXACT_ATOL)

    def test_affine_warp_matches_cpu_within_tolerance(self) -> None:
        image = _smooth_image(24, 28)
        matrix = cv2.getRotationMatrix2D((14.0, 12.0), 7.5, 0.96).astype(np.float32)
        output_shape = image.shape[:2]

        expected = self.cpu.warp_affine(
            image=image,
            matrix=matrix,
            output_shape=output_shape,
        )
        actual = self.to_numpy(
            self.cupy.warp_affine(
                image=self.to_cupy(image),
                matrix=matrix,
                output_shape=output_shape,
            )
        )

        self.assert_backend_close(
            "affine warp",
            actual,
            expected,
            atol=AFFINE_ATOL,
        )

    def test_affine_warp_uses_constant_zero_at_fractional_border(self) -> None:
        image = np.full((2, 4, 3), 200, dtype=np.uint8)
        matrix = np.array(
            [[1.0, 0.0, -0.75], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        expected = self.cpu.warp_affine(image, matrix, image.shape[:2])
        actual = self.to_numpy(
            self.cupy.warp_affine(
                self.to_cupy(image),
                matrix,
                image.shape[:2],
            )
        )

        np.testing.assert_array_equal(actual, expected)

    def test_resize_matches_cpu_within_tolerance(self) -> None:
        image = _smooth_image(12, 15)
        output_shape = (19, 23)

        expected = self.cpu.resize_image(image=image, output_shape=output_shape)
        actual = self.to_numpy(
            self.cupy.resize_image(
                image=self.to_cupy(image),
                output_shape=output_shape,
            )
        )

        self.assert_backend_close(
            "resize",
            actual,
            expected,
            atol=RESIZE_ATOL,
        )

    def test_area_downscale_preserves_high_frequency_average(self) -> None:
        checkerboard = (
            (np.indices((64, 64)).sum(axis=0) % 2) * 255
        ).astype(np.uint8)
        image = np.repeat(checkerboard[:, :, None], 3, axis=2)
        output_shape = (17, 19)

        expected = self.cpu.resize_image(image, output_shape)
        actual = self.to_numpy(
            self.cupy.resize_image(self.to_cupy(image), output_shape)
        )

        self.assert_backend_close(
            "area downscale",
            actual,
            expected,
            atol=1.0,
        )

    def test_histogram_matching_matches_cpu(self) -> None:
        generator = np.random.default_rng(4273)
        channel = generator.integers(0, 256, (18, 20), dtype=np.uint8)
        reference_channel = generator.integers(0, 256, (18, 20), dtype=np.uint8)
        rows, columns = np.mgrid[:18, :20]
        mask = (rows + columns) % 3 != 0
        reference_mask = (2 * rows + columns) % 4 != 0

        expected = self.cpu.match_histogram_channel(
            channel=channel,
            mask=mask,
            reference_channel=reference_channel,
            reference_mask=reference_mask,
        )
        actual = self.to_numpy(
            self.cupy.match_histogram_channel(
                channel=self.to_cupy(channel),
                mask=self.to_cupy(mask),
                reference_channel=self.to_cupy(reference_channel),
                reference_mask=self.to_cupy(reference_mask),
            )
        )

        self.assert_backend_close(
            "histogram matching",
            actual,
            expected,
            atol=EXACT_ATOL,
        )

    def test_histogram_matching_accepts_different_reference_shape(self) -> None:
        generator = np.random.default_rng(917)
        channel = generator.integers(0, 256, (6, 7), dtype=np.uint8)
        reference_channel = generator.integers(0, 256, (9, 11), dtype=np.uint8)
        mask = np.ones(channel.shape, dtype=bool)
        reference_mask = np.ones(reference_channel.shape, dtype=bool)

        expected = self.cpu.match_histogram_channel(
            channel,
            mask,
            reference_channel,
            reference_mask,
        )
        actual = self.to_numpy(
            self.cupy.match_histogram_channel(
                self.to_cupy(channel),
                self.to_cupy(mask),
                self.to_cupy(reference_channel),
                self.to_cupy(reference_mask),
            )
        )

        self.assert_backend_close(
            "different-shape histogram matching",
            actual,
            expected,
            atol=EXACT_ATOL,
        )

    def test_convex_hull_mask_differs_only_at_rasterized_boundary(self) -> None:
        points = np.array(
            [[3.4, 2.8], [15.7, 1.2], [21.1, 8.9], [17.5, 17.8], [4.2, 15.6]],
            dtype=np.float32,
        )
        shape = (20, 24)

        expected = self.cpu.convex_hull_mask(shape, points)
        actual = self.to_numpy(self.cupy.convex_hull_mask(shape, points))

        kernel = np.ones((3, 3), dtype=np.uint8)
        expected_interior = cv2.erode(expected.astype(np.uint8), kernel).astype(bool)
        expected_envelope = cv2.dilate(expected.astype(np.uint8), kernel).astype(bool)
        assert np.all(actual[expected_interior])
        assert np.all(expected_envelope[actual])

    def test_lightness_equalization_matches_cpu_within_tolerance(self) -> None:
        image = _smooth_image(18, 20)
        reference = np.clip(image.astype(np.int16) + 35, 0, 255).astype(np.uint8)
        rows, columns = np.mgrid[:18, :20]
        mask = (rows > 1) & (rows < 16) & (columns > 2) & (columns < 18)

        expected = self.cpu.equalize_lightness(
            reference_image=reference,
            image_to_equalize=image,
            reference_mask=mask,
            equalize_mask=mask,
        )
        actual = self.to_numpy(
            self.cupy.equalize_lightness(
                reference_image=self.to_cupy(reference),
                image_to_equalize=self.to_cupy(image),
                reference_mask=self.to_cupy(mask),
                equalize_mask=self.to_cupy(mask),
            )
        )

        self.assert_backend_close(
            "lightness equalization",
            actual,
            expected,
            atol=EQUALIZE_LIGHTNESS_ATOL,
        )

    def test_masked_copy_matches_cpu(self) -> None:
        image = _smooth_image(13, 17)
        background = np.flip(image, axis=1).copy()
        rows, columns = np.mgrid[:13, :17]
        mask = (rows - 6) ** 2 + (columns - 8) ** 2 <= 25

        expected = self.cpu.copy_with_mask(
            image=image,
            background=background,
            mask=mask,
        )
        actual = self.to_numpy(
            self.cupy.copy_with_mask(
                image=self.to_cupy(image),
                background=self.to_cupy(background),
                mask=self.to_cupy(mask),
            )
        )

        self.assert_backend_close(
            "masked copy",
            actual,
            expected,
            atol=EXACT_ATOL,
        )

    def test_feather_fields_match_cpu_within_tolerances(self) -> None:
        mask = np.zeros((31, 35), dtype=bool)
        cv2.ellipse(mask, (17, 15), (11, 9), 0, 0, 360, True, -1)
        element_size = 5

        expected_eroded, expected_distance = self.cpu.feather_fields(
            mask=mask,
            element_size=element_size,
        )
        actual_eroded, actual_distance = self.cupy.feather_fields(
            mask=self.to_cupy(mask),
            element_size=element_size,
        )

        self.assert_backend_close(
            "feather erosion",
            self.to_numpy(actual_eroded),
            expected_eroded,
            atol=EXACT_ATOL,
        )
        self.assert_backend_close(
            "feather distance field",
            self.to_numpy(actual_distance),
            expected_distance,
            atol=DISTANCE_FIELD_ATOL,
        )

    @pytest.mark.parametrize("mode", ["scalar", "alpha"])
    def test_blending_matches_cpu_within_tolerance(self, mode: str) -> None:
        image1 = _smooth_image(12, 14)
        image2 = np.flip(image1, axis=(0, 1)).copy()

        if mode == "scalar":
            expected = self.cpu.blend(
                image1=image1,
                image2=image2,
                blending_factor=0.37,
            )
            actual = self.to_numpy(
                self.cupy.blend(
                    image1=self.to_cupy(image1),
                    image2=self.to_cupy(image2),
                    blending_factor=0.37,
                )
            )
            self.assert_backend_close(
                "scalar blend",
                actual,
                expected,
                atol=BLEND_ATOL,
            )
        else:
            foreground_alpha = np.linspace(
                0.0,
                1.0,
                image1.shape[0] * image1.shape[1],
                dtype=np.float32,
            ).reshape(image1.shape[:2])
            expected = self.cpu.blend(
                image1=image1,
                image2=image2,
                foreground_alpha=foreground_alpha,
            )
            actual = self.to_numpy(
                self.cupy.blend(
                    image1=self.to_cupy(image1),
                    image2=self.to_cupy(image2),
                    foreground_alpha=self.to_cupy(foreground_alpha),
                )
            )
            self.assert_backend_close(
                "alpha blend",
                actual,
                expected,
                atol=BLEND_ATOL,
            )

    def test_triangle_warp_interior_matches_cpu_within_tolerance(self) -> None:
        image = _smooth_image(32, 32)
        source_points = np.array(
            [[5.0, 5.0], [26.0, 6.0], [8.0, 27.0]],
            dtype=np.float32,
        )
        target_points = np.array(
            [[6.0, 4.0], [25.0, 8.0], [10.0, 26.0]],
            dtype=np.float32,
        )
        triangles = [(0, 1, 2)]

        expected = self.cpu.warp_triangles(
            image=image,
            source_points=source_points,
            target_points=target_points,
            triangles=triangles,
        )
        actual = self.to_numpy(
            self.cupy.warp_triangles(
                image=self.to_cupy(image),
                source_points=source_points,
                target_points=target_points,
                triangles=triangles,
            )
        )
        target_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(target_mask, target_points.astype(np.int32), 1)
        shared_interior = cv2.erode(
            target_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)

        self.assert_backend_close(
            "triangle warp shared interior",
            actual[shared_interior],
            expected[shared_interior],
            atol=TRIANGLE_INTERIOR_ATOL,
        )

    def test_triangle_warp_identity_has_no_boundary_or_shared_edge_gaps(self) -> None:
        image = np.full((8, 9, 3), 137, dtype=np.uint8)
        points = np.array(
            [[0.0, 0.0], [8.0, 0.0], [0.0, 7.0], [8.0, 7.0]],
            dtype=np.float32,
        )

        actual = self.to_numpy(
            self.cupy.warp_triangles(
                image=self.to_cupy(image),
                source_points=points,
                target_points=points,
                triangles=[(0, 1, 2), (1, 3, 2)],
            )
        )

        np.testing.assert_array_equal(actual, image)

    def test_overlapping_triangle_warp_is_deterministic_and_last_wins(self) -> None:
        image = _smooth_image(16, 16)
        source_points = np.array(
            [[1, 1], [7, 1], [1, 7], [8, 8], [14, 8], [8, 14]],
            dtype=np.float32,
        )
        target_points = np.array(
            [[1, 1], [7, 1], [1, 7], [1, 1], [7, 1], [1, 7]],
            dtype=np.float32,
        )
        triangles = [(0, 1, 2), (3, 4, 5)]

        results = [
            self.to_numpy(
                self.cupy.warp_triangles(
                    self.to_cupy(image),
                    source_points,
                    target_points,
                    triangles,
                )
            )
            for _ in range(5)
        ]

        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])
        np.testing.assert_array_equal(results[0][2, 2], image[9, 9])
