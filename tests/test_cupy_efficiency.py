from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ubo_morph.morphing.core import _substitute_background
from ubo_morph.morphing.cpu import CPUBackend

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    try:
        from ubo_morph.morphing.cupy import CuPyBackend as CuPyBackendType
        from ubo_morph.morphing.cupy import backend as gpu_backend
    except ModuleNotFoundError as error:
        if error.name != "cupy":
            raise
        CuPyBackendType: Any = None


def _backend_without_compiled_kernels() -> Any:
    return object.__new__(CuPyBackendType)


@pytest.mark.skipif(CuPyBackendType is None, reason="CuPy extra is not installed")
class TestCuPyAlignmentEfficiency:
    @pytest.fixture(autouse=True)
    def _image(self) -> None:
        self.image = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)

    def test_resize_uses_one_ndimage_zoom_call(self) -> None:
        def resize(
            array: np.ndarray,
            factors: tuple[float, ...],
            **kwargs: object,
        ) -> np.ndarray:
            assert array.dtype == np.uint8
            assert kwargs["output"] == np.uint8
            return np.zeros(
                tuple(
                    round(size * factor)
                    for size, factor in zip(array.shape, factors)
                ),
                dtype=np.uint8,
            )

        zoom = MagicMock(side_effect=resize)
        fake_cp = SimpleNamespace(
            clip=np.clip,
            uint8=np.uint8,
        )
        with (
            patch.object(gpu_backend, "cp", fake_cp),
            patch.object(gpu_backend.cpnd, "zoom", zoom),
        ):
            resized = _backend_without_compiled_kernels().resize_image(
                self.image[:4, :4],
                (8, 6),
            )

        assert zoom.call_count == 1
        assert resized.shape == (8, 6, 3)

    def test_affine_warp_uses_one_kernel_without_matrix_upload(self) -> None:
        launches: list[tuple[object, ...]] = []

        def warp(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append(arguments)
            np.asarray(arguments[-1]).fill(0)

        fake_cp = SimpleNamespace(
            ascontiguousarray=np.ascontiguousarray,
            empty=np.empty,
            uint8=np.uint8,
        )
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        backend = _backend_without_compiled_kernels()
        backend._affine_warp_kernel = MagicMock(side_effect=warp)
        with (
            patch.object(gpu_backend, "cp", fake_cp),
            patch.object(
                gpu_backend.cpnd,
                "affine_transform",
                side_effect=AssertionError("ndimage affine path used"),
            ),
        ):
            result = backend.warp_affine(
                self.image,
                matrix,
                (20, 20),
            )

        assert len(launches) == 1
        assert all(np.isscalar(value) for value in launches[0][1:-1])
        assert result.shape == self.image.shape


@pytest.mark.skipif(CuPyBackendType is None, reason="CuPy extra is not installed")
class TestCuPyRetouchingEfficiency:
    def test_convex_hull_mask_uploads_geometry_instead_of_full_mask(self) -> None:
        points = np.array(
            [[10.0, 12.0], [90.0, 10.0], [100.0, 70.0], [15.0, 75.0]],
            dtype=np.float32,
        )
        uploads: list[np.ndarray] = []
        launches: list[tuple[object, ...]] = []

        def upload(value: object) -> np.ndarray:
            result = np.asarray(value)
            uploads.append(result)
            return result

        def rasterize(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append(arguments)
            np.asarray(arguments[-1]).fill(True)

        fake_cp = SimpleNamespace(
            asarray=upload,
            zeros=np.zeros,
            bool_=np.bool_,
        )
        backend = _backend_without_compiled_kernels()
        backend._convex_hull_kernel = MagicMock(side_effect=rasterize)
        with patch.object(gpu_backend, "cp", fake_cp):
            mask = backend.convex_hull_mask((80, 120), points)

        assert len(launches) == 1
        assert len(uploads) == 2
        assert max(upload.size for upload in uploads) < mask.size

    def test_feather_composite_fuses_alpha_and_blend(self) -> None:
        image = np.full((8, 8, 3), 200, dtype=np.uint8)
        background = np.full((8, 8, 3), 20, dtype=np.uint8)
        mask = np.ones((8, 8), dtype=bool)
        launches: list[tuple[object, ...]] = []

        def composite(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append(arguments)
            np.asarray(arguments[-1]).fill(200)

        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            ascontiguousarray=np.ascontiguousarray,
            bool_=np.bool_,
            empty_like=np.empty_like,
            float32=np.float32,
            zeros=np.zeros,
        )
        fake_cpnd = SimpleNamespace(
            binary_erosion=lambda value, structure: value.copy(),
            distance_transform_edt=lambda value, **kwargs: np.ones(
                value.shape,
                dtype=np.float32,
            ),
        )
        backend = _backend_without_compiled_kernels()
        backend._feather_blend_kernel = MagicMock(side_effect=composite)

        def rasterize(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            np.asarray(arguments[-1]).fill(True)

        backend._convex_hull_kernel = MagicMock(side_effect=rasterize)
        with (
            patch.object(gpu_backend, "cp", fake_cp),
            patch.object(gpu_backend, "cpnd", fake_cpnd),
        ):
            result = backend.blend_with_feather(
                image,
                background,
                mask,
                element_size=3,
                transition=1,
            )

        assert len(launches) == 1
        assert result.dtype == np.uint8

    def test_color_histogram_matching_batches_all_channels(self) -> None:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        reference = np.flip(image, axis=0).copy()
        mask = np.ones(image.shape[:2], dtype=bool)
        launches: list[str] = []
        histogram_launches: list[tuple[object, dict[str, object]]] = []

        def histogram(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
            **kwargs: object,
        ) -> None:
            del block, arguments
            launches.append("histogram")
            histogram_launches.append((grid, kwargs))

        def lookup(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append("lookup")
            np.asarray(arguments[-1])[:] = np.arange(256, dtype=np.uint8)

        def apply(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append("apply")
            np.asarray(arguments[-1])[:] = np.asarray(arguments[0])

        fake_cp = SimpleNamespace(
            ascontiguousarray=np.ascontiguousarray,
            empty=np.empty,
            empty_like=np.empty_like,
            zeros=np.zeros,
            uint8=np.uint8,
            uint32=np.uint32,
        )
        backend = _backend_without_compiled_kernels()
        backend._histogram_kernel = MagicMock(side_effect=histogram)
        backend._lookup_kernel = MagicMock(side_effect=lookup)
        backend._apply_lookup_kernel = MagicMock(side_effect=apply)
        with patch.object(gpu_backend, "cp", fake_cp):
            result = backend.match_histogram_image(
                image,
                mask,
                reference,
                mask,
            )

        assert launches == ["histogram", "lookup", "apply"]
        assert histogram_launches == [((1, 3), {})]
        assert result.shape == image.shape

    @pytest.mark.parametrize(
        ("method_name", "kernel_name"),
        [
            ("_bgr_to_hls", "_bgr_to_hls_kernel"),
            ("_hls_to_bgr", "_hls_to_bgr_kernel"),
        ],
    )
    def test_color_conversion_uses_one_fused_kernel(
        self,
        method_name: str,
        kernel_name: str,
    ) -> None:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        launches: list[tuple[object, ...]] = []

        def convert(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append(arguments)
            output = arguments[-1]
            assert isinstance(output, np.ndarray)
            output.fill(0)

        fake_cp = SimpleNamespace(
            ascontiguousarray=np.ascontiguousarray,
            empty_like=np.empty_like,
        )
        backend = _backend_without_compiled_kernels()
        setattr(backend, kernel_name, MagicMock(side_effect=convert))
        with patch.object(gpu_backend, "cp", fake_cp):
            result = getattr(backend, method_name)(image)

        assert len(launches) == 1
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("uses_alpha", [False, True])
    def test_blend_uses_one_fused_uint8_kernel(self, uses_alpha: bool) -> None:
        image1 = np.full((3, 4, 3), 200, dtype=np.uint8)
        image2 = np.full((3, 4, 3), 20, dtype=np.uint8)
        alpha = (
            np.full((3, 4), 0.25, dtype=np.float32) if uses_alpha else None
        )
        launches: list[tuple[object, ...]] = []

        def blend(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append(arguments)
            output = arguments[-1]
            assert isinstance(output, np.ndarray)
            output.fill(65)

        fake_cp = SimpleNamespace(
            ascontiguousarray=np.ascontiguousarray,
            empty_like=np.empty_like,
            float32=np.float32,
        )
        backend = _backend_without_compiled_kernels()
        backend._blend_kernel = MagicMock(side_effect=blend)
        with patch.object(gpu_backend, "cp", fake_cp):
            result = backend.blend(
                image1,
                image2,
                blending_factor=0.25,
                foreground_alpha=alpha,
            )

        assert len(launches) == 1
        assert np.asarray(launches[0][0]).dtype == np.uint8
        assert np.asarray(launches[0][1]).dtype == np.uint8
        assert result.dtype == np.uint8

    def test_background_distance_transform_stays_float32(self) -> None:
        image = np.full((8, 8, 3), 200, dtype=np.uint8)
        background = np.full((8, 8, 3), 20, dtype=np.uint8)
        points = np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=np.float32)
        distance_transform = MagicMock(return_value=np.ones((8, 8), dtype=np.float32))
        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            ascontiguousarray=np.ascontiguousarray,
            bool_=np.bool_,
            empty_like=np.empty_like,
            float32=np.float32,
            zeros=np.zeros,
        )
        fake_cpnd = SimpleNamespace(
            binary_erosion=lambda mask, structure: np.zeros_like(mask, dtype=bool),
            distance_transform_edt=distance_transform,
        )

        backend = _backend_without_compiled_kernels()

        def composite(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            np.asarray(arguments[-1]).fill(0)

        backend._feather_blend_kernel = MagicMock(side_effect=composite)

        def rasterize(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            np.asarray(arguments[-1]).fill(True)

        backend._convex_hull_kernel = MagicMock(side_effect=rasterize)
        with (
            patch.object(gpu_backend, "cp", fake_cp),
            patch.object(gpu_backend, "cpnd", fake_cpnd),
        ):
            result = _substitute_background(
                backend,
                image,
                points,
                background,
                blend=True,
                eye_distance=8.0,
            )

        assert distance_transform.call_args.args[0].dtype == np.bool_
        assert distance_transform.call_args.kwargs.get("float64_distances") is False
        assert result.dtype == np.uint8

    def test_lightness_equalization_stays_on_backend(self) -> None:
        class NumpyCuPy:
            def __init__(self) -> None:
                self.asnumpy = MagicMock(
                    side_effect=AssertionError("equalization left the backend")
                )

            def __getattr__(self, name: str) -> Any:
                return getattr(np, name)

        image = np.array(
            [
                [[10, 30, 90], [30, 80, 20], [100, 40, 15], [50, 50, 50]],
                [[20, 60, 140], [80, 120, 30], [160, 70, 25], [90, 90, 90]],
                [[40, 90, 180], [100, 160, 50], [200, 100, 40], [130, 130, 130]],
                [[70, 120, 220], [140, 200, 80], [230, 150, 70], [180, 180, 180]],
            ],
            dtype=np.uint8,
        )
        reference = np.flip(image, axis=0).copy()
        mask = np.ones(image.shape[:2], dtype=bool)
        expected = CPUBackend().equalize_lightness(reference, image, mask, mask)
        fake_cp = NumpyCuPy()
        backend = _backend_without_compiled_kernels()

        def convert(code: int) -> Any:
            def kernel(
                grid: object,
                block: object,
                arguments: tuple[object, ...],
            ) -> None:
                del grid, block
                source, _, output = arguments
                np.asarray(output)[:] = cv2.cvtColor(np.asarray(source), code)

            return MagicMock(side_effect=kernel)

        backend._bgr_to_hls_kernel = convert(cv2.COLOR_BGR2HLS)
        backend._hls_to_bgr_kernel = convert(cv2.COLOR_HLS2BGR)

        def lookup(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            np.asarray(arguments[-1])[:] = np.arange(256, dtype=np.uint8)

        def apply(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            np.asarray(arguments[-1])[:] = np.asarray(arguments[0])

        backend._histogram_kernel = MagicMock()
        backend._lookup_kernel = MagicMock(side_effect=lookup)
        backend._apply_lookup_kernel = MagicMock(side_effect=apply)

        with patch.object(gpu_backend, "cp", fake_cp):
            result = backend.equalize_lightness(
                reference,
                image,
                mask,
                mask,
            )

        fake_cp.asnumpy.assert_not_called()
        assert result.dtype == np.uint8
        np.testing.assert_allclose(result, expected, atol=4)


@pytest.mark.skipif(CuPyBackendType is None, reason="CuPy extra is not installed")
class TestCuPyWarpingEfficiency:
    def test_overlapping_triangles_use_deterministic_membership_fallback(
        self,
    ) -> None:
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        source_points = np.array(
            [[0, 0], [7, 0], [0, 7], [7, 7], [0, 7], [7, 0]],
            dtype=np.float32,
        )
        target_points = np.array(
            [[0, 0], [7, 0], [0, 7], [0, 0], [7, 0], [0, 7]],
            dtype=np.float32,
        )
        launches: list[str] = []

        def membership(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append("membership")
            np.asarray(arguments[-1]).fill(1)

        def sample(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            launches.append("sample")
            np.asarray(arguments[-1]).fill(0)

        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            ascontiguousarray=np.ascontiguousarray,
            empty_like=np.empty_like,
            full=np.full,
            int32=np.int32,
            zeros_like=np.zeros_like,
        )
        backend = _backend_without_compiled_kernels()
        backend._triangle_warp_kernel = MagicMock(
            side_effect=AssertionError("racy direct kernel used")
        )
        backend._triangle_membership_kernel = MagicMock(side_effect=membership)
        backend._triangle_sample_kernel = MagicMock(side_effect=sample)
        with patch.object(gpu_backend, "cp", fake_cp):
            result = backend.warp_triangles(
                image,
                source_points,
                target_points,
                [(0, 1, 2), (3, 4, 5)],
            )

        assert launches == ["membership", "sample"]
        assert result.shape == image.shape

    def test_triangle_warp_uses_one_direct_bounding_box_kernel(self) -> None:
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        points = np.array([[1, 1], [6, 1], [1, 6]], dtype=np.float32)
        launches: list[tuple[object, object, tuple[object, ...]]] = []

        def warp(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            launches.append((grid, block, arguments))
            output = arguments[-1]
            assert isinstance(output, np.ndarray)
            output.fill(0)

        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            ascontiguousarray=np.ascontiguousarray,
            zeros_like=np.zeros_like,
            float32=np.float32,
            int64=np.int64,
            uint8=np.uint8,
        )
        backend = _backend_without_compiled_kernels()
        backend._triangle_warp_kernel = MagicMock(side_effect=warp)
        with patch.object(gpu_backend, "cp", fake_cp):
            result = backend.warp_triangles(
                image,
                points,
                points,
                [(0, 1, 2)],
            )

        assert len(launches) == 1
        _, _, arguments = launches[0]
        assert np.asarray(arguments[0]).dtype == np.uint8
        assert np.asarray(arguments[1]).shape == (1, 16)
        assert np.asarray(arguments[2]).shape == (2,)
        assert np.asarray(arguments[-1]).dtype == np.uint8
        assert result.shape == image.shape

    def test_triangle_warp_clips_work_to_the_image(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        points = np.array([[-10, -20], [20, -20], [-10, 30]], dtype=np.float32)
        captured_offsets: list[np.ndarray] = []

        def warp(
            grid: object,
            block: object,
            arguments: tuple[object, ...],
        ) -> None:
            del grid, block
            captured_offsets.append(np.asarray(arguments[2]))

        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            ascontiguousarray=np.ascontiguousarray,
            zeros_like=np.zeros_like,
            float32=np.float32,
            int64=np.int64,
            uint8=np.uint8,
        )
        backend = _backend_without_compiled_kernels()
        backend._triangle_warp_kernel = MagicMock(side_effect=warp)
        with patch.object(gpu_backend, "cp", fake_cp):
            backend.warp_triangles(image, points, points, [(0, 1, 2)])

        np.testing.assert_array_equal(captured_offsets[0], np.array([0, 20]))
