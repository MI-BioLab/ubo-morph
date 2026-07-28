from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ubo_morph.morphing.core import _delaunay_triangles, _substitute_background
from ubo_morph.morphing.points import convex_hull_mask
from ubo_morph.morphing.cpu import CPUBackend


def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cupy_extra_includes_raw_kernel_toolkit() -> None:
    configuration = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert any(
        requirement.startswith("cupy-cuda12x[ctk]")
        for requirement in configuration["project"]["optional-dependencies"]["cupy"]
    )


def test_cpu_package_import_does_not_load_cupy() -> None:
    result = _run_fresh_python(
        "import sys\n"
        "import ubo_morph\n"
        "assert not any(name == 'cupy' or name.startswith(('cupy.', 'cupyx.')) "
        "for name in sys.modules), sorted(name for name in sys.modules "
        "if name.startswith(('cupy', 'cupyx')))\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_cli_import_does_not_require_pandas() -> None:
    result = _run_fresh_python(
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'pandas' or name.startswith('pandas.'):\n"
        "        raise AssertionError('pandas imported')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n"
        "import ubo_morph.cli\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout


class TestNativePrimitives:
    def test_cpu_blend_accepts_keywords_and_implicit_background_alpha(self) -> None:
        image = np.full((2, 2, 3), 200, dtype=np.uint8)
        background = np.full((2, 2, 3), 20, dtype=np.uint8)
        foreground_alpha = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32)

        result = CPUBackend().blend(
            image1=image,
            image2=background,
            foreground_alpha=foreground_alpha,
        )

        expected = np.array([[20, 65], [155, 200]], dtype=np.uint8)
        np.testing.assert_array_equal(result[:, :, 0], expected)

    def test_degenerate_convex_hull_skips_opencv_hull_construction(self) -> None:
        with patch(
            "ubo_morph.morphing.points.cv2.convexHull",
            side_effect=AssertionError("degenerate hull passed to OpenCV"),
        ):
            mask = convex_hull_mask((5, 6), np.array([[2.0, 3.0]], dtype=np.float32))

        assert mask.shape == (5, 6)
        assert mask.dtype == np.bool_
        assert not mask.any()

    def test_collinear_convex_hull_is_empty(self) -> None:
        points = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

        with patch(
            "ubo_morph.morphing.points.cv2.fillConvexPoly",
            side_effect=AssertionError("degenerate hull filled"),
        ):
            mask = convex_hull_mask((5, 5), points)

        assert not mask.any()

    def test_triangulation_uses_subdivision_vertex_lookup(self) -> None:
        points = np.array(
            [[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [0.0, 9.0], [4.0, 4.0]],
            dtype=np.float32,
        )

        with patch(
            "ubo_morph.morphing.core.np.linalg.norm",
            side_effect=AssertionError("quadratic nearest-point scan used"),
        ):
            triangles = _delaunay_triangles(points)

        assert len(triangles) >= 2

    def test_binary_background_substitution_uses_copy_to(self) -> None:
        image = np.full((8, 8, 3), 200, dtype=np.uint8)
        background = np.full((8, 8, 3), 20, dtype=np.uint8)
        points = np.array([[2, 2], [5, 2], [5, 5], [2, 5]], dtype=np.float32)
        mask = convex_hull_mask(image.shape[:2], points)

        with patch(
            "ubo_morph.morphing.cpu.backend.cv2.copyTo",
            wraps=cv2.copyTo,
        ) as copy_to:
            result = CPUBackend().copy_with_mask(
                image,
                background,
                mask,
            )

        copy_to.assert_called_once()
        np.testing.assert_array_equal(result[3, 3], image[3, 3])
        np.testing.assert_array_equal(result[0, 0], background[0, 0])

    def test_feathered_background_substitution_uses_blend_linear(self) -> None:
        image = np.full((16, 16, 3), 200, dtype=np.uint8)
        background = np.full((16, 16, 3), 20, dtype=np.uint8)
        points = np.array([[2, 2], [13, 2], [13, 13], [2, 13]], dtype=np.float32)
        backend = CPUBackend()

        with patch(
            "ubo_morph.morphing.cpu.backend.cv2.blendLinear",
            wraps=cv2.blendLinear,
        ) as blend_linear:
            result = _substitute_background(
                backend,
                image,
                points,
                background,
                blend=True,
                eye_distance=8.0,
            )

        blend_linear.assert_called_once()
        assert result.dtype == np.uint8

    def test_triangle_warp_uses_blend_linear(self) -> None:
        image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
        source = np.array([[1, 1], [8, 1], [1, 8]], dtype=np.float32)
        target = np.array([[1, 1], [8, 2], [2, 8]], dtype=np.float32)

        with patch(
            "ubo_morph.morphing.cpu.backend.cv2.blendLinear",
            wraps=cv2.blendLinear,
        ) as blend_linear:
            result = CPUBackend().warp_triangles(
                image,
                source,
                target,
                [(0, 1, 2)],
            )

        blend_linear.assert_called_once()
        assert result.shape == image.shape
