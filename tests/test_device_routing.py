from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from ubo_morph import (
    Landmarks,
    MorphResult,
    blend_images,
    delaunay_triangles,
    morph_with_landmarks,
)


class DeviceRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
        self.points = np.array(
            [
                [0.0, 0.0],
                [19.0, 0.0],
                [19.0, 19.0],
                [0.0, 19.0],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        )
        self.landmarks = Landmarks(
            left_eye=np.array([14.0, 8.0], dtype=np.float32),
            right_eye=np.array([6.0, 8.0], dtype=np.float32),
            points=self.points,
        )

    def test_dispatchers_import_backends_lazily(self) -> None:
        script = """
import sys
import numpy as np
from ubo_morph import blend_images, delaunay_triangles

morph_cpu_package = "ubo_morph.morphing.cpu"
morph_gpu_package = "ubo_morph.morphing.gpu"
triangulation_cpu_module = "ubo_morph.morphing.cpu.triangulation"
triangulation_gpu_module = "ubo_morph.morphing.gpu.triangulation"
assert morph_cpu_package not in sys.modules
assert morph_gpu_package not in sys.modules
assert triangulation_cpu_module not in sys.modules
assert triangulation_gpu_module not in sys.modules
assert "numba" not in sys.modules
assert "dlib" not in sys.modules
assert "mediapipe" not in sys.modules
image = np.zeros((2, 2, 3), dtype=np.uint8)
blend_images(image, image, 0.5, device="cpu")
assert morph_cpu_package in sys.modules
assert f"{morph_cpu_package}.blending" in sys.modules
assert morph_gpu_package not in sys.modules
assert triangulation_cpu_module in sys.modules
points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
delaunay_triangles(points, device="cpu")
assert triangulation_cpu_module in sys.modules
assert triangulation_gpu_module not in sys.modules
assert "numba" not in sys.modules
assert "dlib" not in sys.modules
assert "mediapipe" not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cpu_triangulation_returns_point_indices(self) -> None:
        triangles = delaunay_triangles(self.points, device="cpu")

        self.assertGreaterEqual(len(triangles), 2)
        self.assertTrue(
            all(
                len(set(triangle)) == 3
                and all(0 <= index < len(self.points) for index in triangle)
                for triangle in triangles
            )
        )

    def test_cpu_blending_uses_requested_factor(self) -> None:
        black = np.zeros((2, 2, 3), dtype=np.uint8)
        white = np.full((2, 2, 3), 200, dtype=np.uint8)

        result = blend_images(black, white, 0.25, device="cpu")

        np.testing.assert_array_equal(result, np.full_like(result, 50))

    def test_high_level_api_forwards_cpu_device(self) -> None:
        result = morph_with_landmarks(
            self.image,
            self.image,
            self.landmarks,
            self.landmarks,
            align_eye_centers=False,
            add_border_points=False,
            automatic_retouching=False,
            device="cpu",
            return_details=True,
        )

        self.assertIsInstance(result, MorphResult)
        self.assertEqual(result.image.shape, self.image.shape)
        self.assertEqual(result.image.dtype, np.uint8)
        np.testing.assert_allclose(result.morphed_points, self.points)

    def test_invalid_device_is_rejected_at_public_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            morph_with_landmarks(
                self.image,
                self.image,
                self.landmarks,
                self.landmarks,
                device="cuda",
            )

    def test_gpu_route_is_an_explicit_placeholder(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "GPU"):
            morph_with_landmarks(
                self.image,
                self.image,
                self.landmarks,
                self.landmarks,
                align_eye_centers=False,
                add_border_points=False,
                automatic_retouching=False,
                device="gpu",
            )

    def test_gpu_triangulation_route_is_an_explicit_placeholder(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "GPU"):
            delaunay_triangles(self.points, device="gpu")


if __name__ == "__main__":
    unittest.main()
