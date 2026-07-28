from __future__ import annotations

import numpy as np
import pytest

from ubo_morph import Landmarks
from ubo_morph.morphing.alignment import alignment_geometry, scale_landmarks


@pytest.fixture
def landmarks() -> Landmarks:
    return Landmarks(
        left_eye=np.array([14.0, 8.0], dtype=np.float32),
        right_eye=np.array([6.0, 8.0], dtype=np.float32),
        points=np.array(
            [[4.0, 4.0], [15.0, 4.0], [15.0, 15.0], [4.0, 15.0]],
            dtype=np.float32,
        ),
    )


def test_alignment_geometry_contains_backend_independent_image_plan(
    landmarks: Landmarks,
) -> None:
    geometry = alignment_geometry((20, 20), landmarks)

    assert geometry.border == (5, 5)
    assert geometry.padded_shape == (30, 30)
    assert geometry.crop_shape == (43, 32)
    assert geometry.rotation_matrix.dtype == np.float32
    assert geometry.landmarks.points.dtype == np.float32


def test_scale_landmarks_uses_xy_scale(landmarks: Landmarks) -> None:
    scaled = scale_landmarks(landmarks, 2.0, 3.0)

    np.testing.assert_array_equal(scaled.left_eye, [28.0, 24.0])
    np.testing.assert_array_equal(scaled.right_eye, [12.0, 24.0])
    np.testing.assert_array_equal(scaled.points[0], [8.0, 12.0])
