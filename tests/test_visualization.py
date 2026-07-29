from __future__ import annotations

import numpy as np
import pytest

from ubo_morph.visualization import annotate_landmark_mesh


def test_landmark_label_is_kept_inside_image_at_top_right_corner() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    annotated = annotate_landmark_mesh(
        image,
        mesh_points=np.array([[119.0, 0.0]], dtype=np.float32),
        triangles=[],
        landmark_points=np.array([[119.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(image, np.zeros_like(image))
    grayscale_label_pixels = (
        (annotated[:, :, 0] == annotated[:, :, 1])
        & (annotated[:, :, 1] == annotated[:, :, 2])
        & (annotated[:, :, 0] > 0)
    )
    assert grayscale_label_pixels.any()


def test_large_mesh_uses_compact_strokes() -> None:
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    annotated = annotate_landmark_mesh(
        image,
        mesh_points=np.array(
            [[100.0, 100.0], [900.0, 100.0], [500.0, 900.0]],
            dtype=np.float32,
        ),
        triangles=[(0, 1, 2)],
        landmark_points=np.empty((0, 2), dtype=np.float32),
    )

    assert np.any(annotated != 0, axis=2).sum() < 9000


def test_large_landmark_point_is_compact() -> None:
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    annotated = annotate_landmark_mesh(
        image,
        mesh_points=np.empty((0, 2), dtype=np.float32),
        triangles=[],
        landmark_points=np.array([[500.0, 500.0]], dtype=np.float32),
    )

    green_pixels = (
        (annotated[:, :, 1] > annotated[:, :, 0])
        & (annotated[:, :, 1] > annotated[:, :, 2])
    )
    assert green_pixels.sum() < 60


def test_large_landmark_font_is_compact() -> None:
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    annotated = annotate_landmark_mesh(
        image,
        mesh_points=np.empty((0, 2), dtype=np.float32),
        triangles=[],
        landmark_points=np.array([[500.0, 500.0]], dtype=np.float32),
    )

    assert np.all(annotated == 255, axis=2).sum() < 60


@pytest.mark.parametrize(
    ("size", "maximum_label_pixels"),
    [(250, 20), (500, 36), (1000, 65)],
)
def test_landmark_font_stays_compact_across_resolutions(
    size: int,
    maximum_label_pixels: int,
) -> None:
    image = np.zeros((size, size, 3), dtype=np.uint8)

    annotated = annotate_landmark_mesh(
        image,
        mesh_points=np.empty((0, 2), dtype=np.float32),
        triangles=[],
        landmark_points=np.array([[size / 2, size / 2]], dtype=np.float32),
    )

    grayscale_label_pixels = (
        (annotated[:, :, 0] == annotated[:, :, 1])
        & (annotated[:, :, 1] == annotated[:, :, 2])
        & (annotated[:, :, 0] > 0)
    )
    assert grayscale_label_pixels.sum() < maximum_label_pixels
