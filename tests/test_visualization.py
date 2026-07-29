from __future__ import annotations

import numpy as np

from ubo_morph.visualization import annotate_landmark_mesh


def test_landmark_label_is_kept_inside_image_at_top_right_corner() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    annotated = annotate_landmark_mesh(
        image,
        points=np.array([[119.0, 0.0]], dtype=np.float32),
        triangles=[],
        landmark_indices=np.array([123], dtype=np.int32),
    )

    np.testing.assert_array_equal(image, np.zeros_like(image))
    assert np.all(annotated == 255, axis=2).any()
