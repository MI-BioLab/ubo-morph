from __future__ import annotations

import unittest

import numpy as np

from ubo_morph import equalize_face


class HistogramMatchingTests(unittest.TestCase):
    def test_color_method_matches_each_channel_independently(self) -> None:
        points = np.array(
            [[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]],
            dtype=np.float32,
        )
        image = np.full((8, 8, 3), (20, 40, 60), dtype=np.uint8)
        reference = np.full((8, 8, 3), (80, 120, 200), dtype=np.uint8)

        result = equalize_face(
            reference,
            image,
            points,
            points,
            method="color",
            device="cpu",
        )

        np.testing.assert_array_equal(result, reference)


if __name__ == "__main__":
    unittest.main()
