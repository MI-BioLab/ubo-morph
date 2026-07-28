from __future__ import annotations

import numpy as np

from ubo_morph.morphing.cpu import CPUBackend


def test_cpu_backend_matches_each_channel_independently() -> None:
    image = np.full((8, 8, 3), (20, 40, 60), dtype=np.uint8)
    reference = np.full((8, 8, 3), (80, 120, 200), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    backend = CPUBackend()
    result = image.copy()

    for channel_index in range(3):
        result[:, :, channel_index] = backend.match_histogram_channel(
            image[:, :, channel_index],
            mask,
            reference[:, :, channel_index],
            mask,
        )

    np.testing.assert_array_equal(result, reference)
