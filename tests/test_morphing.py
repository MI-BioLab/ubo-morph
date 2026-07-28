from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import numpy as np
import pytest

from ubo_morph import (
    LandmarkExtractor,
    Landmarks,
    MorphResult,
    morph_images,
    morph_with_landmarks,
)
from ubo_morph.morphing.cpu import CPUBackend


class TestMorphing:
    @pytest.fixture(autouse=True)
    def _image_and_landmarks(self) -> None:
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

    def test_blending_uses_requested_factor(self) -> None:
        black = np.zeros((2, 2, 3), dtype=np.uint8)
        white = np.full((2, 2, 3), 200, dtype=np.uint8)

        result = CPUBackend().blend(black, white, 0.25)

        np.testing.assert_array_equal(result, np.full_like(result, 50))

    def test_short_side_landmark_limit_keeps_full_size_morph_inputs(self) -> None:
        full_points = self.points
        full_height, full_width = self.image.shape[:2]

        class RecordingExtractor(LandmarkExtractor):
            left_eye_indices = (0,)
            right_eye_indices = (1,)

            def __init__(self) -> None:
                self.extraction_shapes: list[tuple[int, int]] = []

            def _extract_points(self, image: np.ndarray) -> np.ndarray:
                height, width = image.shape[:2]
                self.extraction_shapes.append((height, width))
                return full_points * np.array(
                    [width / full_width, height / full_height],
                    dtype=np.float32,
                )

        extractor = RecordingExtractor()
        with patch(
            "ubo_morph.morphing.core._morph_pipeline",
            return_value=self.image,
        ) as pipeline:
            result = morph_images(
                self.image,
                self.image,
                extractor,
                landmark_extraction_short_side=10,
            )

        assert result is self.image
        assert extractor.extraction_shapes == [(10, 10), (10, 10)]
        first_image, second_image, landmarks1, landmarks2 = pipeline.call_args.args
        assert first_image.shape == self.image.shape
        assert second_image.shape == self.image.shape
        np.testing.assert_allclose(landmarks1.points, full_points)
        np.testing.assert_allclose(landmarks2.points, full_points)
        assert isinstance(pipeline.call_args.kwargs["backend"], CPUBackend)

    def test_high_level_api_returns_details(self) -> None:
        result = morph_with_landmarks(
            self.image,
            self.image,
            self.landmarks,
            self.landmarks,
            align_eye_centers=False,
            points_per_border=0,
            automatic_retouching=False,
            return_details=True,
        )

        assert isinstance(result, MorphResult)
        assert result.image.shape == self.image.shape
        assert result.image.dtype == np.uint8
        np.testing.assert_allclose(result.morphed_points, self.points)
        assert result.original_landmarks1 is self.landmarks
        assert result.original_landmarks2 is self.landmarks
        assert result.before_background_substitution is None
        assert not hasattr(result, "before_equalization_image1")
        assert not hasattr(result, "before_equalization_image2")
        assert result.after_equalization_image1 is None
        assert result.after_equalization_image2 is None

    def test_points_per_border_is_forwarded_and_zero_disables_it(self) -> None:
        counts: list[int] = []

        def record_count(
            image: np.ndarray,
            points: np.ndarray,
            points_per_border: int,
        ) -> np.ndarray:
            del image
            counts.append(points_per_border)
            return points

        with patch(
            "ubo_morph.morphing.core.add_border_points",
            side_effect=record_count,
        ):
            morph_with_landmarks(
                self.image,
                self.image,
                self.landmarks,
                self.landmarks,
                align_eye_centers=False,
                points_per_border=3,
                automatic_retouching=False,
            )
            morph_with_landmarks(
                self.image,
                self.image,
                self.landmarks,
                self.landmarks,
                align_eye_centers=False,
                points_per_border=0,
                automatic_retouching=False,
            )

        assert counts == [3, 3]

    @pytest.mark.parametrize("count", [-1, 1])
    def test_points_per_border_rejects_one_and_negative_counts(
        self,
        count: int,
    ) -> None:
        with pytest.raises(ValueError, match="points_per_border"):
            morph_with_landmarks(
                self.image,
                self.image,
                self.landmarks,
                self.landmarks,
                align_eye_centers=False,
                points_per_border=count,
                automatic_retouching=False,
            )

    @pytest.mark.parametrize(
        ("blending_factor", "equalized_index"),
        [(0.25, 2), (0.75, 1)],
    )
    def test_details_capture_equalization_and_background_boundaries(
        self,
        blending_factor: float,
        equalized_index: int,
    ) -> None:
        image1 = self.image
        image2 = np.full_like(image1, 180)
        substituted = np.full_like(image1, 240)

        class BoundaryBackend(CPUBackend):
            def match_histogram_channel(
                self,
                channel: np.ndarray,
                mask: np.ndarray,
                reference_channel: np.ndarray,
                reference_mask: np.ndarray,
            ) -> np.ndarray:
                del self, mask, reference_channel, reference_mask
                return np.full_like(channel, 60)

            def warp_triangles(
                self,
                image: np.ndarray,
                source_points: np.ndarray,
                target_points: np.ndarray,
                triangles: Sequence[tuple[int, int, int]],
            ) -> np.ndarray:
                del self, source_points, target_points, triangles
                return image

            def blend(
                self,
                image1: np.ndarray,
                image2: np.ndarray,
                blending_factor: float = 0.5,
                *,
                foreground_alpha: np.ndarray | None = None,
            ) -> np.ndarray:
                if foreground_alpha is not None:
                    return substituted
                return super().blend(
                    image1=image1,
                    image2=image2,
                    blending_factor=blending_factor,
                )

        backend = BoundaryBackend()

        with patch("ubo_morph.morphing.core.get_backend", return_value=backend):
            result = morph_with_landmarks(
                image1,
                image2,
                self.landmarks,
                self.landmarks,
                blending_factor=blending_factor,
                align_eye_centers=False,
                points_per_border=0,
                return_details=True,
            )

        assert isinstance(result, MorphResult)
        np.testing.assert_array_equal(result.image, substituted)
        np.testing.assert_array_equal(result.aligned_image1, image1)
        np.testing.assert_array_equal(result.aligned_image2, image2)
        after1 = result.after_equalization_image1
        after2 = result.after_equalization_image2
        expected_equalized = np.full_like(image1, 60)
        changed = (after1, after2)[equalized_index - 1]
        unchanged = (after1, after2)[2 - equalized_index]
        assert changed is not None
        np.testing.assert_array_equal(changed, expected_equalized)
        assert unchanged is None
        work1 = expected_equalized if equalized_index == 1 else image1
        work2 = expected_equalized if equalized_index == 2 else image2
        np.testing.assert_array_equal(
            result.before_background_substitution,
            CPUBackend().blend(work1, work2, blending_factor),
        )
