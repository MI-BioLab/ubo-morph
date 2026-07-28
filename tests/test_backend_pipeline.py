from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ubo_morph import LandmarkExtractor, Landmarks, MorphResult
from ubo_morph.morphing import Backend, morph_images, morph_with_landmarks
from ubo_morph.morphing.core import _substitute_background


class RecordingBackend(Backend[np.ndarray]):
    name = "cpu"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _record(self, name: str) -> None:
        self.calls.append(name)

    def to_backend(self, array: np.ndarray) -> np.ndarray:
        self._record("to_backend")
        return array.copy()

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        self._record("to_numpy")
        return array.copy()

    def pad_image(
        self,
        image: np.ndarray,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        self._record("pad_image")
        top, bottom, left, right = padding
        return np.pad(image, ((top, bottom), (left, right), (0, 0)))

    def warp_affine(
        self,
        image: np.ndarray,
        matrix: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        self._record("warp_affine")
        height, width = output_shape
        return cv2.warpAffine(image, matrix, (width, height))

    def resize_image(
        self,
        image: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        self._record("resize_image")
        height, width = output_shape
        return cv2.resize(image, (width, height))

    def match_histogram_channel(
        self,
        channel: np.ndarray,
        mask: np.ndarray,
        reference_channel: np.ndarray,
        reference_mask: np.ndarray,
    ) -> np.ndarray:
        del reference_channel, reference_mask
        self._record("match_histogram_channel")
        result = channel.copy()
        result[mask.astype(bool)] = 64
        return result

    def equalize_lightness(
        self,
        reference_image: np.ndarray,
        image_to_equalize: np.ndarray,
        reference_mask: np.ndarray,
        equalize_mask: np.ndarray,
    ) -> np.ndarray:
        del reference_image, reference_mask, equalize_mask
        self._record("equalize_lightness")
        return image_to_equalize.copy()

    def copy_with_mask(
        self,
        image: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        self._record("copy_with_mask")
        return np.where(mask[:, :, None], image, background)

    def feather_fields(
        self,
        mask: np.ndarray,
        element_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del element_size
        self._record("feather_fields")
        foreground = mask.astype(np.float32)
        return foreground, 1.0 - foreground

    def blend(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        blending_factor: float = 0.5,
        *,
        foreground_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        self._record("blend")
        if foreground_alpha is not None:
            self._record("alpha_blend")
            alpha = foreground_alpha[:, :, None]
            return (alpha * image1 + (1.0 - alpha) * image2).astype(np.uint8)
        return cv2.addWeighted(
            image1,
            1.0 - blending_factor,
            image2,
            blending_factor,
            0.0,
        )

    def warp_triangles(
        self,
        image: np.ndarray,
        source_points: np.ndarray,
        target_points: np.ndarray,
        triangles: Sequence[tuple[int, int, int]],
    ) -> np.ndarray:
        del source_points, target_points
        self._record("warp_triangles")
        if not triangles:
            raise AssertionError("core did not create Delaunay triangles")
        return image.copy()


class TestBackendPipeline:
    @pytest.fixture(autouse=True)
    def _images_and_landmarks(self) -> None:
        self.image1 = np.full((24, 24, 3), 20, dtype=np.uint8)
        self.image2 = np.full((28, 30, 3), 200, dtype=np.uint8)
        points1 = np.array(
            [[3, 3], [20, 3], [20, 20], [3, 20], [12, 12]],
            dtype=np.float32,
        )
        points2 = np.array(
            [[4, 4], [25, 4], [25, 23], [4, 23], [15, 14]],
            dtype=np.float32,
        )
        self.landmarks1 = Landmarks(
            left_eye=np.array([17, 9], dtype=np.float32),
            right_eye=np.array([7, 9], dtype=np.float32),
            points=points1,
        )
        self.landmarks2 = Landmarks(
            left_eye=np.array([23, 10], dtype=np.float32),
            right_eye=np.array([7, 10], dtype=np.float32),
            points=points2,
        )

    def test_core_owns_shared_pipeline_and_uses_only_backend_primitives(self) -> None:
        backend = RecordingBackend()

        with patch("ubo_morph.morphing.core.get_backend", return_value=backend):
            result = morph_with_landmarks(
                self.image1,
                self.image2,
                self.landmarks1,
                self.landmarks2,
                points_per_border=3,
                return_details=True,
                backend="cpu",
            )

        assert isinstance(result, MorphResult)
        assert result.image.dtype == np.uint8
        assert result.aligned_image1.shape == result.aligned_image2.shape
        assert len(result.morphed_points) > len(self.landmarks1.points)
        assert backend.calls.count("warp_affine") == 2
        assert backend.calls.count("pad_image") >= 2
        assert backend.calls.count("resize_image") >= 1
        assert backend.calls.count("match_histogram_channel") == 3
        assert backend.calls.count("warp_triangles") == 2
        assert backend.calls.count("blend") == 2
        assert backend.calls.count("alpha_blend") == 1
        assert backend.calls.count("feather_fields") == 1
        assert "blend_with_alpha" not in backend.calls
        assert "copy_with_mask" not in backend.calls
        assert backend.calls.count("to_backend") >= 4
        assert backend.calls.count("to_numpy") >= 6

    def test_backend_is_resolved_before_landmark_extraction(self) -> None:
        extractor = MagicMock(spec=LandmarkExtractor)
        events: list[str] = []
        extractor.extract.side_effect = lambda *args, **kwargs: events.append("extract")

        def fail_backend(name: str) -> None:
            del name
            events.append("backend")
            raise RuntimeError("stop")

        with patch(
            "ubo_morph.morphing.core.get_backend",
            side_effect=fail_backend,
        ):
            with pytest.raises(RuntimeError, match="stop"):
                morph_images(self.image1, self.image2, extractor, backend="cpu")

        assert events == ["backend"]
        extractor.extract.assert_not_called()

    def test_core_calculates_feather_alpha_from_backend_fields(self) -> None:
        class FeatherBackend(RecordingBackend):
            foreground_alpha: np.ndarray | None = None
            background_alpha: np.ndarray | None = None

            def feather_fields(
                self,
                mask: np.ndarray,
                element_size: int,
            ) -> tuple[np.ndarray, np.ndarray]:
                del element_size
                eroded = np.zeros_like(mask, dtype=bool)
                distance = np.zeros_like(mask, dtype=np.float32)
                distance[1, 1] = 1.5
                return eroded, distance

            def blend(
                self,
                image1: np.ndarray,
                image2: np.ndarray,
                blending_factor: float = 0.5,
                *,
                foreground_alpha: np.ndarray | None = None,
            ) -> np.ndarray:
                del image2, blending_factor
                assert foreground_alpha is not None
                self.foreground_alpha = foreground_alpha.copy()
                self.background_alpha = 1.0 - foreground_alpha
                return image1

        backend = FeatherBackend()
        image = np.full((3, 3, 3), 200, dtype=np.uint8)
        background = np.full_like(image, 20)
        points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)

        _substitute_background(
            backend,
            image,
            points,
            background,
            blend=True,
            eye_distance=20.0,
        )

        assert backend.foreground_alpha is not None
        assert backend.background_alpha is not None
        assert float(backend.foreground_alpha[1, 1]) == 0.5
        assert float(backend.background_alpha[1, 1]) == 0.5
        assert np.all(backend.foreground_alpha[[0, -1], :] == 0.0)

    def test_backend_is_resolved_before_image_normalization(self) -> None:
        events: list[str] = []

        def fail_backend(name: str) -> None:
            del name
            events.append("backend")
            raise RuntimeError("stop")

        with (
            patch(
                "ubo_morph.morphing.core.get_backend",
                side_effect=fail_backend,
            ),
            patch(
                "ubo_morph.morphing.core.ensure_bgr_uint8",
                side_effect=lambda image: events.append("normalize") or image,
            ),
        ):
            with pytest.raises(RuntimeError, match="stop"):
                morph_with_landmarks(
                    self.image1,
                    self.image2,
                    self.landmarks1,
                    self.landmarks2,
                    backend="cpu",
                )

        assert events == ["backend"]
