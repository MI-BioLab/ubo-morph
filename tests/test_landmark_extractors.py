from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ubo_morph import (
    DlibLandmarkExtractor,
    LandmarkExtractor,
    MediaPipeLandmarkExtractor,
)


class _RecordingLandmarkExtractor(LandmarkExtractor):
    left_eye_indices = (0,)
    right_eye_indices = (1,)

    def __init__(self) -> None:
        self.extraction_shape: tuple[int, int] | None = None

    def _extract_points(self, image: np.ndarray) -> np.ndarray:
        self.extraction_shape = image.shape[:2]
        return np.array(
            [[1.0, 1.0], [3.0, 1.0], [2.0, 2.0]],
            dtype=np.float32,
        )


class _FakeRectangle:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def area(self) -> int:
        width = max(self._right - self._left + 1, 0)
        height = max(self._bottom - self._top + 1, 0)
        return width * height


class _FakeDlibDetector:
    def __init__(self, faces: list[_FakeRectangle]) -> None:
        self.faces = faces
        self.image: np.ndarray | None = None
        self.upsample_times: int | None = None

    def __call__(self, image: np.ndarray, upsample_times: int) -> list[_FakeRectangle]:
        self.image = image
        self.upsample_times = upsample_times
        return self.faces


class _FakeDlibPredictor:
    def __init__(self) -> None:
        self.face: _FakeRectangle | None = None

    def __call__(self, image: np.ndarray, face: _FakeRectangle) -> SimpleNamespace:
        self.face = face
        points = [SimpleNamespace(x=index, y=index * 2) for index in range(68)]
        return SimpleNamespace(parts=lambda: points)


class TestLandmarkExtractor:
    def test_short_side_limit_restores_full_image_coordinates(self) -> None:
        extractor = _RecordingLandmarkExtractor()
        image = np.zeros((7, 11, 3), dtype=np.uint8)

        landmarks = extractor.extract(image, max_short_side=4)

        assert extractor.extraction_shape == (4, 6)
        np.testing.assert_allclose(
            landmarks.points,
            np.array(
                [
                    [11 / 6, 7 / 4],
                    [11 / 2, 7 / 4],
                    [11 / 3, 7 / 2],
                ],
                dtype=np.float32,
            ),
        )

    def test_short_side_limit_does_not_upscale_and_rejects_negative_values(
        self,
    ) -> None:
        extractor = _RecordingLandmarkExtractor()
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        landmarks = extractor.extract(image, max_short_side=20)

        assert extractor.extraction_shape == (10, 10)
        np.testing.assert_allclose(
            landmarks.points,
            np.array([[1, 1], [3, 1], [2, 2]], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="max_short_side"):
            extractor.extract(image, max_short_side=-1)

    def test_dlib_uses_largest_face_and_anatomical_eye_indices(
        self,
        tmp_path: Path,
    ) -> None:
        small_face = _FakeRectangle(0, 0, 1, 1)
        large_face = _FakeRectangle(0, 0, 9, 9)
        detector = _FakeDlibDetector([small_face, large_face])
        predictor = _FakeDlibPredictor()
        fake_dlib = SimpleNamespace(
            get_frontal_face_detector=lambda: detector,
            shape_predictor=lambda model_path: predictor,
        )
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[0, 0] = (1, 2, 3)

        model_path = tmp_path / "shape_predictor.dat"
        model_path.touch()
        with patch("ubo_morph.landmarks.dlib.import_module", return_value=fake_dlib):
            landmarks = DlibLandmarkExtractor(model_path, upsample_times=2).extract(
                image
            )

        assert predictor.face is large_face
        assert detector.upsample_times == 2
        assert detector.image is not None
        np.testing.assert_array_equal(detector.image[0, 0], np.array([3, 2, 1]))
        assert landmarks.points.shape == (68, 2)
        np.testing.assert_allclose(landmarks.left_eye, (44.5, 89.0))
        np.testing.assert_allclose(landmarks.right_eye, (38.5, 77.0))

    def test_dlib_reports_when_no_face_is_detected(self, tmp_path: Path) -> None:
        fake_dlib = SimpleNamespace(
            get_frontal_face_detector=lambda: _FakeDlibDetector([]),
            shape_predictor=lambda model_path: _FakeDlibPredictor(),
        )
        model_path = tmp_path / "shape_predictor.dat"
        model_path.touch()
        with patch("ubo_morph.landmarks.dlib.import_module", return_value=fake_dlib):
            extractor = DlibLandmarkExtractor(model_path)
            with pytest.raises(ValueError, match="did not detect a face"):
                extractor.extract(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_mediapipe_converts_largest_face_to_pixel_coordinates_and_closes(
        self,
        tmp_path: Path,
    ) -> None:
        small_face = _normalized_face(0.01)
        large_face = _CountingFace(_normalized_face(0.02))
        detector = _FakeMediaPipeDetector([small_face, large_face])
        created_options: list[SimpleNamespace] = []

        def create_options(**values: object) -> SimpleNamespace:
            options = SimpleNamespace(**values)
            created_options.append(options)
            return options

        fake_mediapipe = SimpleNamespace(
            Image=lambda **values: SimpleNamespace(**values),
            ImageFormat=SimpleNamespace(SRGB="srgb"),
            tasks=SimpleNamespace(
                BaseOptions=lambda **values: SimpleNamespace(**values),
                vision=SimpleNamespace(
                    FaceLandmarkerOptions=create_options,
                    RunningMode=SimpleNamespace(IMAGE="image"),
                    FaceLandmarker=SimpleNamespace(
                        create_from_options=lambda options: detector,
                    ),
                ),
            ),
        )
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[0, 0] = (1, 2, 3)

        model_path = tmp_path / "face_landmarker.task"
        model_path.touch()
        with patch(
            "ubo_morph.landmarks.mediapipe.import_module",
            return_value=fake_mediapipe,
        ):
            extractor = MediaPipeLandmarkExtractor(model_path, max_faces=2)
            landmarks = extractor.extract(image)
            extractor.close()

        assert created_options[0].num_faces == 2
        assert detector.image is not None
        assert detector.image.image_format == "srgb"
        np.testing.assert_array_equal(detector.image.data[0, 0], np.array([3, 2, 1]))
        assert landmarks.points.shape == (468, 2)
        np.testing.assert_allclose(landmarks.points[0], (2.0, 2.0))
        np.testing.assert_allclose(landmarks.points[19], (9.6, 2.0))
        assert large_face.iterations == 1
        assert detector.closed
        with pytest.raises(RuntimeError, match="is closed"):
            extractor.extract(image)


class _FakeMediaPipeDetector:
    def __init__(self, faces: list[list[SimpleNamespace]]) -> None:
        self.faces = faces
        self.image: SimpleNamespace | None = None
        self.closed = False

    def detect(self, image: SimpleNamespace) -> SimpleNamespace:
        self.image = image
        return SimpleNamespace(face_landmarks=self.faces)

    def close(self) -> None:
        self.closed = True


class _CountingFace(list[SimpleNamespace]):
    def __init__(self, values: list[SimpleNamespace]) -> None:
        super().__init__(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def _normalized_face(spread: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=0.1 + (index % 20) * spread,
            y=0.2 + (index // 20) * spread,
        )
        for index in range(468)
    ]
