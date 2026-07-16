from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ubo_morph import DlibLandmarkExtractor, MediaPipeLandmarkExtractor


class _FakeRectangle:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self) -> int:
        return self._left

    def top(self) -> int:
        return self._top

    def right(self) -> int:
        return self._right

    def bottom(self) -> int:
        return self._bottom


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
        return SimpleNamespace(
            num_parts=len(points),
            part=lambda index: points[index],
        )


class LandmarkExtractorTests(unittest.TestCase):
    def test_dlib_uses_largest_face_and_anatomical_eye_indices(self) -> None:
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

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory, "shape_predictor.dat")
            model_path.touch()
            with patch("ubo_morph.landmarks.dlib.import_module", return_value=fake_dlib):
                landmarks = DlibLandmarkExtractor(model_path, upsample_times=2).extract(image)

        self.assertIs(predictor.face, large_face)
        self.assertEqual(detector.upsample_times, 2)
        assert detector.image is not None
        np.testing.assert_array_equal(detector.image[0, 0], np.array([3, 2, 1]))
        self.assertEqual(landmarks.points.shape, (68, 2))
        np.testing.assert_allclose(landmarks.left_eye, (44.5, 89.0))
        np.testing.assert_allclose(landmarks.right_eye, (38.5, 77.0))

    def test_dlib_reports_when_no_face_is_detected(self) -> None:
        fake_dlib = SimpleNamespace(
            get_frontal_face_detector=lambda: _FakeDlibDetector([]),
            shape_predictor=lambda model_path: _FakeDlibPredictor(),
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory, "shape_predictor.dat")
            model_path.touch()
            with patch("ubo_morph.landmarks.dlib.import_module", return_value=fake_dlib):
                extractor = DlibLandmarkExtractor(model_path)
                with self.assertRaisesRegex(ValueError, "did not detect a face"):
                    extractor.extract(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_mediapipe_converts_largest_face_to_pixel_coordinates_and_closes(self) -> None:
        small_face = _normalized_face(0.01)
        large_face = _normalized_face(0.02)
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

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory, "face_landmarker.task")
            model_path.touch()
            with patch(
                "ubo_morph.landmarks.mediapipe.import_module",
                return_value=fake_mediapipe,
            ):
                extractor = MediaPipeLandmarkExtractor(model_path, max_faces=2)
                landmarks = extractor.extract(image)
                extractor.close()

        self.assertEqual(created_options[0].num_faces, 2)
        assert detector.image is not None
        self.assertEqual(detector.image.image_format, "srgb")
        np.testing.assert_array_equal(detector.image.data[0, 0], np.array([3, 2, 1]))
        self.assertEqual(landmarks.points.shape, (468, 2))
        np.testing.assert_allclose(landmarks.points[0], (2.0, 2.0))
        np.testing.assert_allclose(landmarks.points[19], (9.6, 2.0))
        self.assertTrue(detector.closed)
        with self.assertRaisesRegex(RuntimeError, "is closed"):
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


def _normalized_face(spread: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=0.1 + (index % 20) * spread,
            y=0.2 + (index // 20) * spread,
        )
        for index in range(468)
    ]


if __name__ == "__main__":
    unittest.main()
