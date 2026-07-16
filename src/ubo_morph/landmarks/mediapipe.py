from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ubo_morph.landmarks import LandmarkExtractor


class MediaPipeLandmarkExtractor(LandmarkExtractor):
    """Extract facial landmarks using the MediaPipe Face Landmarker task."""

    left_eye_indices = (
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    )
    right_eye_indices = (
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
    )

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f'MediaPipe face-landmarker model "{model_path}" does not exist.')
        if max_faces < 1:
            raise ValueError("max_faces must be at least 1")
        detection_confidence = _validate_confidence(
            min_face_detection_confidence,
            "min_face_detection_confidence",
        )
        presence_confidence = _validate_confidence(
            min_face_presence_confidence,
            "min_face_presence_confidence",
        )
        tracking_confidence = _validate_confidence(
            min_tracking_confidence,
            "min_tracking_confidence",
        )

        mediapipe = _import_mediapipe()
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mediapipe.tasks.vision.RunningMode.IMAGE,
            num_faces=int(max_faces),
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mediapipe: Any = mediapipe
        self._landmarker: Any | None = (
            mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
        )

    def _extract_points(self, image: np.ndarray) -> np.ndarray:
        if self._landmarker is None:
            raise RuntimeError("MediaPipeLandmarkExtractor is closed.")

        rgb_image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mediapipe_image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=rgb_image,
        )
        result = self._landmarker.detect(mediapipe_image)
        if not result.face_landmarks:
            raise ValueError("MediaPipe did not detect a face in the image.")

        face_landmarks = max(result.face_landmarks, key=_normalized_face_area)
        normalized_points = np.asarray(
            [(landmark.x, landmark.y) for landmark in face_landmarks],
            dtype=np.float32,
        )
        height, width = image.shape[:2]
        return normalized_points * np.array([width, height], dtype=np.float32)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def _normalized_face_area(landmarks: Any) -> float:
    points = np.asarray(
        [(landmark.x, landmark.y) for landmark in landmarks],
        dtype=np.float32,
    )
    span = np.ptp(points, axis=0)
    return float(span[0] * span[1])


def _validate_confidence(value: float, name: str) -> float:
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return confidence


def _import_mediapipe() -> Any:
    try:
        return import_module("mediapipe")
    except ModuleNotFoundError as error:
        if error.name != "mediapipe":
            raise
        raise ModuleNotFoundError(
            "MediaPipeLandmarkExtractor requires mediapipe; "
            'install it with `pip install "ubo-morph[mediapipe]"`.'
        ) from None


__all__ = ["MediaPipeLandmarkExtractor"]
