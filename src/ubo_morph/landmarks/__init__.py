from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ubo_morph.utils import ensure_bgr_uint8, as_point


@dataclass(slots=True)
class Landmarks:
    """Facial points and anatomical left/right eye centers in image coordinates."""

    left_eye: np.ndarray
    right_eye: np.ndarray
    points: np.ndarray

    def __post_init__(self) -> None:
        self.left_eye = as_point(self.left_eye)
        self.right_eye = as_point(self.right_eye)
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("landmark points must have shape (n, 2)")
        if len(self.points) < 3:
            raise ValueError("at least three landmark points are required")


class LandmarkExtractor(ABC):
    left_eye_indices: ClassVar[tuple[int, ...]]
    right_eye_indices: ClassVar[tuple[int, ...]]

    def extract(self, image: np.ndarray) -> Landmarks:
        points = np.asarray(
            self._extract_points(ensure_bgr_uint8(image)),
            dtype=np.float32,
        )
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("extracted landmark points must have shape (n, 2)")
        if len(points) < 3:
            raise ValueError("at least three landmark points are required")

        eye_indices = self.left_eye_indices + self.right_eye_indices
        required_point_count = max(eye_indices) + 1
        if len(points) < required_point_count:
            raise ValueError(
                f"{type(self).__name__} returned {len(points)} landmarks, "
                f"but its eye indices require at least {required_point_count}"
            )

        left_eye = points[list(self.left_eye_indices)].mean(axis=0)
        right_eye = points[list(self.right_eye_indices)].mean(axis=0)
        return Landmarks(left_eye, right_eye, points)

    @abstractmethod
    def _extract_points(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> LandmarkExtractor:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


from ubo_morph.landmarks.dlib import DlibLandmarkExtractor  # noqa: E402
from ubo_morph.landmarks.mediapipe import MediaPipeLandmarkExtractor  # noqa: E402

__all__ = [
    "DlibLandmarkExtractor",
    "LandmarkExtractor",
    "Landmarks",
    "MediaPipeLandmarkExtractor",
]
