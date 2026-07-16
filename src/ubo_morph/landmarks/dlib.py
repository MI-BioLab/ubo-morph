from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ubo_morph.landmarks import LandmarkExtractor


class DlibLandmarkExtractor(LandmarkExtractor):
    """Extract 68-point facial landmarks using dlib's shape predictor."""

    left_eye_indices = (42, 43, 44, 45, 46, 47)
    right_eye_indices = (36, 37, 38, 39, 40, 41)

    def __init__(
        self,
        model_path: str | Path,
        *,
        upsample_times: int = 1,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f'Dlib shape-predictor model "{model_path}" does not exist.')
        if upsample_times < 0:
            raise ValueError("upsample_times must be non-negative")

        dlib = _import_dlib()
        self._detector: Any = dlib.get_frontal_face_detector()
        self._predictor: Any = dlib.shape_predictor(str(model_path))
        self._upsample_times = int(upsample_times)

    def _extract_points(self, image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = list(self._detector(rgb_image, self._upsample_times))
        if not faces:
            raise ValueError("Dlib did not detect a face in the image.")

        face = max(faces, key=_rectangle_area)
        shape = self._predictor(rgb_image, face)
        point_count = int(shape.num_parts)
        if point_count < 68:
            raise ValueError(
                "DlibLandmarkExtractor requires a 68-point shape predictor; "
                f"the configured model returned {point_count} points."
            )
        return np.asarray(
            [(shape.part(index).x, shape.part(index).y) for index in range(point_count)],
            dtype=np.float32,
        )


def _rectangle_area(rectangle: Any) -> int:
    width = max(int(rectangle.right()) - int(rectangle.left()) + 1, 0)
    height = max(int(rectangle.bottom()) - int(rectangle.top()) + 1, 0)
    return width * height


def _import_dlib() -> Any:
    try:
        return import_module("dlib")
    except ModuleNotFoundError as error:
        if error.name != "dlib":
            raise
        raise ModuleNotFoundError(
            'DlibLandmarkExtractor requires dlib; install it with `pip install "ubo-morph[dlib]"`.'
        ) from None


__all__ = ["DlibLandmarkExtractor"]
