from __future__ import annotations

import cv2
import numpy as np


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    blending_factor: float,
) -> np.ndarray:
    return cv2.addWeighted(image1, 1.0 - blending_factor, image2, blending_factor, 0.0)
