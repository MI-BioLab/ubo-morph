from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ubo_morph.landmarks import Landmarks
from ubo_morph.utils import round_away


@dataclass(frozen=True, slots=True)
class AlignmentGeometry:
    """Backend-independent geometry for face tokenization."""

    border: tuple[int, int]
    padded_shape: tuple[int, int]
    rotation_matrix: np.ndarray
    crop_origin: tuple[int, int]
    crop_shape: tuple[int, int]
    crop_padding: tuple[int, int, int, int]
    landmarks: Landmarks


def alignment_geometry(
    image_shape: tuple[int, int],
    landmarks: Landmarks,
) -> AlignmentGeometry:
    """Compute the shared CPU/CuPy alignment transform and crop coordinates."""
    height, width = image_shape
    border_x = round_away(width * 0.25)
    border_y = round_away(height * 0.25)
    padded_height = height + 2 * border_y
    padded_width = width + 2 * border_x

    padding_int = np.array([border_x, border_y], dtype=np.int64)
    left_eye = np.floor(landmarks.left_eye + 0.5).astype(np.int64) + padding_int
    right_eye = np.floor(landmarks.right_eye + 0.5).astype(np.int64) + padding_int
    rotation_angle = math.atan2(
        int(right_eye[1]) - int(left_eye[1]),
        int(left_eye[0]) - int(right_eye[0]),
    )
    rotation_center = ((padded_width - 1) / 2.0, (padded_height - 1) / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(
        rotation_center,
        -math.degrees(rotation_angle),
        1.0,
    ).astype(np.float32)

    eye_points = np.array([left_eye, right_eye], dtype=np.float32)
    rotated_eyes = np.floor(
        cv2.transform(eye_points[None], rotation_matrix)[0] + 0.5
    ).astype(np.int64)
    eye_delta = rotated_eyes[0] - rotated_eyes[1]
    eye_distance = math.hypot(float(eye_delta[0]), float(eye_delta[1]))
    if eye_distance == 0.0:
        raise ValueError("left and right eye centers must be distinct")

    eye_midpoint = np.floor(rotated_eyes.mean(axis=0) + 0.5).astype(np.int64)
    crop_width = round_away(4.0 * eye_distance)
    crop_height = round_away(4.0 * crop_width / 3.0)
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("eye centers produce an empty alignment crop")
    crop_left = int(eye_midpoint[0]) - crop_width // 2
    crop_top = int(eye_midpoint[1]) - round_away(3.0 * crop_width / 5.0)

    pad_left = max(-crop_left, 0)
    pad_top = max(-crop_top, 0)
    pad_right = max(crop_left + crop_width - padded_width, 0)
    pad_bottom = max(crop_top + crop_height - padded_height, 0)

    all_points = np.vstack(
        [landmarks.left_eye[None], landmarks.right_eye[None], landmarks.points]
    ).astype(np.float32)
    all_points += np.array([border_x, border_y], dtype=np.float32)
    transformed = cv2.transform(all_points[None], rotation_matrix)[0]
    transformed -= np.array([crop_left, crop_top], dtype=np.float32)
    aligned_landmarks = Landmarks(
        left_eye=transformed[0],
        right_eye=transformed[1],
        points=transformed[2:],
    )

    return AlignmentGeometry(
        border=(border_x, border_y),
        padded_shape=(padded_height, padded_width),
        rotation_matrix=rotation_matrix,
        crop_origin=(crop_left + pad_left, crop_top + pad_top),
        crop_shape=(crop_height, crop_width),
        crop_padding=(pad_top, pad_bottom, pad_left, pad_right),
        landmarks=aligned_landmarks,
    )


def scale_landmarks(
    landmarks: Landmarks,
    scale_x: float,
    scale_y: float,
) -> Landmarks:
    scale = np.array([scale_x, scale_y], dtype=np.float32)
    return Landmarks(
        landmarks.left_eye * scale,
        landmarks.right_eye * scale,
        landmarks.points * scale,
    )
