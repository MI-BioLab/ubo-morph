from __future__ import annotations

import math

import cv2
import numpy as np

from ubo_morph.landmarks import Landmarks
from ubo_morph.utils import ensure_bgr_uint8, round_away


def align_face_images(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
) -> tuple[np.ndarray, np.ndarray, Landmarks, Landmarks]:
    tokenized1, aligned_landmarks1 = _tokenize_image_and_landmarks(image1, landmarks1)
    tokenized2, aligned_landmarks2 = _tokenize_image_and_landmarks(image2, landmarks2)
    target_width = max(tokenized1.shape[1], tokenized2.shape[1])
    target_height = max(tokenized1.shape[0], tokenized2.shape[0])
    output1, aligned_landmarks1 = _resize_image_and_landmarks(
        tokenized1, aligned_landmarks1, target_width, target_height
    )
    output2, aligned_landmarks2 = _resize_image_and_landmarks(
        tokenized2, aligned_landmarks2, target_width, target_height
    )
    return output1, output2, aligned_landmarks1, aligned_landmarks2


def _tokenize_image_and_landmarks(
    image: np.ndarray,
    landmarks: Landmarks,
) -> tuple[np.ndarray, Landmarks]:
    image = ensure_bgr_uint8(image)
    height, width = image.shape[:2]
    left_eye_int = np.floor(landmarks.left_eye + 0.5).astype(np.int64)
    right_eye_int = np.floor(landmarks.right_eye + 0.5).astype(np.int64)
    border_x = round_away(width * 0.25)
    border_y = round_away(height * 0.25)
    padded = cv2.copyMakeBorder(
        image,
        border_y,
        border_y,
        border_x,
        border_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    padding_int = np.array([border_x, border_y], dtype=np.int64)
    padded_left = left_eye_int + padding_int
    padded_right = right_eye_int + padding_int
    rotation_angle = math.atan2(
        int(padded_right[1]) - int(padded_left[1]),
        int(padded_left[0]) - int(padded_right[0]),
    )

    padded_height, padded_width = padded.shape[:2]
    rotation_center = ((padded_width - 1) / 2.0, (padded_height - 1) / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(
        rotation_center,
        -math.degrees(rotation_angle),
        1.0,
    ).astype(np.float32)
    rotated = cv2.warpAffine(
        padded,
        rotation_matrix,
        (padded_width, padded_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    eye_points = np.array([padded_left, padded_right], dtype=np.float32)
    rotated_eyes = np.floor(
        cv2.transform(eye_points[None], rotation_matrix)[0] + 0.5
    ).astype(np.int64)
    rotated_left, rotated_right = rotated_eyes

    eye_distance = np.linalg.norm(
        (rotated_left - rotated_right).astype(np.float32)
    ).item()
    if eye_distance == 0.0:
        raise ValueError("left and right eye centers must be distinct")
    eye_midpoint = np.floor((rotated_left + rotated_right) / 2.0 + 0.5).astype(np.int64)
    crop_width = round_away(4.0 * eye_distance)
    crop_height = round_away(4.0 * crop_width / 3.0)
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("eye centers produce an empty alignment crop")
    crop_left = int(eye_midpoint[0]) - crop_width // 2
    crop_top = int(eye_midpoint[1]) - round_away(3.0 * crop_width / 5.0)
    crop_origin = np.array([crop_left, crop_top], dtype=np.float32)

    rotated, crop_left, crop_top = _pad_to_include_rect(
        rotated, crop_left, crop_top, crop_width, crop_height
    )
    cropped = rotated[
        crop_top : crop_top + crop_height,
        crop_left : crop_left + crop_width,
    ].copy()

    padding = np.array([border_x, border_y], dtype=np.float32)
    input_eyes = np.array([landmarks.left_eye, landmarks.right_eye]) + padding
    tokenized_eyes = cv2.transform(input_eyes[None], rotation_matrix)[0] - crop_origin
    tokenized_left, tokenized_right = tokenized_eyes
    input_points = landmarks.points + padding
    tokenized_points = (
        cv2.transform(input_points[None], rotation_matrix)[0] - crop_origin
    )
    return cropped, Landmarks(tokenized_left, tokenized_right, tokenized_points)


def _resize_image_and_landmarks(
    image: np.ndarray,
    landmarks: Landmarks,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, Landmarks]:
    height, width = image.shape[:2]
    if width == target_width and height == target_height:
        return image, landmarks
    scale_x = target_width / width
    scale_y = target_height / height
    interpolation = (
        cv2.INTER_CUBIC if scale_x > 1.0 or scale_y > 1.0 else cv2.INTER_AREA
    )
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    scale = np.array([scale_x, scale_y], dtype=np.float32)
    return resized, Landmarks(
        landmarks.left_eye * scale,
        landmarks.right_eye * scale,
        landmarks.points * scale,
    )


def _pad_to_include_rect(
    image: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int]:
    pad_left = max(-left, 0)
    pad_top = max(-top, 0)
    pad_right = max(left + width - image.shape[1], 0)
    pad_bottom = max(top + height - image.shape[0], 0)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        left += pad_left
        top += pad_top
    return image, left, top
