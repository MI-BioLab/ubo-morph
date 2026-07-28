from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache
from typing import Any, ClassVar, Generic, Literal, TypeVar

import numpy as np


BackendName = Literal["cpu", "cupy"]
ArrayT = TypeVar("ArrayT")


class Backend(ABC, Generic[ArrayT]):
    """Array-operation contract used by the backend-independent morphing core."""

    name: ClassVar[BackendName]

    @abstractmethod
    def to_backend(self, array: np.ndarray) -> ArrayT:
        """Move a NumPy array to this backend without changing its values."""

    @abstractmethod
    def to_numpy(self, array: ArrayT) -> np.ndarray:
        """Return a backend array as a NumPy array."""

    @abstractmethod
    def pad_image(
        self,
        image: ArrayT,
        padding: tuple[int, int, int, int],
    ) -> ArrayT:
        """Pad an image with zeros using (top, bottom, left, right)."""

    @abstractmethod
    def warp_affine(
        self,
        image: ArrayT,
        matrix: np.ndarray,
        output_shape: tuple[int, int],
    ) -> ArrayT:
        """Apply a forward affine transform into (height, width)."""

    @abstractmethod
    def resize_image(
        self,
        image: ArrayT,
        output_shape: tuple[int, int],
    ) -> ArrayT:
        """Resize an image to (height, width)."""

    def convex_hull_mask(
        self,
        shape: tuple[int, int],
        points: np.ndarray,
    ) -> ArrayT:
        """Build a backend-resident boolean mask for a point-set hull."""
        from ubo_morph.morphing.points import convex_hull_mask

        return self.to_backend(convex_hull_mask(shape, points))

    @abstractmethod
    def match_histogram_channel(
        self,
        channel: ArrayT,
        mask: ArrayT,
        reference_channel: ArrayT,
        reference_mask: ArrayT,
    ) -> ArrayT:
        """Histogram-match one channel inside the supplied masks."""

    def match_histogram_image(
        self,
        image: ArrayT,
        mask: ArrayT,
        reference_image: ArrayT,
        reference_mask: ArrayT,
    ) -> ArrayT:
        """Histogram-match all image channels, allowing batched backends."""
        image_array: Any = image
        reference_array: Any = reference_image
        result = image_array.copy()
        for channel_index in range(3):
            result[:, :, channel_index] = self.match_histogram_channel(
                image_array[:, :, channel_index],
                mask,
                reference_array[:, :, channel_index],
                reference_mask,
            )
        return result

    @abstractmethod
    def equalize_lightness(
        self,
        reference_image: ArrayT,
        image_to_equalize: ArrayT,
        reference_mask: ArrayT,
        equalize_mask: ArrayT,
    ) -> ArrayT:
        """Match HLS lightness while preserving the other channels."""

    @abstractmethod
    def copy_with_mask(
        self,
        image: ArrayT,
        background: ArrayT,
        mask: ArrayT,
    ) -> ArrayT:
        """Copy image pixels over a background where mask is true."""

    @abstractmethod
    def feather_fields(
        self,
        mask: ArrayT,
        element_size: int,
    ) -> tuple[ArrayT, ArrayT]:
        """Return foreground and background alpha fields for a face mask."""

    def blend_with_feather(
        self,
        image: ArrayT,
        background: ArrayT,
        mask: ArrayT,
        *,
        element_size: int,
        transition: int,
    ) -> ArrayT:
        """Construct a feather alpha field and blend, allowing backend fusion."""
        eroded, distance = self.feather_fields(mask, element_size)
        distance_array: Any = distance
        mask_array: Any = mask
        eroded_array: Any = eroded
        distance_array[0, :] = 0.0
        distance_array[-1, :] = 0.0
        distance_array[:, 0] = 0.0
        distance_array[:, -1] = 0.0
        foreground_alpha = (distance_array - 1.0) / transition
        foreground_alpha[foreground_alpha < 0.0] = 0.0
        foreground_alpha[foreground_alpha > 1.0] = 1.0
        foreground_alpha[eroded_array > 0] = 1.0
        foreground_alpha[mask_array == 0] = 0.0
        return self.blend(
            image1=image,
            image2=background,
            foreground_alpha=foreground_alpha,
        )

    @abstractmethod
    def blend(
        self,
        image1: ArrayT,
        image2: ArrayT,
        blending_factor: float = 0.5,
        *,
        foreground_alpha: ArrayT | None = None,
    ) -> ArrayT:
        """Blend images using a scalar factor or a foreground alpha field."""

    @abstractmethod
    def warp_triangles(
        self,
        image: ArrayT,
        source_points: np.ndarray,
        target_points: np.ndarray,
        triangles: Sequence[tuple[int, int, int]],
    ) -> ArrayT:
        """Piecewise-affine warp an image through matching triangles."""


@cache
def get_backend(name: BackendName = "cpu") -> Backend[Any]:
    """Return the cached concrete backend selected by its exact public name."""
    if name == "cpu":
        from ubo_morph.morphing.cpu import CPUBackend

        return CPUBackend()
    if name == "cupy":
        try:
            from ubo_morph.morphing.cupy import CuPyBackend
        except ModuleNotFoundError as error:
            if error.name != "cupy":
                raise
            raise ImportError(
                'The "cupy" backend is not available. '
                "Install it with: pip install 'ubo-morph[cupy]'"
            ) from None
        return CuPyBackend()
    raise ValueError("backend must be one of: cpu, cupy")
