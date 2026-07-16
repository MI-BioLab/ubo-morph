from __future__ import annotations

from functools import cache

from ubo_morph.device import validate_device
from ubo_morph.morphing._protocols import MorphingImpl


@cache
def _impl(device: str) -> MorphingImpl:
    """Load and return the typed implementation bundle for a device."""
    if validate_device(device) == "cpu":
        from ubo_morph.morphing.cpu import (
            align_face_images,
            blend_images,
            delaunay_triangles,
            equalize_face,
            substitute_background,
            warp_image_by_triangles,
        )
    else:
        from ubo_morph.morphing.gpu import (
            align_face_images,
            blend_images,
            delaunay_triangles,
            equalize_face,
            substitute_background,
            warp_image_by_triangles,
        )

    return MorphingImpl(
        align_face_images=align_face_images,
        blend_images=blend_images,
        delaunay_triangles=delaunay_triangles,
        equalize_face=equalize_face,
        substitute_background=substitute_background,
        warp_image_by_triangles=warp_image_by_triangles,
    )
