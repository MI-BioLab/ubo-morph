from __future__ import annotations

from functools import cache
from importlib.resources import files


KERNEL_NAMES = (
    "affine_warp_uint8",
    "area_resize_uint8",
    "rasterize_convex_hull",
    "warp_triangle_boxes",
    "build_box_membership",
    "sample_box_membership",
    "blend_uint8",
    "bgr_to_hls",
    "hls_to_bgr",
    "build_histograms",
    "build_histogram_lookup",
    "apply_histogram_lookup",
    "feather_blend",
)


@cache
def load_kernel_source() -> str:
    """
    Read the CUDA translation unit shipped beside this module.
    """
    return files(__package__).joinpath("kernels.cu").read_text(encoding="utf-8")

