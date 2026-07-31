"""Deterministic stage profiler for the class-based morphing backends.

Usage:
    uv run python profile_pipeline.py --reps 3

CPU stages are always measured. Optional GPU backends are measured when they
are installed and selectable.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ubo_morph.landmarks import DlibLandmarkExtractor, Landmarks
from ubo_morph.morphing import Backend, BackendName, get_backend
from ubo_morph.morphing.core import (
    _align_face_images,
    _delaunay_triangles,
    _equalize_face,
    _substitute_background,
)
from ubo_morph.morphing.points import (
    add_border_points,
    remove_border_points,
    remove_overlapped_points,
)
from ubo_morph.utils import ensure_bgr_uint8


DEFAULT_IMAGE1 = Path(r"C:\repos\ubo-morphing\face_1.png")
DEFAULT_IMAGE2 = Path(r"C:\repos\ubo-morphing\face_2.png")
DEFAULT_MODEL = Path(
    r"C:\repos\ubo-morphing-package\shape_predictor_68_face_landmarks.dat"
)
BACKENDS: tuple[BackendName, ...] = ("cpu", "cupy")
PERCENTILES = (5, 25, 75, 95)
Stats = dict[str, float]
Synchronize = Callable[[], None]


def _noop() -> None:
    pass


def _time_call(
    function: Callable[..., Any],
    *args: object,
    synchronize: Synchronize = _noop,
    **kwargs: object,
) -> tuple[Any, int]:
    synchronize()
    started = time.perf_counter_ns()
    result = function(*args, **kwargs)
    synchronize()
    return result, time.perf_counter_ns() - started


def _repeat(
    function: Callable[..., Any],
    repetitions: int,
    *args: object,
    synchronize: Synchronize = _noop,
    **kwargs: object,
) -> tuple[Any, list[int]]:
    result: Any = None
    durations: list[int] = []
    for _ in range(repetitions):
        result, duration = _time_call(
            function,
            *args,
            synchronize=synchronize,
            **kwargs,
        )
        durations.append(duration)
    return result, durations


def _stats(durations: list[int]) -> Stats:
    values = np.asarray(durations, dtype=np.float64) / 1_000_000.0
    percentiles = np.percentile(values, PERCENTILES)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        **{
            f"p{percentile:02d}": float(value)
            for percentile, value in zip(PERCENTILES, percentiles, strict=True)
        },
    }


def _profile_backend(
    backend: Backend[Any],
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    repetitions: int,
    *,
    synchronize: Synchronize = _noop,
) -> dict[str, Stats]:
    timings: dict[str, Stats] = {}

    def upload_images() -> tuple[Any, Any]:
        return backend.to_backend(image1), backend.to_backend(image2)

    (_, _), durations = _repeat(
        upload_images,
        repetitions,
        synchronize=synchronize,
    )
    timings["0. upload images"] = _stats(durations)
    backend_image1, backend_image2 = upload_images()

    def align_images() -> tuple[Any, Any, Landmarks, Landmarks]:
        return _align_face_images(
            backend,
            backend_image1,
            backend_image2,
            landmarks1,
            landmarks2,
        )

    _, durations = _repeat(
        align_images,
        repetitions,
        synchronize=synchronize,
    )
    timings["1. shared alignment"] = _stats(durations)
    aligned1, aligned2, aligned_landmarks1, aligned_landmarks2 = align_images()

    def prepare_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        equalization1, equalization2 = remove_overlapped_points(
            aligned_landmarks1.points,
            aligned_landmarks2.points,
        )
        morph1 = add_border_points(aligned1, aligned_landmarks1.points, 5)
        morph2 = add_border_points(aligned2, aligned_landmarks2.points, 5)
        morph1, morph2 = remove_overlapped_points(morph1, morph2)
        return equalization1, equalization2, morph1, morph2

    _, durations = _repeat(prepare_points, repetitions)
    timings["2. shared point cleanup"] = _stats(durations)
    equalization1, equalization2, morph_points1, morph_points2 = prepare_points()

    def equalize() -> Any:
        return _equalize_face(
            backend,
            aligned1,
            aligned2,
            equalization1,
            equalization2,
            method="color",
        )

    _, durations = _repeat(
        equalize,
        repetitions,
        synchronize=synchronize,
    )
    timings["3. shared equalization flow"] = _stats(durations)
    work1 = aligned1
    work2 = equalize()

    morphed_points = morph_points1 + (morph_points2 - morph_points1) * 0.5
    triangles, durations = _repeat(
        _delaunay_triangles,
        repetitions,
        morph_points1,
    )
    timings["4a. shared Delaunay"] = _stats(durations)

    warped1, durations = _repeat(
        backend.warp_triangles,
        repetitions,
        work1,
        morph_points1,
        morphed_points,
        triangles,
        synchronize=synchronize,
    )
    timings["4b. warp image 1"] = _stats(durations)
    warped2, durations = _repeat(
        backend.warp_triangles,
        repetitions,
        work2,
        morph_points2,
        morphed_points,
        triangles,
        synchronize=synchronize,
    )
    timings["4c. warp image 2"] = _stats(durations)

    blended, durations = _repeat(
        backend.blend,
        repetitions,
        warped1,
        warped2,
        0.5,
        synchronize=synchronize,
    )
    timings["5. backend blend"] = _stats(durations)

    face_points = remove_border_points(
        morphed_points,
        int(blended.shape[1]),
        int(blended.shape[0]),
    )
    eye_distance = float(
        np.linalg.norm(aligned_landmarks1.left_eye - aligned_landmarks1.right_eye)
    )
    result, durations = _repeat(
        _substitute_background,
        repetitions,
        backend,
        blended,
        face_points,
        warped1,
        blend=True,
        eye_distance=eye_distance,
        synchronize=synchronize,
    )
    timings["6. shared background flow"] = _stats(durations)

    _, durations = _repeat(
        backend.to_numpy,
        repetitions,
        result,
        synchronize=synchronize,
    )
    timings["7. download result"] = _stats(durations)
    return timings


def _print_table(timings: dict[str, dict[str, Stats]]) -> None:
    backend_width = max(10, *(len(name) for name in timings))
    stage_width = max(
        32,
        *(len(stage) for backend_timings in timings.values() for stage in backend_timings),
    )
    column_width = 12
    metric_names = ("mean", "std", "median", "p05", "p25", "p75", "p95")
    headers = ("backend", "stage", *metric_names, "speedup")
    print(
        f"{headers[0]:<{backend_width}}"
        f"{headers[1]:<{stage_width}}"
        + "".join(f"{header:>{column_width}}" for header in headers[2:])
    )
    print("-" * (backend_width + stage_width + column_width * (len(headers) - 2)))

    cpu_timings = timings.get("cpu", {})
    for backend_name, backend_timings in timings.items():
        for stage, stats in backend_timings.items():
            cpu_median = cpu_timings.get(stage, {}).get("median")
            median = stats["median"]
            speedup = (
                cpu_median / median
                if cpu_median is not None and median
                else float("nan")
            )
            values = [
                f"{backend_name:<{backend_width}}",
                f"{stage:<{stage_width}}",
                *(f"{stats[name]:>{column_width}.2f}" for name in metric_names),
                f"{speedup:>{column_width}.2f}",
            ]
            print("".join(values))


def _profile_configured_backends(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: Landmarks,
    landmarks2: Landmarks,
    repetitions: int,
) -> dict[str, dict[str, Stats]]:
    timings: dict[str, dict[str, Stats]] = {}
    for backend_name in BACKENDS:
        try:
            backend = get_backend(backend_name)
        except ImportError as error:
            print(f"{backend_name} timings skipped: {error}", file=sys.stderr)
            continue

        synchronize = _noop
        if backend_name != "cpu":
            import cupy as cp

            synchronize = cp.cuda.Stream.null.synchronize
        timings[backend_name] = _profile_backend(
            backend,
            image1,
            image2,
            landmarks1,
            landmarks2,
            repetitions,
            synchronize=synchronize,
        )
    return timings


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unable to load image file "{path}".')
    return ensure_bgr_uint8(image)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image1", type=Path, default=DEFAULT_IMAGE1)
    parser.add_argument("--image2", type=Path, default=DEFAULT_IMAGE2)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reps", type=int, default=3)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    image1 = _load_image(args.image1)
    image2 = _load_image(args.image2)
    with DlibLandmarkExtractor(args.model) as extractor:
        landmarks1 = extractor.extract(image1)
        landmarks2 = extractor.extract(image2)

    timings = _profile_configured_backends(
        image1,
        image2,
        landmarks1,
        landmarks2,
        args.reps,
    )
    _print_table(timings)


if __name__ == "__main__":
    main()
