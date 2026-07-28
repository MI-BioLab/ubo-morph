from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import get_args
import cv2
import numpy as np
from tqdm import tqdm

from ubo_morph.landmarks import (
    DlibLandmarkExtractor,
    LandmarkExtractor,
    Landmarks,
    MediaPipeLandmarkExtractor,
)
from ubo_morph.morphing import (
    BackendName,
    MorphResult,
    get_backend,
    morph_with_landmarks,
)
from ubo_morph.utils import ensure_bgr_uint8, round_away


_IMAGE1_HEADERS = {"image1", "image_1", "imagea"}
_IMAGE2_HEADERS = {"image2", "image_2", "imageb"}
_OUTPUT_HEADERS = {"output", "output_filename", "filename"}
_FACTOR_HEADERS = {"factor", "warping_factor", "blending_factor"}
_KNOWN_HEADERS = _IMAGE1_HEADERS | _IMAGE2_HEADERS | _OUTPUT_HEADERS | _FACTOR_HEADERS
_BACKEND_CHOICES = get_args(BackendName)


class CliError(ValueError):
    """A user-facing CLI configuration or input error."""


class PairProcessingError(RuntimeError):
    """A processing failure associated with a specific input pair."""

    def __init__(self, image1: Path, image2: Path, cause: Exception) -> None:
        self.image1 = image1
        self.image2 = image2
        self.cause = cause
        super().__init__(str(self))

    def __str__(self) -> str:
        return f'pair "{self.image1}" and "{self.image2}" failed: {self.cause}'


class _ImageProcessingError(RuntimeError):
    def __init__(self, path: Path, operation: str, cause: Exception) -> None:
        self.path = path
        self.operation = operation
        self.cause = cause
        super().__init__(str(self))

    def __str__(self) -> str:
        return f'{self.operation} failed for image "{self.path}": {self.cause}'


@dataclass(frozen=True, slots=True)
class PairSpec:
    image1: Path
    image2: Path
    output_name: Path | None = None
    factors: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class MorphJob:
    image1: Path
    image2: Path
    output_path: Path
    warping_factor: float
    blending_factor: float
    intermediate_results: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ubo-morph",
        description="Morph two face images or batches of image pairs from CSV.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        metavar="INPUT",
        help="Either IMAGE1 IMAGE2 or one PAIRS.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for generated outputs (default: current directory).",
    )
    parser.add_argument(
        "--extractor",
        "--landmark-extractor",
        dest="extractor",
        choices=("mediapipe", "dlib"),
        required=True,
        help="Landmark extraction backend.",
    )
    parser.add_argument(
        "--model",
        "--landmark-model",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the extractor's model file.",
    )
    parser.add_argument(
        "--landmark-extraction-short-side",
        type=_landmark_extraction_short_side,
        default=0,
        metavar="PIXELS",
        help=(
            "Maximum shortest-side size for landmark extraction; "
            "0 disables resizing (default: 0)."
        ),
    )

    factor_group = parser.add_argument_group("morph factors")
    factor_group.add_argument(
        "--factor",
        nargs="+",
        type=_unit_interval,
        help="One or more linked factors, each used for both warping and blending.",
    )
    factor_group.add_argument(
        "--warping-factor",
        nargs="+",
        type=_unit_interval,
        help="One or more warping factors; combined with all blending factors.",
    )
    factor_group.add_argument(
        "--blending-factor",
        nargs="+",
        type=_unit_interval,
        help="One or more blending factors; combined with all warping factors.",
    )

    morph_group = parser.add_argument_group("morph settings")
    morph_group.add_argument(
        "--align-eye-centers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    morph_group.add_argument(
        "--points-per-border",
        type=_points_per_border,
        default=5,
        metavar="COUNT",
        help="Points per image edge, including corners; 0 disables them (default: 5).",
    )
    morph_group.add_argument(
        "--no-border-points",
        dest="points_per_border",
        action="store_const",
        const=0,
        help="Disable the addition of image-border points.",
    )
    morph_group.add_argument(
        "--automatic-retouching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    morph_group.add_argument(
        "--color-equalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    morph_group.add_argument(
        "--equalization-method",
        choices=("color", "lightness"),
        default="color",
    )
    morph_group.add_argument(
        "--blend-background",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    morph_group.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        default="cpu",
        help="Compute backend for morphing (default: cpu).",
    )
    morph_group.add_argument(
        "--intermediate-results",
        action="store_true",
        help="Save MorphResult image fields in an M_stem1_stem2 folder.",
    )
    morph_group.add_argument(
        "--skip-failing-pairs",
        action="store_true",
        help="Report failing image pairs and continue processing the remaining pairs.",
    )

    dlib_group = parser.add_argument_group("dlib settings")
    dlib_group.add_argument("--dlib-upsample-times", type=int, default=1)

    mediapipe_group = parser.add_argument_group("MediaPipe settings")
    mediapipe_group.add_argument("--max-faces", type=int, default=1)
    mediapipe_group.add_argument(
        "--min-face-detection-confidence",
        type=_unit_interval,
        default=0.5,
    )
    mediapipe_group.add_argument(
        "--min-face-presence-confidence",
        type=_unit_interval,
        default=0.5,
    )
    mediapipe_group.add_argument(
        "--min-tracking-confidence",
        type=_unit_interval,
        default=0.5,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        get_backend(args.backend)
    except (ImportError, NotImplementedError) as error:
        print(f"ubo-morph: error: {error}", file=sys.stderr)
        return 1
    try:
        cli_factors = _cli_factor_pairs(
            factor=args.factor,
            warping_factors=args.warping_factor,
            blending_factors=args.blending_factor,
        )
        cli_factors_were_given = any(
            value is not None
            for value in (args.factor, args.warping_factor, args.blending_factor)
        )

        if len(args.inputs) == 1 and args.inputs[0].suffix.lower() == ".csv":
            pairs, csv_has_factors = _read_pairs_csv(args.inputs[0])
            if csv_has_factors and cli_factors_were_given:
                raise CliError(
                    "CSV factor columns and CLI factor arguments cannot be used together; "
                    "only one factor mode is possible at a time."
                )
        elif len(args.inputs) == 2:
            pairs = [PairSpec(args.inputs[0], args.inputs[1])]
        else:
            raise CliError("pass either two image paths or one CSV path")

        jobs = _build_jobs(
            pairs,
            cli_factors=cli_factors,
            output_dir=args.output_dir,
            intermediate_results=args.intermediate_results,
        )
    except CliError as error:
        parser.error(str(error))

    try:
        _run_jobs(jobs, args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ubo-morph: error: {error}", file=sys.stderr)
        return 1
    return 0


def _cli_factor_pairs(
    *,
    factor: Sequence[float] | None,
    warping_factors: Sequence[float] | None,
    blending_factors: Sequence[float] | None,
) -> tuple[tuple[float, float], ...]:
    if factor is not None and (
        warping_factors is not None or blending_factors is not None
    ):
        raise CliError(
            "--factor cannot be combined with --warping-factor or --blending-factor"
        )
    if factor is not None:
        return tuple((value, value) for value in dict.fromkeys(factor))
    warping = tuple(dict.fromkeys(warping_factors or (0.5,)))
    blending = tuple(dict.fromkeys(blending_factors or (0.5,)))
    return tuple(product(warping, blending))


def _read_pairs_csv(path: Path) -> tuple[list[PairSpec], bool]:
    if not path.is_file():
        raise CliError(f'CSV file "{path}" does not exist')
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            numbered_rows = [
                (reader.line_num, [value.strip() for value in row])
                for row in reader
                if row and any(value.strip() for value in row)
            ]
    except (csv.Error, UnicodeError) as error:
        raise CliError(f'Unable to parse CSV file "{path}": {error}') from None
    except OSError as error:
        raise CliError(f'Unable to read CSV file "{path}": {error}') from None

    if not numbered_rows:
        raise CliError(f'CSV file "{path}" is empty')
    column_count = len(numbered_rows[0][1])
    rows: list[tuple[int, list[str]]] = []
    for row_number, row in numbered_rows:
        if len(row) > column_count:
            raise CliError(
                f'Unable to parse CSV file "{path}": expected {column_count} '
                f"columns but found {len(row)} (row {row_number})"
            )
        rows.append((row_number, [*row, *("" for _ in range(column_count - len(row)))]))

    if column_count < 2:
        raise CliError("CSV files must have at least two columns")

    header = [value.lower().replace(" ", "_").replace("-", "_") for value in rows[0][1]]
    has_header = any(name in _KNOWN_HEADERS for name in header)
    if not has_header:
        if column_count not in (2, 3):
            raise CliError(
                "headerless CSV files must have two image columns and an optional output column"
            )
        pairs = []
        for _, row in rows:
            if not row[0] or not row[1]:
                raise CliError("CSV image paths cannot be empty")
            image1 = _resolve_csv_path(path, row[0])
            image2 = _resolve_csv_path(path, row[1])
            output_name = None
            if column_count == 3:
                if not row[2]:
                    raise CliError("CSV output filenames cannot be empty")
                output_name = Path(row[2])
            pairs.append(PairSpec(image1, image2, output_name))
        return pairs, False

    header_counts = Counter(header)
    if any(count > 1 for count in header_counts.values()):
        duplicates = sorted(name for name, count in header_counts.items() if count > 1)
        raise CliError(f"CSV contains duplicate columns: {', '.join(duplicates)}")

    data_rows = rows[1:]
    if not data_rows:
        raise CliError("CSV header is present but no image pairs were provided")

    image1_columns = [name for name in header if name in _IMAGE1_HEADERS]
    image2_columns = [name for name in header if name in _IMAGE2_HEADERS]
    if len(image1_columns) > 1 or len(image2_columns) > 1:
        raise CliError("CSV contains duplicate image columns")
    image1_column = image1_columns[0] if image1_columns else header[0]
    image2_column = image2_columns[0] if image2_columns else header[1]
    if image1_column == image2_column:
        raise CliError("CSV image columns must be distinct")

    has_linked_factor = "factor" in header
    has_warping_factor = "warping_factor" in header
    has_blending_factor = "blending_factor" in header
    if has_linked_factor and (has_warping_factor or has_blending_factor):
        raise CliError(
            "CSV factor cannot be combined with warping_factor or blending_factor"
        )
    if has_warping_factor != has_blending_factor:
        raise CliError(
            "CSV must contain both warping_factor and blending_factor columns"
        )

    output_columns = [name for name in header if name in _OUTPUT_HEADERS]
    if len(output_columns) > 1:
        raise CliError("CSV contains more than one output filename column")
    output_column = output_columns[0] if output_columns else None

    reserved = {
        image1_column,
        image2_column,
        *(name for name in _FACTOR_HEADERS if name in header),
        *output_columns,
    }
    remaining = [name for name in header if name not in reserved]
    if output_column is None and len(remaining) == 1:
        output_column = remaining[0]
    elif remaining:
        names = ", ".join(repr(name) for name in remaining)
        raise CliError(f"unsupported CSV columns: {names}")

    factor_columns = (
        ("factor",)
        if has_linked_factor
        else (("warping_factor", "blending_factor") if has_warping_factor else ())
    )
    pairs = []
    for row_number, row in data_rows:
        values = dict(zip(header, row, strict=True))
        if not values[image1_column] or not values[image2_column]:
            raise CliError(f"CSV image paths cannot be empty (row {row_number})")

        image1 = _resolve_csv_path(path, values[image1_column])
        image2 = _resolve_csv_path(path, values[image2_column])

        output_name = None
        if output_column is not None:
            if not values[output_column]:
                raise CliError(
                    f"CSV output filenames cannot be empty (row {row_number})"
                )
            output_name = Path(values[output_column])

        parsed_factors = []
        for column in factor_columns:
            try:
                value = float(values[column])
            except ValueError:
                raise CliError(
                    f"CSV {column} value {values[column]!r} is not a number "
                    f"(row {row_number})"
                ) from None
            if not 0.0 <= value <= 1.0:
                raise CliError(f"CSV {column} must be in [0, 1] (row {row_number})")
            parsed_factors.append(value)

        factors = None
        if has_linked_factor:
            factors = ((parsed_factors[0], parsed_factors[0]),)
        elif parsed_factors:
            factors = ((parsed_factors[0], parsed_factors[1]),)
        pairs.append(PairSpec(image1, image2, output_name, factors))

    return pairs, bool(factor_columns)


def _resolve_csv_path(csv_path: Path, value: str) -> Path:
    image_path = Path(value)
    return image_path if image_path.is_absolute() else csv_path.parent / image_path


def _build_jobs(
    pairs: Sequence[PairSpec],
    *,
    cli_factors: Sequence[tuple[float, float]],
    output_dir: Path,
    intermediate_results: bool,
) -> list[MorphJob]:
    jobs: list[MorphJob] = []
    output_paths: set[Path] = set()
    for pair in pairs:
        factors = pair.factors or tuple(cli_factors)
        if (
            pair.output_name is not None
            and len(factors) > 1
            and not intermediate_results
        ):
            raise CliError(
                "a CSV output filename cannot be combined with multiple CLI factor combinations"
            )
        if intermediate_results and len(factors) > 1:
            raise CliError(
                "intermediate results require exactly one factor combination per image pair"
            )
        for warping_factor, blending_factor in factors:
            if intermediate_results:
                output_path = (
                    output_dir
                    / _default_output_name(
                        pair.image1,
                        pair.image2,
                        warping_factor=warping_factor,
                        blending_factor=blending_factor,
                    )
                    / "morphed.png"
                )
            elif pair.output_name is not None:
                output_path = (
                    pair.output_name
                    if pair.output_name.is_absolute()
                    else output_dir / pair.output_name
                )
            else:
                output_path = output_dir / _default_output_name(
                    pair.image1,
                    pair.image2,
                    warping_factor=warping_factor,
                    blending_factor=blending_factor,
                )
            collision_key = output_path.resolve(strict=False)
            if collision_key in output_paths:
                raise CliError(f'multiple jobs would write to "{output_path}"')
            output_paths.add(collision_key)
            jobs.append(
                MorphJob(
                    image1=pair.image1,
                    image2=pair.image2,
                    output_path=output_path,
                    warping_factor=warping_factor,
                    blending_factor=blending_factor,
                    intermediate_results=intermediate_results,
                )
            )
    return jobs


def _default_output_name(
    image1: Path,
    image2: Path,
    *,
    warping_factor: float,
    blending_factor: float,
) -> str:
    blending_percent = round_away(blending_factor * 100.0)
    warping_percent = round_away(warping_factor * 100.0)
    return (
        f"M_{image1.stem}_{image2.stem}_C05_B{blending_percent:02d}_"
        f"W{warping_percent:02d}_PA05_PM00_F00.png"
    )


def _run_jobs(jobs: Sequence[MorphJob], args: argparse.Namespace) -> None:
    extractor = _create_extractor(args)
    image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
    landmark_cache: dict[Path, Landmarks] = {}
    failed_pairs: set[tuple[Path, Path]] = set()
    current_pair: tuple[Path, Path] | None = None
    image1: np.ndarray | None = None
    image2: np.ndarray | None = None
    landmarks1: Landmarks | None = None
    landmarks2: Landmarks | None = None

    with extractor:
        progress = tqdm(
            jobs,
            desc="Morphing",
            unit="image",
            disable=len(jobs) <= 1,
        )
        for job in progress:
            pair_key = (job.image1, job.image2)
            if pair_key in failed_pairs:
                continue
            try:
                if pair_key != current_pair:
                    image1 = _cached_image(job.image1, image_cache)
                    image2 = _cached_image(job.image2, image_cache)
                    landmarks1 = _cached_landmarks(
                        job.image1,
                        image1,
                        extractor=extractor,
                        cache=landmark_cache,
                        max_short_side=args.landmark_extraction_short_side,
                    )
                    landmarks2 = _cached_landmarks(
                        job.image2,
                        image2,
                        extractor=extractor,
                        cache=landmark_cache,
                        max_short_side=args.landmark_extraction_short_side,
                    )
                    current_pair = pair_key
                assert image1 is not None and image2 is not None
                assert landmarks1 is not None and landmarks2 is not None

                result = morph_with_landmarks(
                    image1,
                    image2,
                    landmarks1,
                    landmarks2,
                    warping_factor=job.warping_factor,
                    blending_factor=job.blending_factor,
                    align_eye_centers=args.align_eye_centers,
                    points_per_border=args.points_per_border,
                    automatic_retouching=args.automatic_retouching,
                    color_equalization=args.color_equalization,
                    equalization_method=args.equalization_method,
                    blend_background=args.blend_background,
                    return_details=job.intermediate_results,
                    backend=args.backend,
                )
                if job.intermediate_results:
                    assert isinstance(result, MorphResult)
                    _save_intermediate_result(job.output_path, result)
                else:
                    assert isinstance(result, np.ndarray)
                    _save_image(job.output_path, result)
            except (OSError, ValueError, RuntimeError) as error:
                failure = PairProcessingError(job.image1, job.image2, error)
                if not args.skip_failing_pairs:
                    raise failure from error
                print(f"ubo-morph: skipping {failure}", file=sys.stderr)
                failed_pairs.add(pair_key)
                current_pair = None


def _cached_landmarks(
    path: Path,
    image: np.ndarray,
    *,
    extractor: LandmarkExtractor,
    cache: dict[Path, Landmarks],
    max_short_side: int,
) -> Landmarks:
    cache_key = path.resolve(strict=False)
    if cache_key not in cache:
        try:
            cache[cache_key] = extractor.extract(
                image,
                max_short_side=max_short_side,
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise _ImageProcessingError(path, "landmark extraction", error) from error
    return cache[cache_key]


def _cached_image(path: Path, cache: OrderedDict[Path, np.ndarray]) -> np.ndarray:
    cache_key = path.resolve(strict=False)
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]
    try:
        image = _load_image(path)
    except (OSError, ValueError, RuntimeError) as error:
        raise _ImageProcessingError(path, "loading", error) from error
    cache[cache_key] = image
    cache.move_to_end(cache_key)
    while len(cache) > 2:
        cache.popitem(last=False)
    return image


def _save_intermediate_result(output_path: Path, result: MorphResult) -> None:
    _save_image(output_path, result.image)
    for field in fields(result):
        if field.name == "image":
            continue
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3:
            _save_image(output_path.parent / f"{field.name}.png", value)


def _load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Image file "{path}" does not exist.')
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unable to load image file "{path}".')
    return image


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), ensure_bgr_uint8(image)):
        raise ValueError(f'Unable to save image file "{path}".')


def _create_extractor(args: argparse.Namespace) -> LandmarkExtractor:
    if args.extractor == "dlib":
        return DlibLandmarkExtractor(
            args.model_path,
            upsample_times=args.dlib_upsample_times,
        )
    return MediaPipeLandmarkExtractor(
        args.model_path,
        max_faces=args.max_faces,
        min_face_detection_confidence=args.min_face_detection_confidence,
        min_face_presence_confidence=args.min_face_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


def _unit_interval(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from None
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]")
    return number


def _points_per_border(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be 0 or an integer of at least 2"
        ) from None
    if count < 0 or count == 1:
        raise argparse.ArgumentTypeError("must be 0 or an integer of at least 2")
    return count


def _landmark_extraction_short_side(value: str) -> int:
    try:
        pixels = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from None
    if pixels < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return pixels


if __name__ == "__main__":
    raise SystemExit(main())
