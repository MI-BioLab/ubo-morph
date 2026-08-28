# UBO Morphing Algorithm

This package contains the reference implementation in Python of the UBO landmark-based morphing algorithm
described in the paper "Decoupling texture blending and shape warping in face mopring" by M. Ferrara et al.

## Quick start

The examples below use MediaPipe and its `face_landmarker.task` model. Put the
model and two face images in the working directory:

```console
pip install "ubo-morph[mediapipe]"
```

### Morph from the command line

Run one morph with a shape-warping factor and a texture-blending factor of
`0.5` (the defaults):

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task --output-dir output
```

The resulting PNG is written under `output/` with an `M_...png` filename.
Use `--factor` to produce several linked shape and texture blends in one run:

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task --output-dir output --factor 0.25 0.50 0.75
```

Inspect the landmark mesh and pipeline images for one factor combination with:

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task --output-dir output --intermediate-results
```

This creates a factor-qualified directory containing `morphed.png`, its
`morphed_annotated.png` landmark-mesh visualization, and the available
intermediate images.

### Morph from Python

OpenCV reads and writes BGR images, matching the package API. This example
writes the default midpoint morph to `output.png`:

```python
import cv2

from ubo_morph import MediaPipeLandmarkExtractor, morph_images

image1 = cv2.imread("first.jpg", cv2.IMREAD_COLOR)
image2 = cv2.imread("second.jpg", cv2.IMREAD_COLOR)
if image1 is None or image2 is None:
    raise FileNotFoundError("Could not read first.jpg or second.jpg")

with MediaPipeLandmarkExtractor("face_landmarker.task") as extractor:
    result = morph_images(image1, image2, extractor)

if not cv2.imwrite("output.png", result):
    raise OSError("Could not write output.png")
```

Set `warping_factor` and `blending_factor` independently when the facial shape
and texture should progress at different rates:

```python
with MediaPipeLandmarkExtractor("face_landmarker.task") as extractor:
    result = morph_images(
        image1,
        image2,
        extractor,
        warping_factor=0.25,
        blending_factor=0.75,
    )
```

## Python API

The high-level entry points and both landmark extractors are available directly
from `ubo_morph`. Choose one extractor and provide its compatible model file:

```python
from ubo_morph import (
    DlibLandmarkExtractor,
    MediaPipeLandmarkExtractor,
    morph_images,
    morph_with_landmarks,
)

# MediaPipe returns its face-landmarker mesh.
with MediaPipeLandmarkExtractor("face_landmarker.task") as extractor:
    result = morph_images(image1, image2, extractor)

# Dlib is an alternative extractor that requires a 68-point shape predictor.
with DlibLandmarkExtractor("shape_predictor_68_face_landmarks.dat") as extractor:
    result = morph_images(image1, image2, extractor)
```

Once an extractor is selected, pass it to `morph_images` with any pipeline
options:

```python
with MediaPipeLandmarkExtractor("face_landmarker.task") as extractor:
    # Control the number of points added to each image edge, or disable them.
    result = morph_images(image1, image2, extractor, points_per_border=7)
    result = morph_images(image1, image2, extractor, points_per_border=0)

    # Cap the shortest detector-input side at 640 px, then morph full-size inputs.
    result = morph_images(
        image1,
        image2,
        extractor,
        landmark_extraction_short_side=640,
    )

    # Select one of the exact backend names: "cpu" or "cupy".
    result = morph_images(image1, image2, extractor, backend="cupy")
```

`cpu` is the default. Backend selection is explicit: unavailable accelerators
raise an error and never fall back to CPU.

## Backend interface

The backend contract and lazy selector are available from `ubo_morph.morphing`:

```python
from ubo_morph.morphing import Backend, BackendName, get_backend

cpu = get_backend("cpu")
assert cpu.name == "cpu"
```

Concrete classes are exposed only by their backend subpackages:

```python
from ubo_morph.morphing.cpu import CPUBackend

backend = CPUBackend()
```

Install CuPy support together with at least one landmark extractor before
selecting `backend="cupy"`.

## Extractors and backends

Concrete dlib and MediaPipe extractors are available from the top-level package.
Both require a compatible model file supplied by the caller and select the
largest detected face when multiple faces are returned.

```python
from ubo_morph import DlibLandmarkExtractor, MediaPipeLandmarkExtractor

with DlibLandmarkExtractor("shape_predictor_68_face_landmarks.dat") as extractor:
    dlib_result = morph_images(image1, image2, extractor)

with MediaPipeLandmarkExtractor("face_landmarker.task") as extractor:
    mediapipe_result = morph_images(image1, image2, extractor)
```

Install dlib instead of MediaPipe with:

```console
pip install "ubo-morph[dlib]"
```

## CLI reference

Morph one pair directly with either landmark backend:

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task
ubo-morph first.jpg second.jpg --extractor dlib --model shape_predictor_68_face_landmarks.dat
```

Pass multiple linked factors with `--factor`; each value is used for both
warping and blending. Separate factor lists produce their Cartesian product:

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task --factor 0.25 0.50 0.75
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task --warping-factor 0.25 0.50 --blending-factor 0.50 0.75
```

Batch input is passed as a positional CSV path:

```console
ubo-morph pairs.csv --extractor mediapipe --model face_landmarker.task
```

Headerless CSV files accept two image columns and an optional third output
filename column. Headered CSV files may use `factor`, or both `warping_factor`
and `blending_factor`; an optional `output`, `output_filename`, or `filename`
column controls the destination name. Relative image paths are resolved from the
CSV directory. CSV factor columns cannot be combined with CLI factor arguments.
During each CLI run, landmarks are cached in memory by resolved image path, so
images reused across pairs are extracted only once. The cache is discarded when
the command exits.

By default, a failing pair stops the command. Pass `--skip-failing-pairs` to
report the affected file or files and failure reason, then continue with the
remaining pairs.

Use `--intermediate-results` to create a factor-qualified `M_...png/` directory
containing `morphed.png` and every image-valued intermediate field from
`MorphResult`. Every saved image, including `morphed.png`, has an accompanying
`_annotated.png` version containing indexed facial landmarks, unindexed border
points, and the Delaunay triangulation used by the morph. The intermediate
images include the aligned images before color equalization, the image actually
changed by equalization when it runs, the warped images, and the blended image
when background substitution follows. `MorphResult` also exposes the original
and aligned landmarks for both inputs. Use `--points-per-border COUNT` to change
the default of five; zero disables border points, as does the
`--no-border-points` convenience flag. Run `ubo-morph --help` for all alignment,
retouching, background, and extractor-specific settings.

For faster landmark detection on large inputs, set
`--landmark-extraction-short-side PIXELS`. Images whose shortest side exceeds
that limit are resized so it equals the limit, with the other side scaled
proportionally. Only the extractor input is resized; detected coordinates are
mapped back to the original image size before full-resolution morphing. The
default value of zero disables this resizing.

## Module layout

```text
ubo_morph/
  morphing/
    backend.py          # typed backend contract and lazy selector
    core.py             # shared geometry, retouching, triangulation, and flow
    points.py           # shared point and mask operations
    cpu/backend.py      # NumPy/OpenCV primitive implementation
    cupy/backend.py     # optional CuPy primitive implementation
```

## Validation

```console
uv run pytest -v
uv run ruff check .
uv run ty check src tests
```

## Contributing

Contributions are welcome. Create a focused branch, add or update tests for
behavior changes, and run the validation commands above before opening a pull
request. Keep changes scoped to the issue being addressed and include a clear
description of the behavior change in the pull request. Commit messages must
follow the Conventional Commits specification, for example `feat: add GPU
batching` or `fix(cli): report invalid input`. To validate messages before each
commit, install the optional hook:

```console
uv sync --group dev
uv run pre-commit install --hook-type commit-msg
```
