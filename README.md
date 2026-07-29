# ubo-morph

`ubo-morph` uses NumPy and OpenCV to morph two BGR `uint8` face images from
corresponding landmark points.

## Public API

The high-level entry points are available directly from `ubo_morph`:

```python
from ubo_morph import morph_images, morph_with_landmarks

result = morph_images(image1, image2, extractor)

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

Install CuPy support before selecting `backend="cupy"`:

```console
pip install "ubo-morph[cupy]"
```

## Landmark extractors

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

Install only the backend you need:

```console
pip install "ubo-morph[dlib]"
pip install "ubo-morph[mediapipe]"
```

## Command-line interface

Morph one pair directly with either landmark backend:

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task
ubo-morph first.jpg second.jpg --extractor dlib --model shape_predictor_68_face_landmarks.dat
```

Pass multiple linked factors with `--factor`; each value is used for both
warping and blending. Separate factor lists produce their Cartesian product:

```console
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task \
  --factor 0.25 0.50 0.75
ubo-morph first.jpg second.jpg --extractor mediapipe --model face_landmarker.task \
  --warping-factor 0.25 0.50 --blending-factor 0.50 0.75
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
`MorphResult`. Each original image remains unchanged and is accompanied by an
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
uv run ty check
```
