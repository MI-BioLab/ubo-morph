# ubo-morph

`ubo-morph` morphs two BGR `uint8` face images from corresponding landmark
points. Its compute-heavy operations expose a common `device: str` argument.
The supported values are exactly `"cpu"` and `"gpu"`.

The CPU path is implemented with OpenCV. The GPU path is deliberately left as
private Numba implementation hooks and currently raises `NotImplementedError`.
Numba is an optional dependency, so backend modules are loaded lazily only after
a device is selected.

## Public API

The high-level entry points are available directly from `ubo_morph`:

```python
from ubo_morph import morph_images, morph_with_landmarks

result = morph_images(image1, image2, extractor, device="cpu")
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

Use `--intermediate-results` to create `M_stem1_stem2/` containing `morphed.png`
and every image-valued intermediate field from `MorphResult`. Run
`ubo-morph --help` for all alignment, retouching, background, device, and
extractor-specific settings.

Both high-level functions forward the selected device through every heavy stage.
The lower-level device-aware operations are also public:

- `delaunay_triangles`
- `align_face_images`
- `warp_image_by_triangles`
- `blend_images`
- `equalize_face`
- `substitute_background`

Every function defaults to `device="cpu"`. Selecting `device="gpu"` reaches the
corresponding private GPU hook; no silent CPU fallback is performed.

## Module layout

```text
ubo_morph/
  morphing/
    _backends.py        # typed, lazy device selector
    _protocols.py       # backend callable contracts and bundle
    core.py             # high-level API, result type, and morph flow
    points.py           # lightweight point operations
    cpu/                # OpenCV implementations, including triangulation
    gpu/                # Numba implementation hooks, including triangulation
```

The backend subpackages are implementation details and are not re-exported from
the top-level package. To implement a GPU stage, fill in the matching module in
the relevant `gpu/` subpackage while preserving the CPU function's signature and
return contract. GPU modules may import Numba directly because the typed backend
selectors import them only for `device="gpu"`.

Install the optional dependency with:

```console
pip install "ubo-morph[gpu]"
```

## Validation

```console
uv run python -m unittest discover -v
uv run ruff check .
uv run ty check
```
