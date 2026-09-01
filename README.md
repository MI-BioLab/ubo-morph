# UBO Morphing Algorithm

The official Python implementation of the face morphing algorithm described in
["Decoupling texture blending and shape warping in face morphing"](https://ieeexplore.ieee.org/abstract/document/8897253)
by Ferrara et al.

`ubo-morph` blends two aligned face photos into a single morphed image by warping
facial landmarks and blending textures independently, with optional automatic
retouching, color equalization, and background blending. It ships both a CLI for
batch processing and a Python API for programmatic use.

## Features

- Pluggable landmark extraction backends: [dlib](http://dlib.net/) (68-point) and
  [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- Pluggable compute backends: CPU (always available) and CuPy (CUDA-accelerated)
- Independent warping and blending factors for fine-grained control over shape vs. texture
- Automatic retouching: color/lightness equalization and feathered background blending
- Batch processing from CSV files, with support for multiple factor combinations per pair
- A typed, overloaded Python API returning either a plain image or a rich `MorphResult`
  with every intermediate step

## Installation

```bash
pip install ubo-morph
```

The base install only provides the morphing pipeline and the CPU backend. Landmark
extractors and the GPU backend are optional extras:

```bash
# dlib landmark extractor
pip install "ubo-morph[dlib]"

# MediaPipe landmark extractor
pip install "ubo-morph[mediapipe]"

# CUDA-accelerated backend (requires CUDA 12.x)
pip install "ubo-morph[cupy]"

# everything
pip install "ubo-morph[dlib,mediapipe,cupy]"
```

### Landmark models

Both landmark extractors require a model file that you download separately:

- **dlib**: `shape_predictor_68_face_landmarks.dat`, available from the
  [dlib-models repository](https://github.com/davisking/dlib-models).
- **MediaPipe**: `face_landmarker.task`, available from
  [MediaPipe's model index](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models).

## Quick start: CLI

Morph two images with the `ubo-morph` command, choosing a landmark extractor and
pointing it at the corresponding model file:

```bash
ubo-morph face_a.jpg face_b.jpg \
  --extractor mediapipe --model face_landmarker.task \
  --output-dir out/
```

This produces a single morphed image (50% warping, 50% blending by default) in
`out/`.

Use `--factor` to try one or more linked warping/blending values in a single run,
or `--warping-factor` / `--blending-factor` to sweep them independently (every
combination is generated):

```bash
ubo-morph face_a.jpg face_b.jpg \
  --extractor dlib --model shape_predictor_68_face_landmarks.dat \
  --warping-factor 0.3 0.5 0.7 --blending-factor 0.5 \
  --output-dir out/
```

### Batch processing with a CSV file

Pass a single CSV file instead of two images to process many pairs at once. The
CSV can be headerless (`image1, image2[, output]`), or use a header with any of
`image1/imagea`, `image2/imageb`, `output/filename`, and `factor` /
`warping_factor` + `blending_factor` columns:

```csv
image1,image2,factor
faces/a.jpg,faces/b.jpg,0.5
faces/c.jpg,faces/d.jpg,0.3
```

```bash
ubo-morph pairs.csv --extractor mediapipe --model face_landmarker.task \
  --output-dir out/ --skip-failing-pairs
```

### Useful flags

| Flag | Description |
| --- | --- |
| `--backend {cpu,cupy}` | Compute backend for the morphing pipeline (default: `cpu`) |
| `--align-eye-centers` / `--no-align-eye-centers` | Auto-align and resize images by eye position (default: on) |
| `--points-per-border` / `--no-border-points` | Number of image-edge points added per side (default: 5) |
| `--automatic-retouching` / `--no-automatic-retouching` | Enable retouching (color equalization + background blend) (default: on) |
| `--color-equalization` / `--no-color-equalization` | Toggle color/lightness equalization independently |
| `--equalization-method {color,lightness}` | Equalization method (default: `color`) |
| `--blend-background` / `--no-blend-background` | Feather-blend the substituted background (default: on) |
| `--intermediate-results` | Save every `MorphResult` image field to an `M_<stem1>_<stem2>` folder |
| `--landmark-extraction-short-side PIXELS` | Downscale images for landmark extraction only (0 = disabled) |

Run `ubo-morph --help` for the full list, including dlib (`--dlib-upsample-times`)
and MediaPipe (`--max-faces`, `--min-*-confidence`) specific options.

## Quick start: Python API

```python
import cv2
from ubo_morph import MediaPipeLandmarkExtractor, morph_images

image1 = cv2.imread("face_a.jpg")
image2 = cv2.imread("face_b.jpg")

with MediaPipeLandmarkExtractor("face_landmarker.task") as extractor:
    morphed = morph_images(
        image1,
        image2,
        extractor,
        warping_factor=0.5,
        blending_factor=0.5,
    )

cv2.imwrite("morphed.png", morphed)
```

`morph_images()` extracts landmarks for you. If you already have landmarks (e.g.
from your own extractor, or reused across multiple morphs), use
`morph_with_landmarks()` instead:

```python
from ubo_morph import DlibLandmarkExtractor, Landmarks, morph_with_landmarks

with DlibLandmarkExtractor("shape_predictor_68_face_landmarks.dat") as extractor:
    landmarks1 = extractor.extract(image1)
    landmarks2 = extractor.extract(image2)

morphed = morph_with_landmarks(image1, image2, landmarks1, landmarks2)
```

Both functions accept `return_details=True` to get a `MorphResult` with every
intermediate array (aligned/warped/equalized images, triangle mesh, point
correspondences) instead of just the final image:

```python
result = morph_images(image1, image2, extractor, return_details=True)
result.image             # final morphed image
result.warped_image1      # image1 warped onto the morphed shape
result.aligned_landmarks1  # landmarks after eye-center alignment
result.triangles          # Delaunay triangles used for warping
```

You can implement your own landmark source by subclassing `LandmarkExtractor` and
returning a `Landmarks` instance (points plus left/right eye centers) from
`_extract_points`.

## Backends

`ubo-morph` separates the morphing *algorithm* from the array backend that
executes it. Pass `backend=` to the Python API or `--backend` to the CLI:

| Backend | Value | Requirements | Notes |
| --- | --- | --- | --- |
| CPU | `cpu` | none (default) | NumPy/OpenCV based, always available |
| CuPy | `cupy` | `pip install "ubo-morph[cupy]"` + CUDA 12.x | Runs the pipeline on the GPU for faster batch processing |

```python
morphed = morph_images(image1, image2, extractor, backend="cupy")
```

If a backend's dependencies aren't installed, `get_backend()` (and the CLI) raise
a clear `ImportError` telling you which extra to install.

## Contributing

Contributions are welcome! This project uses [uv](https://docs.astral.sh/uv/) for
dependency management.

1. Clone the repository and install dependencies, including the dev group:

   ```bash
   uv sync --group dev
   ```

2. Make your changes, then run the same checks CI runs:

   ```bash
   uv run ruff check .        # lint
   uv run ty check src tests  # type check
   uv run pytest              # tests
   ```

3. Install the pre-commit hooks so commit messages are checked automatically:

   ```bash
   uv run pre-commit install
   ```

4. Commits must follow [Conventional Commits](https://www.conventionalcommits.org/)
   (enforced by [Commitizen](https://commitizen-tools.github.io/commitizen/)),
   since versioning and the changelog are generated from commit history via
   semantic-release. You can use `uv run cz commit` for a guided commit message.

5. Open a pull request. CI runs linting, type checking, and the test suite across
   Python 3.10–3.14.
