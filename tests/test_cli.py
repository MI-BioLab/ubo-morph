from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import get_args
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from ubo_morph import Landmarks, MorphResult
from ubo_morph.morphing import BackendName
from ubo_morph.cli import (
    CliError,
    PairSpec,
    _build_jobs,
    _cli_factor_pairs,
    _default_output_name,
    _read_pairs_csv,
    build_parser,
    main,
)


class TestCliConfiguration:
    def test_backend_choices_come_from_public_alias_and_default_to_cpu(self) -> None:
        parser = build_parser()
        required = [
            "first.png",
            "second.png",
            "--extractor",
            "dlib",
            "--model",
            "predictor.dat",
        ]
        backend_action = next(
            action for action in parser._actions if action.dest == "backend"
        )

        assert tuple(backend_action.choices or ()) == get_args(BackendName)
        assert parser.parse_args(required).backend == "cpu"

    @pytest.mark.parametrize("value", ["-1", "1.5"])
    def test_landmark_extraction_short_side_cli_validation(self, value: str) -> None:
        parser = build_parser()
        required = [
            "first.png",
            "second.png",
            "--extractor",
            "dlib",
            "--model",
            "predictor.dat",
        ]

        assert parser.parse_args(required).landmark_extraction_short_side == 0
        assert (
            parser.parse_args(
                [*required, "--landmark-extraction-short-side", "640"]
            ).landmark_extraction_short_side
            == 640
        )
        with redirect_stderr(StringIO()), pytest.raises(SystemExit):
            parser.parse_args([*required, "--landmark-extraction-short-side", value])

    def test_points_per_border_cli_default_custom_disable_and_validation(self) -> None:
        parser = build_parser()
        required = [
            "first.png",
            "second.png",
            "--extractor",
            "dlib",
            "--model",
            "predictor.dat",
        ]

        assert parser.parse_args(required).points_per_border == 5
        assert (
            parser.parse_args([*required, "--points-per-border", "7"]).points_per_border
            == 7
        )
        assert (
            parser.parse_args([*required, "--no-border-points"]).points_per_border == 0
        )
        with redirect_stderr(StringIO()), pytest.raises(SystemExit):
            parser.parse_args([*required, "--points-per-border", "1"])

    def test_default_filename_contains_both_factor_percentages(self) -> None:
        filename = _default_output_name(
            Path("first.jpg"),
            Path("second.png"),
            warping_factor=0.25,
            blending_factor=0.75,
        )

        assert filename == "M_first_second_C05_B75_W25_PA05_PM00_F00.png"

    def test_linked_and_separate_cli_factor_modes(self) -> None:
        linked = _cli_factor_pairs(
            factor=(0.2, 0.6),
            warping_factors=None,
            blending_factors=None,
        )
        separate = _cli_factor_pairs(
            factor=None,
            warping_factors=(0.2, 0.4),
            blending_factors=(0.3, 0.7),
        )

        assert linked == ((0.2, 0.2), (0.6, 0.6))
        assert separate == (
            (0.2, 0.3),
            (0.2, 0.7),
            (0.4, 0.3),
            (0.4, 0.7),
        )
        with pytest.raises(CliError, match="cannot be combined"):
            _cli_factor_pairs(
                factor=(0.5,),
                warping_factors=(0.5,),
                blending_factors=None,
            )

    def test_headerless_csv_supports_default_and_explicit_output_names(
        self,
        tmp_path: Path,
    ) -> None:
        two_columns = tmp_path / "pairs.csv"
        two_columns.write_text("first.png,second.png\n", encoding="utf-8")
        three_columns = tmp_path / "named.csv"
        three_columns.write_text(
            "first.png,second.png,custom.png\n",
            encoding="utf-8",
        )

        default_pairs, default_has_factors = _read_pairs_csv(two_columns)
        named_pairs, named_has_factors = _read_pairs_csv(three_columns)

        assert not default_has_factors
        assert not named_has_factors
        assert default_pairs[0].image1 == tmp_path / "first.png"
        assert default_pairs[0].output_name is None
        assert named_pairs[0].output_name == Path("custom.png")

    def test_csv_supports_linked_and_separate_factor_columns(
        self,
        tmp_path: Path,
    ) -> None:
        linked_csv = tmp_path / "linked.csv"
        linked_csv.write_text(
            "image1,image2,factor\nfirst.png,second.png,0.3\n",
            encoding="utf-8",
        )
        separate_csv = tmp_path / "separate.csv"
        separate_csv.write_text(
            "image1,image2,output,warping_factor,blending_factor\n"
            "first.png,second.png,result.png,0.2,0.8\n",
            encoding="utf-8",
        )

        linked_pairs, linked_has_factors = _read_pairs_csv(linked_csv)
        separate_pairs, separate_has_factors = _read_pairs_csv(separate_csv)

        assert linked_has_factors
        assert separate_has_factors
        assert linked_pairs[0].factors == ((0.3, 0.3),)
        assert separate_pairs[0].factors == ((0.2, 0.8),)
        assert separate_pairs[0].output_name == Path("result.png")

    def test_csv_uses_standard_quoting(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            'image1,image2,output\n"first,variant.png",second.png,"result,1.png"\n',
            encoding="utf-8",
        )

        pairs, has_factors = _read_pairs_csv(csv_path)

        assert not has_factors
        assert pairs[0].image1 == tmp_path / "first,variant.png"
        assert pairs[0].output_name == Path("result,1.png")

    def test_csv_supports_utf8_bom_and_legacy_output_header_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            "\ufeffimage1,image2,custom_name\nfirst.png,second.png,result.png\n",
            encoding="utf-8",
        )

        pairs, has_factors = _read_pairs_csv(csv_path)

        assert pairs[0].output_name == Path("result.png")
        assert not has_factors

    def test_csv_rejects_multiple_unsupported_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            "image1,image2,unknown1,unknown2\nfirst.png,second.png,a,b\n",
            encoding="utf-8",
        )

        with pytest.raises(CliError, match="unsupported CSV columns"):
            _read_pairs_csv(csv_path)

    def test_csv_rejects_malformed_quoting(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            'image1,image2\n"unterminated,second.png\n',
            encoding="utf-8",
        )

        with pytest.raises(CliError, match="Unable to parse CSV"):
            _read_pairs_csv(csv_path)

    def test_csv_and_cli_factor_modes_are_mutually_exclusive(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            "image1,image2,factor\nfirst.png,second.png,0.3\n",
            encoding="utf-8",
        )
        with redirect_stderr(StringIO()), pytest.raises(SystemExit) as raised:
            main(
                [
                    str(csv_path),
                    "--extractor",
                    "dlib",
                    "--model",
                    "predictor.dat",
                    "--factor",
                    "0.5",
                ]
            )

        assert raised.value.code == 2

    def test_custom_output_rejects_multiple_cli_factor_combinations(self) -> None:
        with pytest.raises(CliError, match="output filename"):
            _build_jobs(
                [PairSpec(Path("first.png"), Path("second.png"), Path("out.png"))],
                cli_factors=((0.2, 0.2), (0.5, 0.5)),
                output_dir=Path("outputs"),
                intermediate_results=False,
            )


class TestCliExecution:
    def test_backend_failure_is_global_early_and_never_skipped(self) -> None:
        arguments = [
            "first.png",
            "second.png",
            "--extractor",
            "dlib",
            "--model",
            "predictor.dat",
            "--backend",
            "cupy",
            "--skip-failing-pairs",
        ]
        stderr = StringIO()
        with (
            patch(
                "ubo_morph.cli.get_backend",
                side_effect=ImportError("install ubo-morph[cupy]"),
            ) as get_backend,
            patch("ubo_morph.cli._create_extractor") as create_extractor,
            patch("ubo_morph.cli._load_image") as load_image,
            patch("ubo_morph.cli.morph_with_landmarks") as morph,
            redirect_stderr(stderr),
        ):
            exit_code = main(arguments)

        assert exit_code == 1
        get_backend.assert_called_once_with("cupy")
        create_extractor.assert_not_called()
        load_image.assert_not_called()
        morph.assert_not_called()
        assert stderr.getvalue().count("install ubo-morph[cupy]") == 1
        assert "skipping" not in stderr.getvalue()

    @pytest.mark.parametrize(
        ("skip", "expected_exit_code", "expected_morph_count"),
        [(False, 1, 1), (True, 0, 2)],
    )
    def test_skip_failing_pairs_reports_files_and_continues(
        self,
        tmp_path: Path,
        skip: bool,
        expected_exit_code: int,
        expected_morph_count: int,
    ) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        points = np.array([[0, 0], [3, 0], [0, 3]], dtype=np.float32)
        landmarks = Landmarks(
            np.array((3, 1), dtype=np.float32),
            np.array((1, 1), dtype=np.float32),
            points,
        )

        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            "image1,image2\n"
            "first.png,second.png\n"
            "bad.png,third.png\n"
            "first.png,third.png\n",
            encoding="utf-8",
        )
        extractor = MagicMock()
        extractor.__enter__.return_value = extractor
        extractor.extract.return_value = landmarks

        def load_image(path: Path) -> np.ndarray:
            if path.name == "bad.png":
                raise ValueError("broken image data")
            return image

        arguments = [
            str(csv_path),
            "--extractor",
            "dlib",
            "--model",
            "predictor.dat",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
        if skip:
            arguments.append("--skip-failing-pairs")

        stderr = StringIO()
        with (
            patch("ubo_morph.cli._create_extractor", return_value=extractor),
            patch("ubo_morph.cli._load_image", side_effect=load_image),
            patch("ubo_morph.cli._save_image"),
            patch(
                "ubo_morph.cli.morph_with_landmarks",
                side_effect=lambda image1, *args, **kwargs: image1,
            ) as morph,
            patch(
                "ubo_morph.cli.tqdm",
                side_effect=lambda values, **kwargs: values,
            ),
            redirect_stderr(stderr),
        ):
            exit_code = main(arguments)

        message = stderr.getvalue()
        assert exit_code == expected_exit_code
        assert morph.call_count == expected_morph_count
        assert "bad.png" in message
        assert "third.png" in message
        assert "loading failed" in message
        assert "broken image data" in message
        if skip:
            assert "skipping pair" in message
        else:
            assert "error: pair" in message

    def test_landmarks_are_cached_for_repeated_images(self, tmp_path: Path) -> None:
        points = np.array([[0, 0], [3, 0], [0, 3]], dtype=np.float32)
        images = {
            name: np.full((4, 4, 3), value, dtype=np.uint8)
            for name, value in (
                ("first.png", 10),
                ("second.png", 20),
                ("third.png", 30),
            )
        }
        landmarks = {
            name: Landmarks(
                np.array((3, 1), dtype=np.float32),
                np.array((1, 1), dtype=np.float32),
                points + offset,
            )
            for offset, name in enumerate(images)
        }
        extractor = MagicMock()
        extractor.__enter__.return_value = extractor
        extractor.extract.side_effect = lambda image, **kwargs: next(
            landmarks[name] for name, value in images.items() if value is image
        )

        csv_path = tmp_path / "pairs.csv"
        csv_path.write_text(
            "image1,image2\nfirst.png,second.png\nfirst.png,third.png\n",
            encoding="utf-8",
        )
        with (
            patch("ubo_morph.cli._create_extractor", return_value=extractor),
            patch(
                "ubo_morph.cli._load_image",
                side_effect=lambda path: images[path.name],
            ) as load_image,
            patch("ubo_morph.cli._save_image"),
            patch(
                "ubo_morph.cli.morph_with_landmarks",
                side_effect=lambda image1, *args, **kwargs: image1,
            ) as morph,
            patch("ubo_morph.cli.tqdm", side_effect=lambda values, **kwargs: values),
        ):
            exit_code = main(
                [
                    str(csv_path),
                    "--extractor",
                    "dlib",
                    "--model",
                    "predictor.dat",
                    "--output-dir",
                    str(tmp_path / "outputs"),
                ]
            )

        assert exit_code == 0
        assert load_image.call_count == 3
        assert extractor.extract.call_count == 3
        assert morph.call_count == 2
        first_call, second_call = morph.call_args_list
        assert first_call.args[2] is landmarks["first.png"]
        assert first_call.args[3] is landmarks["second.png"]
        assert second_call.args[2] is landmarks["first.png"]
        assert second_call.args[3] is landmarks["third.png"]

    def test_intermediate_mode_saves_all_morph_result_images(
        self,
        tmp_path: Path,
    ) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.array([[20, 60], [30, 60], [20, 70]], dtype=np.float32)
        source_points1 = np.vstack((points, np.array([[0, 0]], dtype=np.float32)))
        source_points2 = source_points1 + np.array([30, 0], dtype=np.float32)
        source_points2[-1] = [99, 0]
        morphed_points = source_points1 + np.array([60, 0], dtype=np.float32)
        morphed_points[-1] = [50, 0]
        triangles = [(0, 1, 2), (0, 2, 3), (0, 3, 1)]
        landmarks = Landmarks(
            np.array((30, 60), dtype=np.float32),
            np.array((20, 60), dtype=np.float32),
            points,
        )
        result = MorphResult(
            image=image,
            morphed_points=morphed_points,
            source_points1=source_points1,
            source_points2=source_points2,
            point_landmark_indices=np.array([0, 1, 2, -1], dtype=np.int32),
            triangles=triangles,
            warped_image1=image,
            warped_image2=image,
            aligned_image1=image,
            aligned_image2=image,
            aligned_landmarks1=landmarks,
            aligned_landmarks2=landmarks,
            original_landmarks1=landmarks,
            original_landmarks2=landmarks,
            before_background_substitution=image,
            after_equalization_image1=image,
            after_equalization_image2=image,
        )
        extractor = MagicMock()
        extractor.__enter__.return_value = extractor
        extractor.extract.side_effect = (landmarks, landmarks)
        saved_images: dict[Path, np.ndarray] = {}

        output_directory = tmp_path / "outputs"
        with (
            patch("ubo_morph.cli._create_extractor", return_value=extractor),
            patch("ubo_morph.cli._load_image", return_value=image),
            patch(
                "ubo_morph.cli._save_image",
                side_effect=lambda path, value: saved_images.__setitem__(
                    path,
                    value.copy(),
                ),
            ),
            patch("ubo_morph.cli.morph_with_landmarks", return_value=result) as morph,
            patch("ubo_morph.cli.tqdm", side_effect=lambda values, **kwargs: values),
        ):
            exit_code = main(
                [
                    "first.jpg",
                    "second.jpg",
                    "--extractor",
                    "dlib",
                    "--model",
                    "predictor.dat",
                    "--output-dir",
                    str(output_directory),
                    "--landmark-extraction-short-side",
                    "640",
                    "--warping-factor",
                    "0.2",
                    "--blending-factor",
                    "0.8",
                    "--no-align-eye-centers",
                    "--no-border-points",
                    "--no-automatic-retouching",
                    "--no-color-equalization",
                    "--equalization-method",
                    "lightness",
                    "--no-blend-background",
                    "--intermediate-results",
                ]
            )

        expected_directory = output_directory / _default_output_name(
            Path("first.jpg"),
            Path("second.jpg"),
            warping_factor=0.2,
            blending_factor=0.8,
        )

        original_names = {
            "morphed.png",
            "warped_image1.png",
            "warped_image2.png",
            "aligned_image1.png",
            "aligned_image2.png",
            "before_background_substitution.png",
            "after_equalization_image1.png",
            "after_equalization_image2.png",
        }
        annotated_names = {
            f"{Path(name).stem}_annotated.png" for name in original_names
        }
        assert exit_code == 0
        assert set(saved_images) == {
            expected_directory / name for name in original_names | annotated_names
        }
        for name in original_names:
            np.testing.assert_array_equal(saved_images[expected_directory / name], image)

        source1_names = {"aligned_image1", "after_equalization_image1"}
        source2_names = {"aligned_image2", "after_equalization_image2"}
        morphed_names = {
            "morphed",
            "warped_image1",
            "warped_image2",
            "before_background_substitution",
        }
        for name in source1_names:
            annotated = saved_images[expected_directory / f"{name}_annotated.png"]
            assert annotated[60, 20].any()
            assert annotated[65, 25].any()
            assert annotated[0, 0].any()
            assert not annotated[60, 50].any()
            assert not annotated[60, 80].any()
            assert not np.all(annotated[:15, :15] == 255, axis=2).any()
        for name in source2_names:
            annotated = saved_images[expected_directory / f"{name}_annotated.png"]
            assert not annotated[60, 20].any()
            assert annotated[60, 50].any()
            assert annotated[65, 55].any()
            assert annotated[0, 99].any()
            assert not annotated[60, 80].any()
            assert not np.all(annotated[:15, 85:] == 255, axis=2).any()
        for name in morphed_names:
            annotated = saved_images[expected_directory / f"{name}_annotated.png"]
            assert not annotated[60, 20].any()
            assert not annotated[60, 50].any()
            assert annotated[60, 80].any()
            assert annotated[65, 85].any()
            assert annotated[0, 50].any()
            assert not np.all(annotated[:15, 45:65] == 255, axis=2).any()

        kwargs = morph.call_args.kwargs
        assert kwargs["warping_factor"] == 0.2
        assert kwargs["blending_factor"] == 0.8
        assert not kwargs["align_eye_centers"]
        assert kwargs["points_per_border"] == 0
        assert not kwargs["automatic_retouching"]
        assert not kwargs["color_equalization"]
        assert kwargs["equalization_method"] == "lightness"
        assert not kwargs["blend_background"]
        assert kwargs["return_details"]
        assert extractor.extract.call_args_list == [
            call(image, max_short_side=640),
            call(image, max_short_side=640),
        ]
