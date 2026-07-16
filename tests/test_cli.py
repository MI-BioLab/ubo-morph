from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ubo_morph import Landmarks, MorphResult
from ubo_morph.cli import (
    CliError,
    PairSpec,
    _build_jobs,
    _cli_factor_pairs,
    _default_output_name,
    _read_pairs_csv,
    main,
)


class CliConfigurationTests(unittest.TestCase):
    def test_default_filename_contains_both_factor_percentages(self) -> None:
        filename = _default_output_name(
            Path("first.jpg"),
            Path("second.png"),
            warping_factor=0.25,
            blending_factor=0.75,
        )

        self.assertEqual(
            filename,
            "M_first_second_C05_B75_W25_PA05_PM00_F00.png",
        )

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

        self.assertEqual(linked, ((0.2, 0.2), (0.6, 0.6)))
        self.assertEqual(
            separate,
            ((0.2, 0.3), (0.2, 0.7), (0.4, 0.3), (0.4, 0.7)),
        )
        with self.assertRaisesRegex(CliError, "cannot be combined"):
            _cli_factor_pairs(
                factor=(0.5,),
                warping_factors=(0.5,),
                blending_factors=None,
            )

    def test_headerless_csv_supports_default_and_explicit_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_directory = Path(directory)
            two_columns = csv_directory / "pairs.csv"
            two_columns.write_text("first.png,second.png\n", encoding="utf-8")
            three_columns = csv_directory / "named.csv"
            three_columns.write_text(
                "first.png,second.png,custom.png\n",
                encoding="utf-8",
            )

            default_pairs, default_has_factors = _read_pairs_csv(two_columns)
            named_pairs, named_has_factors = _read_pairs_csv(three_columns)

        self.assertFalse(default_has_factors)
        self.assertFalse(named_has_factors)
        self.assertEqual(default_pairs[0].image1, csv_directory / "first.png")
        self.assertIsNone(default_pairs[0].output_name)
        self.assertEqual(named_pairs[0].output_name, Path("custom.png"))

    def test_csv_supports_linked_and_separate_factor_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_directory = Path(directory)
            linked_csv = csv_directory / "linked.csv"
            linked_csv.write_text(
                "image1,image2,factor\nfirst.png,second.png,0.3\n",
                encoding="utf-8",
            )
            separate_csv = csv_directory / "separate.csv"
            separate_csv.write_text(
                "image1,image2,output,warping_factor,blending_factor\n"
                "first.png,second.png,result.png,0.2,0.8\n",
                encoding="utf-8",
            )

            linked_pairs, linked_has_factors = _read_pairs_csv(linked_csv)
            separate_pairs, separate_has_factors = _read_pairs_csv(separate_csv)

        self.assertTrue(linked_has_factors)
        self.assertTrue(separate_has_factors)
        self.assertEqual(linked_pairs[0].factors, ((0.3, 0.3),))
        self.assertEqual(separate_pairs[0].factors, ((0.2, 0.8),))
        self.assertEqual(separate_pairs[0].output_name, Path("result.png"))

    def test_csv_uses_standard_quoting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory, "pairs.csv")
            csv_path.write_text(
                'image1,image2,output\n"first,variant.png",second.png,"result,1.png"\n',
                encoding="utf-8",
            )

            pairs, has_factors = _read_pairs_csv(csv_path)

        self.assertFalse(has_factors)
        self.assertEqual(pairs[0].image1, Path(directory, "first,variant.png"))
        self.assertEqual(pairs[0].output_name, Path("result,1.png"))

    def test_csv_and_cli_factor_modes_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory, "pairs.csv")
            csv_path.write_text(
                "image1,image2,factor\nfirst.png,second.png,0.3\n",
                encoding="utf-8",
            )
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
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

        self.assertEqual(raised.exception.code, 2)

    def test_custom_output_rejects_multiple_cli_factor_combinations(self) -> None:
        with self.assertRaisesRegex(CliError, "output filename"):
            _build_jobs(
                [PairSpec(Path("first.png"), Path("second.png"), Path("out.png"))],
                cli_factors=((0.2, 0.2), (0.5, 0.5)),
                output_dir=Path("outputs"),
                intermediate_results=False,
            )


class CliExecutionTests(unittest.TestCase):
    def test_intermediate_mode_saves_all_morph_result_images(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        points = np.array([[0, 0], [3, 0], [0, 3]], dtype=np.float32)
        landmarks = Landmarks(
            np.array((3, 1), dtype=np.float32),
            np.array((1, 1), dtype=np.float32),
            points,
        )
        result = MorphResult(
            image=image,
            morphed_points=points,
            warped_image1=image,
            warped_image2=image,
            aligned_image1=image,
            aligned_image2=image,
            aligned_landmarks1=landmarks,
            aligned_landmarks2=landmarks,
        )
        extractor = MagicMock()
        extractor.__enter__.return_value = extractor
        extractor.extract.side_effect = (landmarks, landmarks)
        saved_paths: list[Path] = []

        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory, "outputs")
            with (
                patch("ubo_morph.cli._create_extractor", return_value=extractor),
                patch("ubo_morph.cli._load_image", return_value=image),
                patch("ubo_morph.cli._save_image", side_effect=lambda path, value: saved_paths.append(path)),
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
                        "--warping-factor",
                        "0.2",
                        "--blending-factor",
                        "0.8",
                        "--no-align-eye-centers",
                        "--no-add-border-points",
                        "--no-automatic-retouching",
                        "--no-color-equalization",
                        "--equalization-method",
                        "lightness",
                        "--no-blend-background",
                        "--intermediate-results",
                    ]
                )

            expected_directory = output_directory / "M_first_second"

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            set(saved_paths),
            {
                expected_directory / "morphed.png",
                expected_directory / "warped_image1.png",
                expected_directory / "warped_image2.png",
                expected_directory / "aligned_image1.png",
                expected_directory / "aligned_image2.png",
            },
        )
        kwargs = morph.call_args.kwargs
        self.assertEqual(kwargs["warping_factor"], 0.2)
        self.assertEqual(kwargs["blending_factor"], 0.8)
        self.assertFalse(kwargs["align_eye_centers"])
        self.assertFalse(kwargs["add_border_points"])
        self.assertFalse(kwargs["automatic_retouching"])
        self.assertFalse(kwargs["color_equalization"])
        self.assertEqual(kwargs["equalization_method"], "lightness")
        self.assertFalse(kwargs["blend_background"])
        self.assertTrue(kwargs["return_details"])


if __name__ == "__main__":
    unittest.main()
