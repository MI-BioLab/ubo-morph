from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np
import pytest

import profile_pipeline
from profile_pipeline import _print_table, _stats
from ubo_morph import Landmarks


def test_stats_reports_distribution_summary_in_milliseconds() -> None:
    stats = _stats([1_000_000, 2_000_000, 3_000_000, 4_000_000])

    assert stats == pytest.approx(
        {
            "mean": 2.5,
            "std": 1.118033988749895,
            "median": 2.5,
            "p05": 1.15,
            "p25": 1.75,
            "p75": 3.25,
            "p95": 3.85,
        }
    )


def test_print_table_reports_metrics_for_every_backend() -> None:
    timings = {
        "cpu": {
            "stage one": {
                "mean": 4.0,
                "std": 0.4,
                "median": 4.0,
                "p05": 3.4,
                "p25": 3.8,
                "p75": 4.2,
                "p95": 4.6,
            }
        },
        "pure-cupy": {
            "stage one": {
                "mean": 3.0,
                "std": 0.3,
                "median": 2.0,
                "p05": 1.4,
                "p25": 1.8,
                "p75": 2.2,
                "p95": 2.6,
            }
        },
        "cupy": {
            "stage one": {
                "mean": 2.0,
                "std": 0.2,
                "median": 1.0,
                "p05": 0.4,
                "p25": 0.8,
                "p75": 1.2,
                "p95": 1.6,
            }
        },
    }
    stdout = StringIO()

    with redirect_stdout(stdout):
        _print_table(timings)

    output = stdout.getvalue()
    for header in (
        "backend",
        "stage",
        "mean",
        "std",
        "median",
        "p05",
        "p25",
        "p75",
        "p95",
        "speedup",
    ):
        assert header in output
    assert "cpu" in output
    assert "pure-cupy" in output
    assert "cupy" in output
    assert "4.00" in output
    assert "2.00" in output
    assert "1.00" in output


def test_configured_backend_profiling_skips_unavailable_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    landmarks = Landmarks(
        left_eye=np.array((1.0, 1.0), dtype=np.float32),
        right_eye=np.array((0.0, 1.0), dtype=np.float32),
        points=np.array(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            dtype=np.float32,
        ),
    )
    expected = {
        "stage one": {
            "mean": 1.0,
            "std": 0.0,
            "median": 1.0,
            "p05": 1.0,
            "p25": 1.0,
            "p75": 1.0,
            "p95": 1.0,
        }
    }
    monkeypatch.setattr(profile_pipeline, "BACKENDS", ("cpu", "unavailable"))
    monkeypatch.setattr(
        profile_pipeline,
        "get_backend",
        lambda name: object()
        if name == "cpu"
        else (_ for _ in ()).throw(ImportError("not installed")),
    )
    monkeypatch.setattr(
        profile_pipeline,
        "_profile_backend",
        lambda *args, **kwargs: expected,
    )
    stderr = StringIO()

    with redirect_stderr(stderr):
        timings = profile_pipeline._profile_configured_backends(
            image,
            image,
            landmarks,
            landmarks,
            3,
        )

    assert timings == {"cpu": expected}
    assert stderr.getvalue() == "unavailable timings skipped: not installed\n"
