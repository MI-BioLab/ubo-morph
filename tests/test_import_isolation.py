from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _assert_import_isolated(statement: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    source = (
        f"{statement}\n"
        "import sys\n"
        "loaded = sorted(name for name in sys.modules "
        "if name.startswith(('cupy', 'cupyx')))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_root_package_import_is_accelerator_isolated() -> None:
    _assert_import_isolated("import ubo_morph")


def test_backend_contract_import_is_accelerator_isolated() -> None:
    _assert_import_isolated("import ubo_morph.morphing.backend")


def test_cpu_backend_import_is_accelerator_isolated() -> None:
    _assert_import_isolated("import ubo_morph.morphing.cpu")

