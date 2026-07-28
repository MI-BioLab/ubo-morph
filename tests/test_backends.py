from __future__ import annotations

import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import get_args

import pytest


class TestBackendContract:
    def test_public_contract_has_exact_names_and_abstract_hooks(self) -> None:
        from ubo_morph.morphing import Backend, BackendName

        assert get_args(BackendName) == ("cpu", "cupy")
        assert inspect.isabstract(Backend)
        assert Backend.__abstractmethods__ == {
            "to_backend",
            "to_numpy",
            "pad_image",
            "warp_affine",
            "resize_image",
            "match_histogram_channel",
            "equalize_lightness",
            "copy_with_mask",
            "feather_fields",
            "blend",
            "warp_triangles",
        }

        for method_name in Backend.__abstractmethods__:
            parameters = inspect.signature(getattr(Backend, method_name)).parameters
            assert inspect.Parameter.POSITIONAL_ONLY not in {
                parameter.kind for parameter in parameters.values()
            }, method_name

        blend_parameters = inspect.signature(Backend.blend).parameters
        assert blend_parameters["blending_factor"].default == 0.5
        assert (
            blend_parameters["foreground_alpha"].kind == inspect.Parameter.KEYWORD_ONLY
        )
        assert blend_parameters["foreground_alpha"].default is None

    def test_concrete_backends_are_exported_only_by_their_subpackages(self) -> None:
        import ubo_morph.morphing as morphing
        from ubo_morph.morphing.cpu import CPUBackend

        from ubo_morph.morphing import Backend

        assert CPUBackend.name == "cpu"
        assert issubclass(CPUBackend, Backend)
        for name in ("CPUBackend", "CuPyBackend"):
            assert name not in morphing.__all__
            assert not hasattr(morphing, name)

    def test_cupy_backend_class_metadata(self) -> None:
        from ubo_morph.morphing import Backend

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                from ubo_morph.morphing.cupy import CuPyBackend
        except ImportError as error:
            pytest.skip(f"CuPy extra is not installed: {error}")

        assert CuPyBackend.name == "cupy"
        assert issubclass(CuPyBackend, Backend)

    def test_legacy_low_level_functions_are_not_public(self) -> None:
        import ubo_morph
        import ubo_morph.morphing as morphing

        legacy_names = {
            "align_face_images",
            "blend_images",
            "delaunay_triangles",
            "equalize_face",
            "substitute_background",
            "warp_image_by_triangles",
        }
        assert legacy_names.isdisjoint(ubo_morph.__all__)
        assert legacy_names.isdisjoint(morphing.__all__)
        for module in (ubo_morph, morphing):
            for name in legacy_names:
                assert not hasattr(module, name), (module.__name__, name)

    def test_get_backend_caches_successful_instances_and_rejects_exact_names(
        self,
    ) -> None:
        from ubo_morph.morphing import get_backend
        from ubo_morph.morphing.cpu import CPUBackend

        get_backend.cache_clear()
        first = get_backend("cpu")
        second = get_backend("cpu")

        assert isinstance(first, CPUBackend)
        assert first is second

    @pytest.mark.parametrize("invalid", ["CPU", "gpu", "", " cpu", "cpu "])
    def test_get_backend_rejects_inexact_names(self, invalid: str) -> None:
        from ubo_morph.morphing import get_backend

        with pytest.raises(
            ValueError,
            match="backend must be one of: cpu, cupy",
        ):
            get_backend(invalid)  # type: ignore[arg-type]

    def test_selector_uses_explicit_branch_local_imports(self) -> None:
        from ubo_morph.morphing import get_backend

        source = inspect.getsource(get_backend.__wrapped__)
        assert "importlib" not in source
        assert "_REGISTRY" not in source
        assert "from ubo_morph.morphing.cpu import CPUBackend" in source
        assert "from ubo_morph.morphing.cupy import CuPyBackend" in source

class TestBackendSelectionProcess:
    @staticmethod
    def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_missing_cupy_has_actionable_guidance(self) -> None:
        result = self._run_fresh_python(
            "import builtins\n"
            "original_import = builtins.__import__\n"
            "def guarded(name, *args, **kwargs):\n"
            "    if name == 'cupy' or name.startswith('cupy.'):\n"
            "        raise ModuleNotFoundError(\"No module named 'cupy'\", name='cupy')\n"
            "    return original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded\n"
            "from ubo_morph.morphing import get_backend\n"
            "try:\n"
            "    get_backend('cupy')\n"
            "except ImportError as error:\n"
            "    message = str(error)\n"
            "    assert 'ubo-morph[cupy]' in message, message\n"
            "else:\n"
            "    raise AssertionError('CuPy backend unexpectedly loaded')\n"
        )

        assert result.returncode == 0, result.stderr or result.stdout

    def test_transitive_backend_import_failure_propagates(self) -> None:
        result = self._run_fresh_python(
            "import builtins\n"
            "original_import = builtins.__import__\n"
            "def guarded(name, *args, **kwargs):\n"
            "    if name == 'cupy':\n"
            "        raise ModuleNotFoundError(\"No module named 'broken_dependency'\", "
            "name='broken_dependency')\n"
            "    return original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded\n"
            "from ubo_morph.morphing import get_backend\n"
            "try:\n"
            "    get_backend('cupy')\n"
            "except ModuleNotFoundError as error:\n"
            "    assert error.name == 'broken_dependency', error\n"
            "else:\n"
            "    raise AssertionError('transitive failure was swallowed')\n"
        )

        assert result.returncode == 0, result.stderr or result.stdout
