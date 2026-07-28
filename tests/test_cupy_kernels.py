from __future__ import annotations

from importlib.resources import files
from unittest.mock import MagicMock, patch

from ubo_morph.morphing.cupy.kernels import KERNEL_NAMES, load_kernel_source


def test_cuda_source_is_a_packaged_resource_with_all_entry_points() -> None:
    resource = files("ubo_morph.morphing.cupy").joinpath("kernels.cu")

    assert resource.is_file()
    source = load_kernel_source()
    assert source == resource.read_text(encoding="utf-8")
    assert all(f'void {name}(' in source for name in KERNEL_NAMES)


def test_backend_builds_one_raw_module_and_retrieves_each_kernel() -> None:
    from ubo_morph.morphing.cupy import backend as gpu_backend

    module = MagicMock()
    with (
        patch.object(gpu_backend.cp, "RawModule", return_value=module) as raw_module,
        patch.object(gpu_backend, "load_kernel_source", return_value="source"),
    ):
        gpu_backend.CuPyBackend()

    raw_module.assert_called_once_with(code="source")
    retrieved = [call.args[0] for call in module.get_function.call_args_list]
    assert len(retrieved) == len(KERNEL_NAMES)
    assert set(retrieved) == set(KERNEL_NAMES)
