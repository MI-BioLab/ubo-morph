from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ubo_morph.morphing.cupy.backend import CuPyBackend

__all__ = [
    "CuPyBackend",
]


def __getattr__(name: str) -> Any:
    """Load the optional backend only when its public class is requested."""
    if name == "CuPyBackend":
        from ubo_morph.morphing.cupy.backend import CuPyBackend

        return CuPyBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
