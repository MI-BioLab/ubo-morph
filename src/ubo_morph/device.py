from __future__ import annotations

from typing import Literal, TypeAlias


Device: TypeAlias = Literal["cpu", "gpu"]
SUPPORTED_DEVICES: tuple[Device, ...] = ("cpu", "gpu")


def validate_device(device: str) -> Device:
    """Validate and narrow a public device argument."""
    if device not in SUPPORTED_DEVICES:
        supported = ", ".join(repr(value) for value in SUPPORTED_DEVICES)
        raise ValueError(f"Unsupported device {device!r}; expected one of: {supported}")
    return device
