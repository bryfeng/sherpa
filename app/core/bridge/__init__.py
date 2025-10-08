"""Bridge orchestration components."""

from typing import TYPE_CHECKING

from .models import BridgeResult

if TYPE_CHECKING:  # pragma: no cover
    from .manager import BridgeManager

__all__ = ["BridgeResult", "BridgeManager"]


def __getattr__(name: str):  # pragma: no cover - simple thunk
    if name == "BridgeManager":
        from .manager import BridgeManager as _BridgeManager

        return _BridgeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
