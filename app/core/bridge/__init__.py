"""Bridge orchestration components."""

from typing import TYPE_CHECKING

from .models import BridgeResult

if TYPE_CHECKING:  # pragma: no cover
    from .manager import BridgeManager
    from .chain_registry import ChainRegistry

__all__ = ["BridgeResult", "BridgeManager", "ChainRegistry", "get_chain_registry"]


def __getattr__(name: str):  # pragma: no cover - simple thunk
    if name == "BridgeManager":
        from .manager import BridgeManager as _BridgeManager
        return _BridgeManager
    if name == "ChainRegistry":
        from .chain_registry import ChainRegistry as _ChainRegistry
        return _ChainRegistry
    if name == "get_chain_registry":
        from .chain_registry import get_chain_registry as _get_chain_registry
        return _get_chain_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
