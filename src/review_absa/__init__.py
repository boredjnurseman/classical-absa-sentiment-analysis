"""Classical aspect-based sentiment analysis components."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .experiments import RunConfig, RunResult


__version__ = "0.1.0"

__all__ = ["RunConfig", "RunResult", "run_experiment", "__version__"]


def __getattr__(name: str) -> Any:
    """Lazily expose experiment entry points without importing pandas at startup.

    Args:
        name: Public attribute requested by the caller.

    Returns:
        The requested experiment class or function.

    Raises:
        AttributeError: If ``name`` is not a supported public attribute.
    """
    if name in {"RunConfig", "RunResult", "run_experiment"}:
        from . import experiments

        return getattr(experiments, name)
    raise AttributeError(name)
