"""Back-compat shim — re-exports :class:`RAGDefender` from its v0.2.0 location."""
import warnings

from ragdefender.defender import RAGDefender  # noqa: F401

warnings.warn(
    "ragdefender.core.defender is deprecated; "
    "use `from ragdefender import RAGDefender` (or `from ragdefender.defender import RAGDefender`).",
    DeprecationWarning,
    stacklevel=2,
)
