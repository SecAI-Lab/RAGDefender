"""Back-compat shim — re-exports :class:`Evaluator` from its v0.2.0 location."""
import warnings

from ragdefender.evaluator import Evaluator  # noqa: F401

warnings.warn(
    "ragdefender.core.evaluator is deprecated; "
    "use `from ragdefender import Evaluator` (or `from ragdefender.evaluator import Evaluator`).",
    DeprecationWarning,
    stacklevel=2,
)
