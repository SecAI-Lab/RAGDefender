"""Back-compat shim — preserves ``from ragdefender.core import RAGDefender`` for v0.1.1 callers.

The real implementations live one level up at ``ragdefender.defender`` and
``ragdefender.evaluator``. This package will be removed in a future release;
see ``CHANGELOG.md``.
"""
import warnings

from ragdefender.defender import RAGDefender  # noqa: F401
from ragdefender.evaluator import Evaluator  # noqa: F401

warnings.warn(
    "ragdefender.core is deprecated; import from ragdefender directly "
    "(e.g. `from ragdefender import RAGDefender`).",
    DeprecationWarning,
    stacklevel=2,
)
