"""Logging setup for the ragdefender package.

Modules use ``logger = logging.getLogger(__name__)``. By default the package
is silent (no handler) so importing it does not steal stdout from the host
application; ``setup_logging()`` wires up a single StreamHandler when called
explicitly (e.g. from the CLI ``--verbose`` flag).
"""
from __future__ import annotations

import logging
from typing import Union

_DEFAULT_FMT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
_LOGGER_NAME = "ragdefender"


def setup_logging(level: Union[int, str] = logging.INFO, fmt: str = _DEFAULT_FMT) -> logging.Logger:
    """Mount a single StreamHandler on the ``ragdefender`` logger.

    Idempotent: calling twice replaces the handler rather than stacking them.
    Safe to call from library code or the CLI; library code that only wants
    to emit messages should just use ``logging.getLogger(__name__)`` and let
    the application decide when to call ``setup_logging``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger(_LOGGER_NAME)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Convenience wrapper so callers don't need to import ``logging`` themselves."""
    return logging.getLogger(name or _LOGGER_NAME)
