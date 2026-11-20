"""Drop-in adapters that plug RAGDefender into mainstream RAG frameworks.

RAGDefender is a post-retrieval filter, so it maps onto the post-processing
hook each framework already exposes. Every adapter is one class, keeps the
framework's own document objects (metadata, scores, ids survive), and needs
only the matching optional extra:

=============  ===============================  ======================================
Framework      Adapter                          Install
=============  ===============================  ======================================
LangChain      :class:`RAGDefenderCompressor`    ``pip install ragdefender[langchain]``
LlamaIndex     :class:`RAGDefenderPostprocessor` ``pip install ragdefender[llama-index]``
Haystack 2.x   :class:`RAGDefenderFilter`        ``pip install ragdefender[haystack]``
Guardrails AI  :class:`RAGDefenderValidator`     ``pip install ragdefender[guardrails]``
=============  ===============================  ======================================

The core ``ragdefender`` package does not depend on any of these frameworks.
Importing this module is always safe; the adapter classes are resolved lazily
on first attribute access and raise an :class:`ImportError` naming the extra
to install when their framework is missing::

    from ragdefender.integrations import RAGDefenderCompressor      # LangChain
    from ragdefender.integrations import RAGDefenderPostprocessor   # LlamaIndex
    from ragdefender.integrations import RAGDefenderFilter          # Haystack
    from ragdefender.integrations import RAGDefenderValidator       # Guardrails AI

Each adapter takes the same configuration as :class:`ragdefender.RAGDefender`
(``task_type``, ``embedder``, ``device``, ``p``, ``m``) plus ``min_passages``
(retrieved sets smaller than this pass through unfiltered; default 3), or a
pre-built ``defender=RAGDefender(...)`` to share one embedder across pipelines.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:  # pragma: no cover — IDE / type-checker support only
    from ragdefender.integrations.guardrails import RAGDefenderValidator  # noqa: F401
    from ragdefender.integrations.haystack import RAGDefenderFilter  # noqa: F401
    from ragdefender.integrations.langchain import RAGDefenderCompressor  # noqa: F401
    from ragdefender.integrations.llama_index import RAGDefenderPostprocessor  # noqa: F401

# public name -> (submodule, pip extra)
_LAZY: Dict[str, Tuple[str, str]] = {
    "RAGDefenderCompressor": ("ragdefender.integrations.langchain", "langchain"),
    "RAGDefenderPostprocessor": ("ragdefender.integrations.llama_index", "llama-index"),
    "RAGDefenderFilter": ("ragdefender.integrations.haystack", "haystack"),
    "RAGDefenderValidator": ("ragdefender.integrations.guardrails", "guardrails"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy loader so the core package never imports a framework."""
    try:
        module_name, _extra = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
