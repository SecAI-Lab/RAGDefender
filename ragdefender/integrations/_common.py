"""Shared glue used by every framework adapter in :mod:`ragdefender.integrations`.

Each adapter does the same three things:

1. pull plain-text passages out of the framework's document object,
2. call :meth:`RAGDefender.defend` with ``return_indices=True``,
3. rebuild the framework's list from the surviving indices so metadata, scores
   and ids travel with their passage.

This module holds steps 2–3 so the adapters stay a screen long.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ragdefender._logging import get_logger
from ragdefender.defender import RAGDefender
from ragdefender.embedders import DEFAULT_EMBEDDER

logger = get_logger(__name__)

#: Below this many passages Stage 1 has nothing to group: hierarchical clustering
#: needs at least two samples and concentration-based grouping degenerates to
#: "drop everything". Adapters pass such retrieved sets through untouched.
DEFAULT_MIN_PASSAGES = 3

TASK_TYPE_HELP = (
    "task_type is required: 'single_hop' for fact-lookup questions (NQ / MS MARCO "
    "style) or 'multi_hop' for compositional questions (HotpotQA style). RAGDefender "
    "picks its Stage-1 grouping strategy from it and deliberately has no "
    "auto-detect (paper §4.1). Either pass task_type=... or hand the adapter a "
    "pre-built RAGDefender via defender=..."
)


def missing_dependency(distribution: str, extra: str) -> ImportError:
    """Build the ImportError raised when an adapter's framework is not installed."""
    return ImportError(
        f"{distribution} is not installed. Install this adapter's optional "
        f"dependencies with:\n\n    pip install 'ragdefender[{extra}]'\n"
    )


def ensure_task_type(defender: Optional[RAGDefender], task_type: Optional[str]) -> None:
    """Raise early (at construction) when neither a defender nor a task_type is given."""
    if defender is None and task_type is None:
        raise ValueError(TASK_TYPE_HELP)


def build_defender(
    defender: Optional[RAGDefender] = None,
    *,
    task_type: Optional[str] = None,
    embedder: str = DEFAULT_EMBEDDER,
    device: str = "auto",
    gpu_id: int = 0,
    p: int = 2,
    m: int = 5,
) -> RAGDefender:
    """Return ``defender`` unchanged, or construct one from the keyword arguments.

    Adapters accept both styles: a pre-built :class:`RAGDefender` (share one
    embedder across several pipelines, or pass a custom SentenceTransformer) or
    the constructor knobs directly (serialisable, framework-native config).
    """
    ensure_task_type(defender, task_type)
    if defender is not None:
        return defender
    return RAGDefender(
        embedder=embedder,
        task_type=task_type,  # type: ignore[arg-type]  # ensure_task_type guarantees str
        device=device,
        gpu_id=gpu_id,
        p=p,
        m=m,
    )


def partition(
    defender: RAGDefender,
    query: str,
    texts: Sequence[str],
    *,
    task_type: Optional[str] = None,
    min_passages: int = DEFAULT_MIN_PASSAGES,
) -> Tuple[List[int], List[int]]:
    """Split the indices of ``texts`` into ``(kept, removed)`` using ``defender``.

    Both lists are sorted ascending so callers can rebuild their document list
    in the original retrieval order. When fewer than ``min_passages`` texts are
    given everything is kept (see :data:`DEFAULT_MIN_PASSAGES`).
    """
    n = len(texts)
    if n < min_passages:
        logger.debug(
            "Retrieved %d passage(s) < min_passages=%d; passing through unfiltered.",
            n,
            min_passages,
        )
        return list(range(n)), []

    _, removed = defender.defend(query, list(texts), task_type=task_type, return_indices=True)
    removed_set = set(removed)
    kept = [i for i in range(n) if i not in removed_set]
    if removed_set:
        logger.info(
            "RAGDefender removed %d of %d retrieved passages: %s",
            len(removed_set),
            n,
            sorted(removed_set),
        )
    return kept, sorted(removed_set)
