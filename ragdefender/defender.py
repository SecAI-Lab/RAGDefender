"""RAGDefender — orchestrator wiring Stage 1 (grouping) and Stage 2 (identification).

Public entry point of the package. The actual algorithms live in
:mod:`ragdefender.grouping` (paper §4.1) and :mod:`ragdefender.identification`
(paper §4.2); this class just glues them together with an embedder.

Migration from v0.1.1
=====================
``similarity_model=`` is now ``embedder=`` (alias kept with DeprecationWarning).
``defend(query, retrieved_docs, mode='multihop')`` is now
``defend(query, R, task_type='multi_hop')`` (aliases kept).

A material behaviour change in v0.2.0: ``defend()`` now actually runs Stage 2
(:class:`IdentifyAdversarial`) to pick the adversarial indices, then drops
those specific passages. v0.1.1 did NOT — it just truncated ``R[:|R|-N_adv]``
on the assumption adversarial passages were already at the end of the list.
See ``CHANGELOG.md`` and ``docs/migration-0.1-to-0.2.md``.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

from ragdefender._logging import get_logger
from ragdefender.embedders import DEFAULT_EMBEDDER, _resolve_device, load_embedder
from ragdefender.grouping import ClusteringBasedGrouping, ConcentrationBasedGrouping
from ragdefender.identification import IdentifyAdversarial

logger = get_logger(__name__)

# Accepted task_type values; old singlehop/multihop strings map here.
_TASK_TYPES = {"single_hop", "multi_hop"}
_DEPRECATED_MODE_MAP = {"singlehop": "single_hop", "multihop": "multi_hop"}


class RAGDefender:
    """Two-stage post-retrieval filter that removes adversarial passages.

    Pipeline:
        1. Stage 1 (grouping) estimates ``N_adv`` — the number of adversarial
           passages in ``R``. Strategy is chosen by ``task_type``:
           ``"single_hop"`` → :class:`ClusteringBasedGrouping`,
           ``"multi_hop"`` → :class:`ConcentrationBasedGrouping`.
        2. Stage 2 (identification) ranks passages by a frequency score over
           the most-similar pairs and returns the indices of the top ``N_adv``.
        3. Those indices are removed from ``R``; the remainder is returned.

    Example::

        from ragdefender import RAGDefender
        defender = RAGDefender(embedder="minilm-paraphrase", task_type="single_hop")
        clean_passages = defender.defend(query, retrieved_passages)
    """

    def __init__(
        self,
        embedder: Union[str, "object"] = DEFAULT_EMBEDDER,
        task_type: str = "auto",
        device: str = "auto",
        gpu_id: int = 0,
        p: int = 2,
        m: int = 5,
        # --- deprecated v0.1.1 kwargs ---
        similarity_model: Optional[Union[str, "object"]] = None,
    ):
        """
        Args:
            embedder: Embedding model. Accepts a preset name
                (``"minilm-all"`` (default, matches v0.1.1), ``"minilm-paraphrase"``,
                ``"stella"``), a Hugging Face id, or an already-instantiated
                SentenceTransformer.
            task_type: ``"single_hop"`` (NQ, MS MARCO), ``"multi_hop"`` (HotpotQA),
                or ``"auto"`` (caller MUST override per ``defend()`` call —
                no heuristic detection; the paper picks the strategy per dataset).
            device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (chooses cuda if available).
            gpu_id: GPU index when ``device == "cuda"``.
            p: Stage-2 frequency-score weighting exponent (paper Eq. 6, default 2).
            m: Stage-1 TF-IDF top-``m`` term count (paper Eq. 1, default 5).
            similarity_model: **Deprecated.** v0.1.1 alias for ``embedder``.
        """
        if similarity_model is not None:
            warnings.warn(
                "similarity_model= is deprecated; use embedder= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            embedder = similarity_model

        self.device = _resolve_device(device)
        self.gpu_id = gpu_id
        self.task_type = self._normalize_task_type(task_type, allow_auto=True)

        self.embedder = load_embedder(embedder, device=self.device, gpu_id=gpu_id)

        # Stage 1 strategies — both held; defend() picks per-call based on task_type.
        self._clustering = ClusteringBasedGrouping(self.embedder, m=m)
        self._concentration = ConcentrationBasedGrouping(self.embedder)
        # Stage 2
        self._identify = IdentifyAdversarial(self.embedder, p=p)

    # ------------------------------------------------------------------ defend
    def defend(
        self,
        query: str,
        R: Optional[List[str]] = None,
        *,
        task_type: Optional[str] = None,
        return_indices: bool = False,
        # --- deprecated v0.1.1 kwargs ---
        retrieved_docs: Optional[List[str]] = None,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Union[List[str], Tuple[List[str], List[int]]]:
        """Filter ``R`` to remove the passages most likely to be adversarial.

        Args:
            query: User query (unused by the current detectors but kept for
                future query-conditioned strategies and API stability).
            R: Retrieved passages — the paper's :math:`\\mathcal{R}`.
            task_type: ``"single_hop"`` or ``"multi_hop"``. Falls back to the
                value passed at construction time. ``"auto"`` raises.
            return_indices: When True, also return the indices removed from ``R``.
            retrieved_docs: **Deprecated** v0.1.1 alias for ``R``.
            mode: **Deprecated** v0.1.1 alias for ``task_type``
                (``"singlehop"`` / ``"multihop"``).
            top_k: **Deprecated** v0.1.1 trim — explicit slicing is clearer.

        Returns:
            ``List[str]`` of the surviving passages (the paper's
            :math:`\\mathcal{R}_{safe}`), or ``(safe, removed_indices)`` when
            ``return_indices=True``.
        """
        # ------ resolve deprecated kwargs ------
        if retrieved_docs is not None:
            warnings.warn(
                "retrieved_docs= is deprecated; pass R= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            R = retrieved_docs
        if mode is not None:
            warnings.warn(
                "mode= is deprecated; pass task_type='single_hop' or 'multi_hop'.",
                DeprecationWarning,
                stacklevel=2,
            )
            task_type = _DEPRECATED_MODE_MAP.get(mode, mode)

        if R is None:
            raise TypeError("defend() missing required argument: R (retrieved passages)")
        if not R:
            return ([], []) if return_indices else []

        effective = self._normalize_task_type(task_type or self.task_type, allow_auto=False)

        # ------ Stage 1: estimate N_adv ------
        # Encode once and share embeddings with Stage 2 to avoid re-encoding.
        embeddings = self.embedder.encode(R, convert_to_tensor=True)
        if effective == "single_hop":
            n_adv = self._clustering.estimate_n_adv(R, embeddings=embeddings)
        else:  # multi_hop
            n_adv = self._concentration.estimate_n_adv(R, embeddings=embeddings)

        logger.debug("task_type=%s |R|=%d N_adv=%d", effective, len(R), n_adv)

        if n_adv <= 0:
            safe = list(R)
            removed: List[int] = []
        else:
            # ------ Stage 2: pick which indices to drop ------
            adv_indices = self._identify.select(R, n_adv, embeddings=embeddings)
            removed = sorted(adv_indices)
            safe = [doc for i, doc in enumerate(R) if i not in set(removed)]

        # legacy top_k trim (deprecated)
        if top_k is not None:
            warnings.warn(
                "top_k= is deprecated; slice the returned list yourself.",
                DeprecationWarning,
                stacklevel=2,
            )
            safe = safe[:top_k]

        # release CUDA memory if applicable
        del embeddings
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        return (safe, removed) if return_indices else safe

    # ------------------------------------------------------------------ metrics
    def get_metrics(
        self,
        original_docs: List[str],
        defended_docs: List[str],
        poisoned_indices: List[int],
    ) -> Dict[str, float]:
        """Precision / Recall / F1 of the defense given ground-truth poisoned indices.

        Note: this matches docs by string equality, which is fine for evaluation
        but breaks if duplicate passages exist in ``original_docs``; pass
        ``return_indices=True`` to ``defend`` for an exact-index alternative.
        """
        kept_indices = [i for i, doc in enumerate(original_docs) if doc in defended_docs]
        removed_indices = [i for i in range(len(original_docs)) if i not in kept_indices]

        poisoned = set(poisoned_indices)
        tp = len(set(removed_indices) & poisoned)
        fp = len(set(removed_indices) - poisoned)
        fn = len(set(kept_indices) & poisoned)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_task_type(task_type: str, *, allow_auto: bool) -> str:
        """Validate task_type, mapping deprecated aliases."""
        if task_type in _DEPRECATED_MODE_MAP:
            warnings.warn(
                f"task_type='{task_type}' is the v0.1.1 spelling; "
                f"use '{_DEPRECATED_MODE_MAP[task_type]}'.",
                DeprecationWarning,
                stacklevel=3,
            )
            task_type = _DEPRECATED_MODE_MAP[task_type]
        if task_type == "auto":
            if allow_auto:
                return task_type
            raise ValueError(
                "task_type='auto' requires an explicit override "
                "(set it in the RAGDefender constructor or pass it to defend()). "
                "There is no heuristic auto-detect: the paper picks "
                "single_hop for NQ/MS MARCO and multi_hop for HotpotQA."
            )
        if task_type not in _TASK_TYPES:
            raise ValueError(
                f"task_type must be 'single_hop' or 'multi_hop', got {task_type!r}"
            )
        return task_type
