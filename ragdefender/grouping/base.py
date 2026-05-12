"""Abstract base for Stage-1 grouping strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class Grouping(ABC):
    """Estimate the number of adversarial passages in a retrieved set.

    Subclasses implement either the clustering-based (single-hop) or the
    concentration-based (multi-hop) strategy from paper §4.1.
    """

    @abstractmethod
    def estimate_n_adv(self, R: List[str], embeddings: Optional["object"] = None) -> int:
        """Return ``N_adv`` for the retrieved set ``R``.

        Args:
            R: Retrieved passages (the paper's :math:`\\mathcal{R}`).
            embeddings: Optional pre-computed sentence embeddings for ``R``.
                When ``None`` the implementation calls ``self.embedder.encode(R)``.
                Passing pre-computed embeddings lets the caller share them
                between Stage 1 and Stage 2 without re-encoding.

        Returns:
            Non-negative integer in ``[0, len(R)]``.
        """
