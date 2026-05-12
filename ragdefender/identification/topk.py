"""Identifying Adversarial Passages (paper §4.2).

Given ``N_adv`` from Stage 1, this module picks which specific passages are
adversarial. Algorithm (Eq. 4–7):

1. Compute pairwise cosine similarity over ``R``; take the top
   :math:`N_{pairs} = \\max(1, \\binom{N_{adv}}{2})` most-similar (i, j) pairs.
2. For each passage :math:`r_i` compute the frequency score
   :math:`f_i = \\sum_{(r_i, r_j) \\in \\mathcal{P}_{top}} \\mathrm{sgn}(\\mathrm{sim}(r_i, r_j)) \\cdot |\\mathrm{sim}(r_i, r_j)|^p`
   with :math:`p = 2` (paper default).
3. Rank passages by :math:`f_i` descending and take the top ``N_adv``.

The v0.1.1 ``RAGDefender.defend()`` did NOT execute this step — it just
truncated ``R[:|R|-N_adv]`` (i.e. assumed adversarial passages were already
sorted to the end of the list). Phase 2 closes that gap by routing the new
``defender.defend()`` through this class.
"""
from __future__ import annotations

from typing import List, Optional

from ragdefender.similarity import n_pairs_for, top_similar_pairs


class IdentifyAdversarial:
    """Stage 2: pick the indices of ``N_adv`` most-likely-adversarial passages."""

    def __init__(self, embedder, p: int = 2):
        """
        Args:
            embedder: SentenceTransformer-like.
            p: Frequency-score weighting exponent. Paper default: 2 (see §4.2,
                ablation in Table 7).
        """
        self.embedder = embedder
        self.p = p

    def select(
        self,
        R: List[str],
        n_adv: int,
        embeddings=None,
    ) -> List[int]:
        """Return indices into ``R`` of the ``n_adv`` most-likely-adversarial passages.

        Indices are returned in score-descending order.
        """
        if n_adv <= 0 or not R:
            return []
        if n_adv >= len(R):
            return list(range(len(R)))

        if embeddings is None:
            embeddings = self.embedder.encode(R, convert_to_tensor=True)

        n_pairs = n_pairs_for(n_adv)
        top_pairs = top_similar_pairs(embeddings, n_pairs)

        # Frequency score f_i (paper Eq. 6)
        f = [0.0] * len(R)
        for (i, j), sim in top_pairs:
            sign = 1.0 if sim >= 0.0 else -1.0
            weight = sign * (abs(sim) ** self.p)
            f[i] += weight
            f[j] += weight

        # Rank descending; ties broken by lower index for determinism
        ranked = sorted(range(len(R)), key=lambda i: (-f[i], i))
        return ranked[:n_adv]
