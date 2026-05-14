"""Concentration-based Grouping for multi-hop QA — paper §4.1.

Adversarial passages tend to cluster tightly in embedding space, while
multi-hop golden passages are dispersed. We compute, for each passage, its
mean and median pairwise cosine similarity to the rest of ``R``, then count
passages whose concentration factors stand out from the global mean/median.

Implementation note (read ``docs/algorithm.md`` first)
======================================================
The paper text states (paraphrased):

    N_adv = count{ i : (s^mean_i > mean(s^mean)) AND (s^median_i > median(s^median)) }

The implementation preserved here from ``ragdefender.core.defender._find_num_adversarial``
(v0.1.1) differs in three ways:
1. It uses ``OR`` rather than ``AND``.
2. The median threshold is ``(avg_median + avg_avg) / 2`` rather than ``median(s^median)``.
3. The result is *flipped* to ``|R| - sum(final)`` whenever ``avg_avg >= avg_median``.

We deliberately do NOT silently rewrite this to match the paper text in
v0.2.0 because the published ``claims/*/expected/result.txt`` files were
produced by the existing computation. A paper-faithful implementation lands
as part of Phase 6 (a separate behaviour-change release) along with
regenerated expected outputs.
"""
from __future__ import annotations

from typing import List, Optional

from ragdefender.grouping.base import Grouping


class ConcentrationBasedGrouping(Grouping):
    """Multi-hop QA grouping (paper §4.1). Datasets: HotpotQA."""

    def __init__(self, embedder):
        """
        Args:
            embedder: A SentenceTransformer-like object exposing ``.encode(texts, convert_to_tensor=True)``.
        """
        self.embedder = embedder

    def estimate_n_adv(self, R: List[str], embeddings=None) -> int:
        if not R:
            return 0
        if embeddings is None:
            embeddings = self.embedder.encode(R, convert_to_tensor=True)

        # Lazy imports — keep top-level package import light
        import torch
        from sentence_transformers import util

        cos_sim = util.cos_sim(embeddings, embeddings)
        avg = torch.mean(cos_sim, dim=0)
        median = torch.median(cos_sim, dim=0)
        avg_avg = avg.mean()
        avg_median = median.values.median()

        # Concentration factors per passage (paper §4.1: s^mean_i, s^median_i)
        above_avg = [1 if score > avg_avg else 0 for score in avg]
        above_median = [
            1 if score > (avg_median + avg_avg) / 2 else 0 for score in median.values
        ]
        # OR rather than AND — see "Implementation note" in this module's docstring.
        final = [
            1 if above_avg[i] == 1 or above_median[i] == 1 else 0
            for i in range(len(above_avg))
        ]

        # Result-flipping branch: see "Implementation note".
        result = (
            sum(final)
            if sum(final) > 0 and avg_avg < avg_median
            else len(R) - sum(final)
        )
        return int(result)
