"""Cosine-similarity helpers shared by Stage 1 (grouping) and Stage 2 (identification).

These were previously duplicated in ``artifacts/src/utils.py`` and inside the
defender. Hosting them here makes both the package and the artifact code import
from a single source of truth.
"""
from __future__ import annotations

import math
from typing import List, Tuple


def cos_sim_matrix(embeddings):
    """Return the |R|x|R| cosine similarity matrix as a torch tensor.

    Thin wrapper over ``sentence_transformers.util.cos_sim`` so callers don't
    need to import that path themselves.
    """
    from sentence_transformers import util
    return util.cos_sim(embeddings, embeddings)


def top_similar_pairs(embeddings, k: int) -> List[Tuple[Tuple[int, int], float]]:
    """Return the top-``k`` most-similar (i, j) index pairs (i < j) with their cos sim.

    Output is sorted by similarity descending.

    Equivalent to the ``top_similar_pairs`` previously living in
    ``artifacts/src/utils.py:40-55``; the artifact module keeps a backward-shim
    re-export so existing research code keeps working.
    """
    from sentence_transformers import util
    n = len(embeddings)
    if n < 2 or k <= 0:
        return []
    pairs: List[Tuple[Tuple[int, int], float]] = []
    sim_mat = util.cos_sim(embeddings, embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(((i, j), float(sim_mat[i][j].item())))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs[:k]


def n_pairs_for(n_adv: int) -> int:
    """Paper §4.2 Eq. 4: ``N_pairs = max(1, C(N_adv, 2))``."""
    if n_adv <= 0:
        return 0
    return max(1, math.comb(n_adv, 2))
