"""Stage 1 of the RAGDefender pipeline (paper §4.1) — Grouping Retrieved Passages.

Two strategies, picked per query type:

* :class:`ClusteringBasedGrouping` — for single-hop QA (NQ, MS MARCO).
  Hierarchical agglomerative clustering on passage embeddings, validated by a
  TF-IDF top-``m`` term-frequency vote (Eq. 1, 2).
* :class:`ConcentrationBasedGrouping` — for multi-hop QA (HotpotQA).
  Counts passages whose mean and median pairwise similarity stand out from the
  global mean/median (Eq. 3).

Both return ``N_adv``, the estimated number of adversarial passages in ``R``.
The actual indices are picked in Stage 2 (:mod:`ragdefender.identification`).
"""
from ragdefender.grouping.base import Grouping
from ragdefender.grouping.clustering import ClusteringBasedGrouping
from ragdefender.grouping.concentration import ConcentrationBasedGrouping

__all__ = ["Grouping", "ClusteringBasedGrouping", "ConcentrationBasedGrouping"]
