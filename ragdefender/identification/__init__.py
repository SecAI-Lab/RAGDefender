"""Stage 2 of the RAGDefender pipeline (paper §4.2) — Identifying Adversarial Passages.

Given ``N_adv`` from Stage 1, pick the actual indices of the ``N_adv`` most
likely-adversarial passages by ranking each passage on a frequency score
:math:`f_i` computed over the top-:math:`N_{pairs}` most-similar passage pairs.
"""
from ragdefender.identification.topk import IdentifyAdversarial

__all__ = ["IdentifyAdversarial"]
