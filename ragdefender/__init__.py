"""RAGDefender — Efficient defense against knowledge corruption attacks on RAG systems.

Paper: "Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption
Attacks on RAG Systems" — Kim, Lee, Koo (Sungkyunkwan University), ACSAC 2025.
DOI: https://doi.org/10.1109/ACSAC67867.2025.00093

Quick start::

    from ragdefender import RAGDefender
    defender = RAGDefender(task_type="single_hop")          # NQ / MS MARCO
    safe_passages = defender.defend(query, retrieved_passages)

The full Stage-1 / Stage-2 building blocks are also exposed for callers who
want to recompose them or run ablations:

    from ragdefender import (
        ClusteringBasedGrouping, ConcentrationBasedGrouping,  # Stage 1
        IdentifyAdversarial,                                   # Stage 2
        load_embedder,
    )
"""
__version__ = "0.2.0"
__author__ = "SecAI Lab — Sungkyunkwan University"
__license__ = "MIT"

from ragdefender._logging import setup_logging
from ragdefender.defender import RAGDefender
from ragdefender.embedders import load_embedder
from ragdefender.evaluator import Evaluator
from ragdefender.grouping import (
    ClusteringBasedGrouping,
    ConcentrationBasedGrouping,
)
from ragdefender.identification import IdentifyAdversarial

__all__ = [
    "RAGDefender",
    "Evaluator",
    "ClusteringBasedGrouping",
    "ConcentrationBasedGrouping",
    "IdentifyAdversarial",
    "load_embedder",
    "setup_logging",
    "__version__",
]
