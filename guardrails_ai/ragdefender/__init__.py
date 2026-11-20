"""Guardrails-style import path for the RAGDefender validator.

Guardrails validators are published under the PEP 420 ``guardrails_ai`` namespace
(``from guardrails_ai.detect_pii import DetectPII``). This module makes RAGDefender
importable the same way once ``ragdefender[guardrails]`` is installed::

    from guardrails_ai.ragdefender import RAGDefenderValidator

The implementation lives in :mod:`ragdefender.integrations.guardrails`; nothing is
duplicated here.
"""

from ragdefender.integrations.guardrails import RAGDefenderValidator

__all__ = ["RAGDefenderValidator"]
