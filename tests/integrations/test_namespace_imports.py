"""Framework-native import paths resolve to the same adapter classes.

``guardrails_ai.ragdefender`` ships inside the ragdefender wheel; the
``llama_index.postprocessor.ragdefender`` package is a separate distribution under
``packages/`` and is only exercised when it is installed.
"""

import warnings

import pytest


def test_guardrails_namespace_module():
    pytest.importorskip("guardrails")
    from guardrails_ai.ragdefender import RAGDefenderValidator as namespaced
    from ragdefender.integrations.guardrails import RAGDefenderValidator

    assert namespaced is RAGDefenderValidator


def test_guardrails_hub_shim_resolves_validator():
    pytest.importorskip("guardrails")
    hub = pytest.importorskip("guardrails.hub")
    if not hasattr(hub, "_resolve_from_namespace"):
        pytest.skip("guardrails < 0.11 has no guardrails_ai namespace shim")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolved = hub.RAGDefenderValidator

    from ragdefender.integrations.guardrails import RAGDefenderValidator

    assert resolved is RAGDefenderValidator


def test_llama_index_namespace_package():
    pytest.importorskip("llama_index.core")
    namespaced = pytest.importorskip("llama_index.postprocessor.ragdefender")

    from ragdefender.integrations.llama_index import RAGDefenderPostprocessor

    assert namespaced.RAGDefenderPostprocessor is RAGDefenderPostprocessor
