"""End-to-end tests for the v0.2.0 RAGDefender orchestrator."""
from __future__ import annotations

import warnings

import pytest

from ragdefender import RAGDefender


def _build_defender(task_type: str) -> RAGDefender:
    """Build a CPU defender that uses the v0.1.1-compatible MiniLM checkpoint."""
    return RAGDefender(
        embedder="minilm-all",  # Hugging Face: sentence-transformers/all-MiniLM-L6-v2
        task_type=task_type,
        device="cpu",
    )


def test_defend_single_hop_removes_ground_truth(single_hop_R):
    """v0.2.0 with Stage 2 enabled must remove exactly the ground-truth indices.

    v0.1.1 only achieved F1=0.67 on this fixture because it truncated R[:|R|-N_adv]
    rather than picking specific indices.
    """
    defender = _build_defender("single_hop")
    safe, removed = defender.defend(
        single_hop_R["query"], single_hop_R["R"], return_indices=True
    )
    assert sorted(removed) == sorted(single_hop_R["adversarial_indices"])
    assert len(safe) == len(single_hop_R["R"]) - len(single_hop_R["adversarial_indices"])


def test_defend_multi_hop_removes_ground_truth(multi_hop_R):
    defender = _build_defender("multi_hop")
    safe, removed = defender.defend(
        multi_hop_R["query"], multi_hop_R["R"], return_indices=True
    )
    assert sorted(removed) == sorted(multi_hop_R["adversarial_indices"])
    assert len(safe) == len(multi_hop_R["R"]) - len(multi_hop_R["adversarial_indices"])


def test_defend_returns_strings_only_by_default(single_hop_R):
    """Without return_indices, the return type is a plain list of strings."""
    defender = _build_defender("single_hop")
    safe = defender.defend(single_hop_R["query"], single_hop_R["R"])
    assert isinstance(safe, list)
    assert all(isinstance(d, str) for d in safe)


def test_task_type_auto_in_constructor_and_call_raises(single_hop_R):
    """No heuristic auto-detect — both layers default to auto → must error."""
    defender = RAGDefender(embedder="minilm-all", device="cpu", task_type="auto")
    with pytest.raises(ValueError, match="task_type='auto'"):
        defender.defend(single_hop_R["query"], single_hop_R["R"])


def test_task_type_auto_overridable_by_call(single_hop_R):
    """Constructor task_type='auto' is fine if the call provides one."""
    defender = RAGDefender(embedder="minilm-all", device="cpu", task_type="auto")
    safe = defender.defend(
        single_hop_R["query"], single_hop_R["R"], task_type="single_hop"
    )
    assert isinstance(safe, list)


def test_empty_R_returns_empty():
    defender = _build_defender("single_hop")
    safe, removed = defender.defend("any q", [], return_indices=True)
    assert safe == []
    assert removed == []


# ----------------------------------------------------------- deprecation paths
def test_legacy_kwarg_aliases_emit_deprecation(single_hop_R):
    """retrieved_docs= and mode= still work but warn."""
    defender = _build_defender("single_hop")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        defender.defend(
            single_hop_R["query"],
            retrieved_docs=single_hop_R["R"],
            mode="singlehop",
        )
    msgs = [str(x.message) for x in caught if issubclass(x.category, DeprecationWarning)]
    assert any("retrieved_docs=" in m for m in msgs)
    assert any("mode=" in m for m in msgs)


def test_legacy_similarity_model_kwarg_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RAGDefender(
            similarity_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            task_type="single_hop",
        )
    msgs = [str(x.message) for x in caught if issubclass(x.category, DeprecationWarning)]
    assert any("similarity_model=" in m for m in msgs)


def test_legacy_core_import_warns():
    """from ragdefender.core.defender import RAGDefender — still works, deprecated."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # importlib so the warning fires every test run regardless of cache
        import importlib
        importlib.reload(importlib.import_module("ragdefender.core.defender"))
    msgs = [str(x.message) for x in caught if issubclass(x.category, DeprecationWarning)]
    assert any("ragdefender.core.defender" in m for m in msgs)
