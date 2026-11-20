"""Tests for :class:`ragdefender.integrations.haystack.RAGDefenderFilter`."""

from __future__ import annotations

from typing import List

import pytest

pytest.importorskip("haystack")

from haystack import Document, Pipeline, component  # noqa: E402
from haystack.components.builders import PromptBuilder  # noqa: E402

from ragdefender.integrations.haystack import RAGDefenderFilter  # noqa: E402

from .conftest import assert_removed_ground_truth  # noqa: E402


def _docs(texts):
    return [Document(content=t, meta={"pos": i}) for i, t in enumerate(texts)]


def test_run_splits_kept_and_removed(shared_defender, single_hop_R):
    flt = RAGDefenderFilter(defender=shared_defender)
    out = flt.run(documents=_docs(single_hop_R["R"]), query=single_hop_R["query"])
    assert set(out) == {"documents", "removed_documents"}
    assert_removed_ground_truth((d.meta["pos"] for d in out["documents"]), single_hop_R)
    assert sorted(d.meta["pos"] for d in out["removed_documents"]) == sorted(
        single_hop_R["adversarial_indices"]
    )


def test_task_type_override_per_call(shared_defender, multi_hop_R):
    flt = RAGDefenderFilter(defender=shared_defender, task_type="multi_hop")
    out = flt.run(documents=_docs(multi_hop_R["R"]), query=multi_hop_R["query"])
    assert_removed_ground_truth((d.meta["pos"] for d in out["documents"]), multi_hop_R)


def test_task_type_required_without_defender():
    with pytest.raises(ValueError, match="task_type"):
        RAGDefenderFilter()


def test_warm_up_is_lazy_and_idempotent(single_hop_R):
    flt = RAGDefenderFilter(task_type="single_hop", embedder="minilm-all", device="cpu")
    assert flt.defender is None  # nothing loaded at construction
    flt.warm_up()
    loaded = flt.defender
    assert loaded is not None
    flt.warm_up()
    assert flt.defender is loaded
    out = flt.run(documents=_docs(single_hop_R["R"]), query=single_hop_R["query"])
    assert_removed_ground_truth((d.meta["pos"] for d in out["documents"]), single_hop_R)


def test_run_without_warm_up_loads_on_demand(shared_defender):
    flt = RAGDefenderFilter(task_type="single_hop", embedder="minilm-all", device="cpu")
    out = flt.run(documents=_docs(["a", "b"]), query="q")  # below min_passages → passthrough
    assert len(out["documents"]) == 2 and out["removed_documents"] == []
    assert flt.defender is not None


def test_serialisation_roundtrip():
    flt = RAGDefenderFilter(task_type="multi_hop", device="cpu", min_passages=4, p=3)
    data = flt.to_dict()
    assert data["init_parameters"]["task_type"] == "multi_hop"
    assert "defender" not in data["init_parameters"]
    restored = RAGDefenderFilter.from_dict(data)
    assert (restored.task_type, restored.min_passages, restored.p) == ("multi_hop", 4, 3)
    assert restored.defender is None


def test_pipeline_retriever_to_prompt(shared_defender, single_hop_R):
    """retriever ──► RAGDefenderFilter ──► PromptBuilder, the documented topology."""

    @component
    class FakeRetriever:
        def __init__(self, docs):
            self.docs = docs

        @component.output_types(documents=List[Document])
        def run(self, query: str):
            return {"documents": self.docs}

    pipe = Pipeline()
    pipe.add_component("retriever", FakeRetriever(_docs(single_hop_R["R"])))
    pipe.add_component("defender", RAGDefenderFilter(defender=shared_defender))
    pipe.add_component(
        "prompt",
        PromptBuilder(
            template="{% for d in documents %}[{{ d.meta.pos }}]{% endfor %}",
            required_variables=["documents"],
        ),
    )
    pipe.connect("retriever.documents", "defender.documents")
    pipe.connect("defender.documents", "prompt.documents")

    q = single_hop_R["query"]
    result = pipe.run({"retriever": {"query": q}, "defender": {"query": q}})
    rendered = result["prompt"]["prompt"]
    for pos in single_hop_R["adversarial_indices"]:
        assert f"[{pos}]" not in rendered
    assert rendered.count("[") == len(single_hop_R["R"]) - len(single_hop_R["adversarial_indices"])
