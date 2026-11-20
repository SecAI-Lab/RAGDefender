"""Tests for :class:`ragdefender.integrations.langchain.RAGDefenderCompressor`."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402

from ragdefender.integrations.langchain import RAGDefenderCompressor  # noqa: E402

from .conftest import assert_removed_ground_truth  # noqa: E402


def _docs(texts):
    return [Document(page_content=t, metadata={"pos": i}) for i, t in enumerate(texts)]


def test_removes_ground_truth_and_keeps_documents(shared_defender, single_hop_R):
    comp = RAGDefenderCompressor(defender=shared_defender)
    out = comp.compress_documents(_docs(single_hop_R["R"]), single_hop_R["query"])
    assert all(isinstance(d, Document) for d in out)
    assert_removed_ground_truth((d.metadata["pos"] for d in out), single_hop_R)


def test_task_type_override_per_call(shared_defender, multi_hop_R):
    """A single_hop defender can be driven in multi_hop mode by the adapter's task_type."""
    comp = RAGDefenderCompressor(defender=shared_defender, task_type="multi_hop")
    out = comp.compress_documents(_docs(multi_hop_R["R"]), multi_hop_R["query"])
    assert_removed_ground_truth((d.metadata["pos"] for d in out), multi_hop_R)


def test_task_type_required_without_defender():
    with pytest.raises(ValueError, match="task_type"):
        RAGDefenderCompressor()


def test_builds_defender_from_config(single_hop_R):
    comp = RAGDefenderCompressor(task_type="single_hop", embedder="minilm-all", device="cpu")
    assert comp.defender.task_type == "single_hop"
    out = comp.compress_documents(_docs(single_hop_R["R"]), single_hop_R["query"])
    assert_removed_ground_truth((d.metadata["pos"] for d in out), single_hop_R)


def test_config_is_serialisable_and_hides_defender(shared_defender):
    comp = RAGDefenderCompressor(defender=shared_defender, task_type="single_hop")
    dumped = comp.model_dump()
    assert dumped["task_type"] == "single_hop"
    assert "_defender" not in dumped and "defender" not in dumped


def test_small_and_empty_sets_pass_through(shared_defender):
    comp = RAGDefenderCompressor(defender=shared_defender)
    docs = _docs(["Paris is the capital of France.", "Lyon is the capital of France."])
    assert comp.compress_documents(docs, "q") == docs
    assert comp.compress_documents([], "q") == []


def test_async_path(shared_defender, single_hop_R):
    comp = RAGDefenderCompressor(defender=shared_defender)
    out = asyncio.run(comp.acompress_documents(_docs(single_hop_R["R"]), single_hop_R["query"]))
    assert_removed_ground_truth((d.metadata["pos"] for d in out), single_hop_R)


def test_wrap_retriever_contextual_compression(shared_defender, single_hop_R):
    pytest.importorskip("langchain_classic")
    from langchain_core.retrievers import BaseRetriever

    class FakeRetriever(BaseRetriever):
        docs: list

        def _get_relevant_documents(self, query, *, run_manager):
            return self.docs

    comp = RAGDefenderCompressor(defender=shared_defender)
    retriever = comp.wrap_retriever(FakeRetriever(docs=_docs(single_hop_R["R"])))
    out = retriever.invoke(single_hop_R["query"])
    assert_removed_ground_truth((d.metadata["pos"] for d in out), single_hop_R)
