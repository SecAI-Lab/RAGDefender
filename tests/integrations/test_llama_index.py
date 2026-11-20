"""Tests for :class:`ragdefender.integrations.llama_index.RAGDefenderPostprocessor`."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode  # noqa: E402

from ragdefender.integrations.llama_index import RAGDefenderPostprocessor  # noqa: E402

from .conftest import assert_removed_ground_truth  # noqa: E402


def _nodes(texts):
    return [
        NodeWithScore(node=TextNode(text=t, metadata={"pos": i}), score=1.0 - i / 100)
        for i, t in enumerate(texts)
    ]


def _positions(nodes):
    return [n.node.metadata["pos"] for n in nodes]


def test_removes_ground_truth_and_keeps_scores(shared_defender, single_hop_R):
    pp = RAGDefenderPostprocessor(defender=shared_defender)
    nodes = _nodes(single_hop_R["R"])
    out = pp.postprocess_nodes(nodes, query_str=single_hop_R["query"])
    assert all(isinstance(n, NodeWithScore) for n in out)
    assert_removed_ground_truth(_positions(out), single_hop_R)
    assert [n.score for n in out] == [nodes[p].score for p in _positions(out)]


def test_accepts_query_bundle(shared_defender, single_hop_R):
    pp = RAGDefenderPostprocessor(defender=shared_defender)
    out = pp.postprocess_nodes(_nodes(single_hop_R["R"]), QueryBundle(single_hop_R["query"]))
    assert_removed_ground_truth(_positions(out), single_hop_R)


def test_task_type_override_per_call(shared_defender, multi_hop_R):
    pp = RAGDefenderPostprocessor(defender=shared_defender, task_type="multi_hop")
    out = pp.postprocess_nodes(_nodes(multi_hop_R["R"]), query_str=multi_hop_R["query"])
    assert_removed_ground_truth(_positions(out), multi_hop_R)


def test_task_type_required_without_defender():
    with pytest.raises(ValueError, match="task_type"):
        RAGDefenderPostprocessor()


def test_builds_defender_from_config_and_roundtrips(single_hop_R):
    pp = RAGDefenderPostprocessor(task_type="single_hop", embedder="minilm-all", device="cpu")
    assert pp.class_name() == "RAGDefenderPostprocessor"
    dumped = pp.to_dict()
    assert dumped["task_type"] == "single_hop" and "_defender" not in dumped
    restored = RAGDefenderPostprocessor.from_dict(dumped)
    out = restored.postprocess_nodes(_nodes(single_hop_R["R"]), query_str=single_hop_R["query"])
    assert_removed_ground_truth(_positions(out), single_hop_R)


def test_metadata_mode_is_configurable(shared_defender, single_hop_R):
    pp = RAGDefenderPostprocessor(defender=shared_defender, metadata_mode=MetadataMode.EMBED)
    out = pp.postprocess_nodes(_nodes(single_hop_R["R"]), query_str=single_hop_R["query"])
    assert 0 < len(out) <= len(single_hop_R["R"])


def test_small_sets_pass_through(shared_defender):
    pp = RAGDefenderPostprocessor(defender=shared_defender)
    nodes = _nodes(["Paris is the capital of France.", "Lyon is the capital of France."])
    assert pp.postprocess_nodes(nodes, query_str="q") == nodes
    assert pp.postprocess_nodes([], query_str="q") == []


def test_async_path(shared_defender, single_hop_R):
    pp = RAGDefenderPostprocessor(defender=shared_defender)
    out = asyncio.run(
        pp.apostprocess_nodes(_nodes(single_hop_R["R"]), query_str=single_hop_R["query"])
    )
    assert_removed_ground_truth(_positions(out), single_hop_R)


def test_plugs_into_query_engine(shared_defender, single_hop_R):
    """The documented hook: index.as_query_engine(node_postprocessors=[...])."""
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM

    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM()
    index = VectorStoreIndex.from_documents(
        [Document(text=t, metadata={"pos": i}) for i, t in enumerate(single_hop_R["R"])]
    )
    engine = index.as_query_engine(
        similarity_top_k=len(single_hop_R["R"]),
        node_postprocessors=[RAGDefenderPostprocessor(defender=shared_defender)],
    )
    response = engine.query(single_hop_R["query"])
    assert_removed_ground_truth(sorted(_positions(response.source_nodes)), single_hop_R)
