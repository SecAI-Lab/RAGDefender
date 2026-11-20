"""RAGDefender inside LlamaIndex, as a node postprocessor.

    pip install "ragdefender[llama-index]"

Shows the postprocessor on a raw ``List[NodeWithScore]`` and plugged into a
query engine. The index uses LlamaIndex's built-in mock embedding / mock LLM so
the script needs no API key; swap them for your real models.
"""
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from ragdefender.integrations import RAGDefenderPostprocessor

from _passages import PASSAGES, QUERY, report


def main() -> None:
    postprocessor = RAGDefenderPostprocessor(task_type="single_hop", device="cpu")

    # 1. On retrieved nodes directly
    nodes = [
        NodeWithScore(node=TextNode(text=t, metadata={"pos": i}), score=1.0 - i / 100)
        for i, t in enumerate(PASSAGES)
    ]
    safe = postprocessor.postprocess_nodes(nodes, query_str=QUERY)
    report("postprocess_nodes()", (n.node.metadata["pos"] for n in safe))

    # 2. Inside a query engine (mock embedding + mock LLM; no API key needed)
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM()
    index = VectorStoreIndex.from_documents(
        [Document(text=t, metadata={"pos": i}) for i, t in enumerate(PASSAGES)]
    )
    engine = index.as_query_engine(
        similarity_top_k=len(PASSAGES), node_postprocessors=[postprocessor]
    )
    response = engine.query(QUERY)
    report("as_query_engine(node_postprocessors=[...]).query()",
           (n.node.metadata["pos"] for n in response.source_nodes))


if __name__ == "__main__":
    main()
