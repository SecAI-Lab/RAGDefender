"""RAGDefender as a LlamaIndex node postprocessor.

    pip install llama-index-postprocessor-ragdefender

    from llama_index.postprocessor.ragdefender import RAGDefenderPostprocessor

    query_engine = index.as_query_engine(
        node_postprocessors=[RAGDefenderPostprocessor(task_type="single_hop")],
    )

The implementation lives in :mod:`ragdefender.integrations.llama_index`; this package
only exposes it under the ``llama_index.postprocessor`` namespace that LlamaIndex
integrations use.
"""

from ragdefender.integrations.llama_index import RAGDefenderPostprocessor

__all__ = ["RAGDefenderPostprocessor"]
