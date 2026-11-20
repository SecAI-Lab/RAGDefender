# llama-index-postprocessor-ragdefender

[RAGDefender](https://github.com/SecAI-Lab/RAGDefender) as a LlamaIndex
`BaseNodePostprocessor`. It sits after retrieval and drops passages that look
like a knowledge-corruption (poisoning) attack before they reach the LLM,
using the method from *Rescuing the Unpoisoned: Efficient Defense against
Knowledge Corruption Attacks on RAG Systems* (ACSAC 2025,
[arXiv:2511.01268](https://arxiv.org/abs/2511.01268)). No LLM calls, no
training; one embedding pass plus clustering.

```bash
pip install llama-index-postprocessor-ragdefender
```

```python
from llama_index.postprocessor.ragdefender import RAGDefenderPostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[RAGDefenderPostprocessor(task_type="single_hop")],
)
```

`task_type` is `"single_hop"` for factoid questions answered by one passage or
`"multi_hop"` for questions that need several. Node metadata, scores and
retrieval order are preserved; retrieval sets smaller than `min_passages`
(default 3) pass through untouched.

This package is a thin namespace wrapper: the class is implemented in
[`ragdefender.integrations.llama_index`](https://github.com/SecAI-Lab/RAGDefender/blob/main/ragdefender/integrations/llama_index.py)
and documented in
[docs/integrations.md](https://github.com/SecAI-Lab/RAGDefender/blob/main/docs/integrations.md#llamaindex).
Issues: <https://github.com/SecAI-Lab/RAGDefender/issues>.
