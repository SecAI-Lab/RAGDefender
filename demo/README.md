---
title: RAGDefender
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Drop poisoned passages from RAG retrieval results
---

# RAGDefender demo

Paste a query and the passages your retriever returned, and see which ones RAGDefender removes.

RAGDefender is a post-retrieval defense against knowledge-corruption attacks on RAG systems
(PoisonedRAG, GARAG and similar). It embeds the retrieved passages, clusters them and drops the
group that is abnormally concentrated around the query, the footprint of injected passages.
No LLM calls, no training.

- Paper: *Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG
  Systems*, ACSAC 2025 ([DOI](https://doi.org/10.1109/ACSAC67867.2025.00093),
  [arXiv](https://arxiv.org/abs/2511.01268))
- Code and framework adapters: <https://github.com/SecAI-Lab/RAGDefender>
- Install: `pip install ragdefender`

This Space runs the default MiniLM embedder on CPU. The source of the app is
[`demo/app.py`](https://github.com/SecAI-Lab/RAGDefender/blob/main/demo/app.py) in the repo.
