# Framework integrations

RAGDefender is a post-retrieval filter: it takes the passages a retriever
returned for a query and drops the ones that look adversarial before they reach
the generator. Every major RAG framework exposes exactly that hook, and
`ragdefender.integrations` ships one thin adapter per framework.

| Framework | Hook | Adapter | Install |
|---|---|---|---|
| LangChain | `BaseDocumentCompressor` → `ContextualCompressionRetriever` | `RAGDefenderCompressor` | `pip install "ragdefender[langchain]"` |
| LlamaIndex | `BaseNodePostprocessor` → `node_postprocessors=[...]` | `RAGDefenderPostprocessor` | `pip install "ragdefender[llama-index]"` |
| Haystack 2.x | `@component` between retriever and `PromptBuilder` | `RAGDefenderFilter` | `pip install "ragdefender[haystack]"` |
| Guardrails AI | `Validator` on the retrieved context | `RAGDefenderValidator` | `pip install "ragdefender[guardrails]"` |

`pip install "ragdefender[integrations]"` installs all four. The core package
never imports a framework: `from ragdefender.integrations import <Adapter>`
resolves lazily and raises an `ImportError` naming the extra to install when the
framework is missing. All four frameworks require Python ≥ 3.10.

Runnable versions of every snippet below live in
[`examples/integrations/`](../examples/integrations/).

## Shared behaviour

Every adapter

- takes the same knobs as [`RAGDefender`](algorithm.md) — `task_type`,
  `embedder`, `device`, `gpu_id`, `p`, `m` — **or** a pre-built
  `defender=RAGDefender(...)` so several pipelines can share one embedder
  (or use a custom `SentenceTransformer` instance);
- requires `task_type` unless `defender=` is given: `"single_hop"` for
  fact-lookup questions (NQ / MS MARCO style) or `"multi_hop"` for compositional
  questions (HotpotQA style). There is deliberately no auto-detect (paper §4.1).
  An adapter-level `task_type` overrides the defender's own per call, so one
  shared defender can serve both kinds of pipeline;
- keeps the framework's own document objects: metadata, ids and scores survive,
  and retrieval order is preserved;
- passes retrieved sets with fewer than `min_passages` (default 3) through
  unfiltered. Stage 1 needs several passages to find a suspicious group; with
  one or two passages the method degenerates (clustering needs ≥ 2 samples,
  concentration-based grouping would drop everything);
- embeds the passage text only: `Document.page_content` (LangChain),
  `node.get_content(metadata_mode=MetadataMode.NONE)` (LlamaIndex, configurable),
  `Document.content` (Haystack), the string or list value (Guardrails);
- logs removals at `INFO` on the `ragdefender` logger. The package is silent by
  default; call `ragdefender.setup_logging()` to see them.

Feed the adapters the *retrieved set* (top-k passages, typically 5–20), not the
whole corpus. That is what the paper evaluates and what the detectors expect.

## LangChain

```bash
pip install "ragdefender[langchain]" langchain-classic   # langchain-classic ⇒ ContextualCompressionRetriever
```

```python
from ragdefender.integrations import RAGDefenderCompressor

compressor = RAGDefenderCompressor(task_type="single_hop")

# 1. Wrap any retriever — the filter runs on every retrieval
safe_retriever = compressor.wrap_retriever(vectorstore.as_retriever(search_kwargs={"k": 10}))
docs = safe_retriever.invoke("What is the capital of France?")

# 2. Or apply it to a list of Documents yourself
safe_docs = compressor.compress_documents(docs, query)
```

`RAGDefenderCompressor` subclasses `langchain_core.documents.compressor.BaseDocumentCompressor`,
so it also works anywhere LangChain accepts a compressor (e.g. inside a
`DocumentCompressorPipeline` after a reranker). `acompress_documents` is
inherited and runs the filter in a worker thread. `wrap_retriever()` imports
`ContextualCompressionRetriever` from `langchain_classic` (LangChain ≥ 1.0) or
`langchain` (0.x). The compressor is a pydantic model: `compressor.model_dump()`
gives you its configuration (the embedder instance itself is private and not
serialised).

## LlamaIndex

```bash
pip install "ragdefender[llama-index]"
# or, for the import path LlamaIndex integrations use
# (from llama_index.postprocessor.ragdefender import RAGDefenderPostprocessor):
pip install llama-index-postprocessor-ragdefender
```

```python
from ragdefender.integrations import RAGDefenderPostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[RAGDefenderPostprocessor(task_type="single_hop")],
)
response = query_engine.query("What is the capital of France?")

# or on a retriever's output directly
nodes = retriever.retrieve(question)
safe_nodes = RAGDefenderPostprocessor(task_type="single_hop").postprocess_nodes(nodes, query_str=question)
```

`RAGDefenderPostprocessor` subclasses `BaseNodePostprocessor`, so it composes
with LlamaIndex's other postprocessors (put it *after* a reranker if you use
one, so the reranker's scores survive on the kept nodes). `metadata_mode`
controls which view of each node is embedded (`MetadataMode.NONE` by default,
i.e. the bare passage text). `to_dict()` / `from_dict()` round-trip the
configuration.

## Haystack 2.x

```bash
pip install "ragdefender[haystack]"
```

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from ragdefender.integrations import RAGDefenderFilter

pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("defender", RAGDefenderFilter(task_type="single_hop"))
pipe.add_component("prompt", PromptBuilder(template=TEMPLATE))
pipe.connect("retriever.documents", "defender.documents")
pipe.connect("defender.documents", "prompt.documents")

result = pipe.run({"retriever": {"query": q}, "defender": {"query": q}, "prompt": {"question": q}})
```

`RAGDefenderFilter.run(documents, query)` returns two outputs: `documents`
(the survivors, in retrieval order) and `removed_documents` (what was dropped,
handy to log or audit). The embedder loads in `warm_up()`, which Haystack calls
before the first pipeline run, so constructing and serialising pipelines stays
cheap; `to_dict()` / `from_dict()` and pipeline YAML work as usual. A component
built with `defender=` cannot be serialised (the instance is not YAML-able).

## Guardrails AI

```bash
pip install "ragdefender[guardrails]"
```

This also installs the `guardrails_ai.ragdefender` namespace module, so the
validator imports the way Guardrails validators are published today
(`from guardrails_ai.ragdefender import RAGDefenderValidator`); the deprecated
`from guardrails.hub import RAGDefenderValidator` shim resolves it as well.

```python
from guardrails import Guard
from ragdefender.integrations import RAGDefenderValidator

guard = Guard().use(RAGDefenderValidator(task_type="single_hop", on_fail="fix"))

context = "\n\n".join(retrieved_passages)              # passages joined by `separator`
outcome = guard.validate(context, metadata={"query": question})
safe_context = outcome.validated_output
```

The validator treats the *retrieved context* as the value under validation. It
passes when no passage is flagged and otherwise fails with a `fix_value`
holding the filtered context, so Guardrails' `on_fail` policy decides what
happens:

| `on_fail` | Effect |
|---|---|
| `"fix"` | transparent filter — `validated_output` is the context minus flagged passages |
| `"exception"` (default) | hard gate — raises `ValidationError` naming the flagged indices |
| `"noop"` | observe only — failure is recorded, context passes through |

The query is read from `metadata["query"]` (key configurable via `query_key`).
Strings are split on `separator` (default blank line); lists of passages are
also accepted when you call `validator.validate(passages, metadata)` directly,
in which case `fix_value` is the filtered list. The class is registered as
`secai-lab/ragdefender` and follows the Guardrails Hub validator contract.

## Where RAGDefender is listed (maintainer notes)

None of the four frameworks takes third-party integration code into its own
repository any more, so "being part of" each one means the following:

- **LangChain** lists external packages in its docs through an *Integration
  listing* issue in [`langchain-ai/docs`](https://github.com/langchain-ai/docs/issues/new/choose)
  (display name `RAGDefenderCompressor`, component `retrievers`, PyPI package
  `ragdefender`). Hosted guides are reserved for packages above 50k monthly
  downloads; everything else gets a provider card that links back to this page.
- **LlamaIndex** stopped accepting new integration packages in the
  `run-llama/llama_index` monorepo; the convention is one standalone PyPI
  package per integration. That is what
  [`packages/llama-index-postprocessor-ragdefender/`](../packages/llama-index-postprocessor-ragdefender)
  is: a thin wrapper that depends on `ragdefender[llama-index]` and exposes
  `llama_index.postprocessor.ragdefender`. The community integrations page of
  the LlamaIndex docs links to it.
- **Haystack**'s integrations page is generated from
  [`deepset-ai/haystack-integrations`](https://github.com/deepset-ai/haystack-integrations);
  `integrations/ragdefender.md` there (type *Custom Component*) documents
  `RAGDefenderFilter`.
- **Guardrails AI** retired the Hub registry in August 2026: validators are now
  plain PyPI packages imported from the `guardrails_ai` namespace, and the
  `guardrails-ai-<name>` distribution prefix is reserved for Guardrails-owned
  validators. The core wheel therefore ships `guardrails_ai/ragdefender/`, so
  `from guardrails_ai.ragdefender import RAGDefenderValidator` works after
  `pip install "ragdefender[guardrails]"` and the `guardrails.hub` shim finds it.
  No separate distribution is needed.

Submissions (opened 2026-11-20):

- LlamaIndex docs, community integrations page: [run-llama/llama_index#24352](https://github.com/run-llama/llama_index/pull/24352)
- Haystack integrations page: [deepset-ai/haystack-integrations#749](https://github.com/deepset-ai/haystack-integrations/pull/749)
- LangChain: file the *Integration listing* issue from the
  [issue form](https://github.com/langchain-ai/docs/issues/new?template=06-integration-submission.yml)
  with the fields above once 0.3.0 is on PyPI.
- Guardrails AI: nothing to submit; the namespace module above is the whole integration.
- Guardrails AI catalog: asked for a third-party listing path in
  [guardrails-ai/guardrails#1709](https://github.com/guardrails-ai/guardrails/issues/1709).
- LlamaIndex example notebook (`docs/examples/node_postprocessor/`): same PR as above.
- Haystack cookbook: proposal issue
  [deepset-ai/haystack-cookbook#324](https://github.com/deepset-ai/haystack-cookbook/issues/324);
  the notebook to submit once a maintainer assigns it is
  [`examples/integrations/haystack_rag_poisoning_defense.ipynb`](../examples/integrations/haystack_rag_poisoning_defense.ipynb).
- Attack benchmarks, as an optional defense flag:
  [sleeepeer/PoisonedRAG#31](https://github.com/sleeepeer/PoisonedRAG/pull/31),
  [HuichiZhou/TrustRAG#8](https://github.com/HuichiZhou/TrustRAG/pull/8).
- OWASP: Top 10 for LLM reference proposal
  [GenAI-LLM-Top10#176](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/issues/176),
  AI Exchange references
  [www-project-ai-security-and-privacy-guide#290](https://github.com/OWASP/www-project-ai-security-and-privacy-guide/pull/290).
  The Solutions Landscape (genai.owasp.org) is a web form and is submitted separately.
