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

## Listing RAGDefender in each ecosystem (maintainer notes)

Where the adapters above get discovered, and what each ecosystem expects:

- **LangChain** no longer takes integration code into its repositories. After a
  PyPI release that contains `ragdefender.integrations`, open an *Integration
  listing* issue in [`langchain-ai/docs`](https://github.com/langchain-ai/docs/issues/new/choose)
  (template `06-integration-submission.yml`: display name
  `RAGDefenderCompressor`, language Python, component type `retrievers`, PyPI
  package `ragdefender`). Full hosted guides are reserved for packages with
  ≥ 50k monthly downloads or featured status; everything else gets a listing.
- **LlamaIndex** lists integrations that live in the
  [`run-llama/llama_index`](https://github.com/run-llama/llama_index) monorepo under
  `llama-index-integrations/postprocessor/llama-index-postprocessor-<name>/`
  (`pyproject.toml`, `llama_index/postprocessor/<name>/{__init__,base}.py`,
  `README.md`, `tests/`). A thin `llama-index-postprocessor-ragdefender` package
  that depends on `ragdefender[llama-index]` and re-exports
  `RAGDefenderPostprocessor` is enough to appear on LlamaHub.
- **Haystack**'s integrations page is generated from
  [`deepset-ai/haystack-integrations`](https://github.com/deepset-ai/haystack-integrations):
  add `integrations/ragdefender.md` with the front-matter the other entries use
  (`layout: integration`, `name`, `description`, `authors`, `pypi`, `repo`,
  `type: Custom Component`, `report_issue`, `version: Haystack 2.0`) followed by
  the usage snippet above.
- **Guardrails Hub** validators are standalone repositories created from
  [`guardrails-ai/validator-template`](https://github.com/guardrails-ai/validator-template)
  (`validator/main.py`, `pyproject.toml`, `tests/`). `RAGDefenderValidator`
  already follows the contract, so `main.py` can simply import it from
  `ragdefender.integrations.guardrails`; submission is through the form linked
  from the Hub. Check the Hub docs for the current process before submitting.
