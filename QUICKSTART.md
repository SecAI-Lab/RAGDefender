# RAGDefender Quickstart

A focused walk-through of how to use the library. For installation, system
requirements, and ACSAC artifact reproduction, see
[`README.md`](README.md). For paper-aligned algorithm details and the
implementation-vs-paper note, see [`docs/algorithm.md`](docs/algorithm.md).

## Minimal example (30 seconds)

```python
from ragdefender import RAGDefender

defender = RAGDefender(task_type="single_hop")    # or "multi_hop"

safe = defender.defend(
    query="What is the capital of France?",
    R=[
        "Paris is the capital of France.",
        "Lyon is the capital of France.",         # adversarial
        "France's capital city is Paris.",
        "The capital of France is Lyon.",         # adversarial
    ],
)
print(safe)
```

That's the whole API. Stage 1 estimates how many passages were poisoned;
Stage 2 picks which specific indices to drop; the survivors are returned.

## Step by step

### 1. Initialize

`task_type` is required. The paper picks the strategy per dataset and so do
you: NQ and MS MARCO use `'single_hop'` (clustering-based grouping); HotpotQA
uses `'multi_hop'` (concentration-based grouping). There is no auto-detect —
the wrong choice will silently degrade results.

```python
from ragdefender import RAGDefender

defender = RAGDefender(
    task_type="single_hop",   # required
    embedder="minilm-all",    # default; "minilm-paraphrase", "stella", or any HF id
    device="auto",            # "auto" | "cuda" | "cpu"
    p=2,                      # frequency-score exponent (paper §4.2 default)
    m=5,                      # top-m TF-IDF terms (paper §4.1 default)
)
```

### 2. Defend a query

```python
safe = defender.defend(query, R)                              # plain list
safe, removed = defender.defend(query, R, return_indices=True)  # with indices
```

`return_indices=True` is useful when you need to log which positions in the
retrieved set were dropped, or when your downstream code keeps a parallel list
of metadata (scores, source URLs).

### 3. Hand the survivors to your generator

```python
context = "\n".join(safe)
prompt = f"Use the following context to answer.\n\n{context}\n\nQuestion: {query}"
answer = your_llm(prompt)
```

The defender doesn't care which generator you use — OpenAI, Anthropic, vLLM,
a local LLaMA. See [`examples/integration_with_retriever.py`](examples/integration_with_retriever.py)
for a complete retriever → defender → generator wiring.

## Common patterns

### Batch processing

`RAGDefender` is reentrant; reuse one instance across queries to avoid
re-loading the embedder.

```python
defender = RAGDefender(task_type="single_hop")
results = {q: defender.defend(q, retrieve(q)) for q in queries}
```

### Mixed single-hop / multi-hop workloads

If you process both kinds of queries, pass `task_type` per call:

```python
defender = RAGDefender(task_type="auto")           # constructor default OK
safe = defender.defend(q, R, task_type="multi_hop")  # but the call must specify
```

`task_type="auto"` at *both* layers raises — explicit is better than implicit.

### Inspecting metrics on labelled data

```python
defender = RAGDefender(task_type="single_hop")
safe = defender.defend(query, R)
metrics = defender.get_metrics(R, safe, poisoned_indices=[2, 3, 5])
# {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, ...}
```

For batch evaluation across a labelled JSON file, use `Evaluator`:

```python
from ragdefender import RAGDefender, Evaluator
ev = Evaluator(RAGDefender(task_type="multi_hop"))
results = ev.evaluate(
    test_data,
    attack_method="poisonedrag",
    task_type="multi_hop",
)
```

`test_data` is a list of `{query, retrieved_docs, poisoned_indices}` dicts.

## Command-line interface

```bash
ragdefender defend \
    --query "What is the capital of France?" \
    --corpus passages.json \
    --task-type single_hop

ragdefender evaluate \
    --test-data test.json \
    --attack poisonedrag \
    --task-type single_hop

ragdefender reproduce claim1     # runs claims/claim1_poisonedrag/run.sh
ragdefender -v defend ...        # -v for INFO, -vv for DEBUG logs
```

`--corpus` accepts either a JSON list, a JSON `{"documents": [...]}` dict,
or a plain text file with one passage per line.

## Choosing the embedder

| Preset | What it is | When to use |
|---|---|---|
| `"minilm-all"` (default) | `sentence-transformers/all-MiniLM-L6-v2`, ~80 MB | Default; matches v0.1.1 behavior |
| `"minilm-paraphrase"` | `sentence-transformers/paraphrase-MiniLM-L6-v2`, ~80 MB | Matches the artifact research code (`artifacts/main.py`) |
| `"stella"` | `dunzhang/stella_en_1.5B_v5`, ~1.5 GB | Paper-faithful; required to reproduce paper numbers exactly. Will become the default in a future release. |

Or pass a raw Hugging Face id: `embedder="BAAI/bge-large-en-v1.5"` etc.
Anything that exposes `.encode(texts, convert_to_tensor=True)` works.

## Tips

- **Determinism**: `RAGDefender.defend` is deterministic for a fixed embedder
  and fixed input; ranking ties in Stage 2 are broken by lower index for
  reproducibility.
- **Empty input**: `defender.defend(query, [])` returns `[]` rather than
  raising. `R` containing one passage is also handled.
- **Memory**: the package uses no GPU memory in steady state — the embedder
  is the only model loaded. RAGDefender itself runs on CPU tensors.
- **Logging**: the package is silent by default. `setup_logging(level=…)`
  or the CLI `-v` flag enables messages.

## Where next

- [`docs/algorithm.md`](docs/algorithm.md) — paper §4 walk-through with the equations and the implementation-vs-paper note.
- [`docs/reproducing-paper.md`](docs/reproducing-paper.md) — which Table / Figure each script produces.
- [`docs/migration-0.1-to-0.2.md`](docs/migration-0.1-to-0.2.md) — moving from v0.1.1 callsites.
- [`examples/`](examples/) — runnable `basic_usage.py` and `integration_with_retriever.py`.
