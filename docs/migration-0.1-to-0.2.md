# Migrating from RAGDefender 0.1.1 → 0.2.0

v0.2.0 is a deliberate breaking change. The renamed APIs and reorganized
modules align the code with the paper's terminology so that what you read in
the paper matches what you import. Every v0.1.1 entry point still works for
this release but emits a `DeprecationWarning`; they will be removed in v0.3.0.

If you only use the package as a library (you `import ragdefender` and call
`defender.defend(...)`), the change is small — three kwarg renames, one
behavior fix. If you depend on the artifact directory layout (running
`claims/claim2/run.sh`, importing from `ragdefender.core.defender`, looking up
`artifacts/blind/`), several paths have moved.

## TL;DR

| Concern | Action |
|---|---|
| You only call `RAGDefender().defend(query, docs, mode='multihop')` | Replace `mode='multihop'` with `task_type='multi_hop'`. The old call still works but warns. |
| You import from `ragdefender.core.defender` | Switch to `from ragdefender import RAGDefender`. The shim still works. |
| You consumed the v0.1.1 `defended_docs` output | Be aware the **same input now removes different documents** (correctness fix; see "Behavior changes"). |
| You run the ACSAC artifact claims | `claims/claim{1,2,3}/` was renamed to `claims/claim{1_poisonedrag,2_tan,3_garag}/` and the `run.sh` files no longer hardcode `/content/RAGDefender/...`. |
| You read raw poisoned-corpus files | `artifacts/blind/` was renamed to `artifacts/tan/`. |
| You consumed CLI flags `--mode`, `--attack blind` | Use `--task-type single_hop|multi_hop`, `--attack tan-et-al`. Old flags accepted with warnings. |

## Public-API rename table

### Constructor

| 0.1.1 | 0.2.0 |
|---|---|
| `RAGDefender(similarity_model="…")` | `RAGDefender(embedder="…")` |
| `RAGDefender(device="cuda")` | `RAGDefender(device="auto" \| "cuda" \| "cpu")` (default flipped to `"auto"`; explicit `"cuda"` still works) |
| *(no equivalent)* | `RAGDefender(task_type="single_hop" \| "multi_hop" \| "auto")` — required to pick the right Stage-1 strategy; `"auto"` raises (no heuristic detection) |
| *(no equivalent)* | `RAGDefender(p=2, m=5)` — paper hyperparameters now exposed |

### `defend()`

| 0.1.1 | 0.2.0 |
|---|---|
| `defender.defend(query, retrieved_docs=docs, mode="multihop")` | `defender.defend(query, R=docs, task_type="multi_hop")` |
| `mode="singlehop"` | `task_type="single_hop"` |
| `mode="multihop"`  | `task_type="multi_hop"`  |
| `top_k=N` | (deprecated; slice the result yourself) |
| *(returns only docs)* | `defender.defend(..., return_indices=True)` returns `(safe, removed_indices)` |

### Stage-1 / Stage-2 building blocks (newly public)

```python
# 0.1.1 — private, undocumented
defender._find_num_adversarial(R)        # multi-hop concentration
defender._find_num_adversarial_agg(R)    # single-hop clustering
defender._find_num_adversarial_tfidf(R)  # TF-IDF helper

# 0.2.0 — public, paper-aligned
from ragdefender import (
    ClusteringBasedGrouping, ConcentrationBasedGrouping, IdentifyAdversarial
)
ClusteringBasedGrouping(embedder, m=5).estimate_n_adv(R)
ConcentrationBasedGrouping(embedder).estimate_n_adv(R)
IdentifyAdversarial(embedder, p=2).select(R, n_adv)
```

### Module paths

| 0.1.1 | 0.2.0 |
|---|---|
| `from ragdefender.core.defender import RAGDefender` | `from ragdefender import RAGDefender` |
| `from ragdefender.core.evaluator import Evaluator` | `from ragdefender import Evaluator` |
| (no equivalent) | `from ragdefender.grouping import ClusteringBasedGrouping, ConcentrationBasedGrouping` |
| (no equivalent) | `from ragdefender.identification import IdentifyAdversarial` |
| (no equivalent) | `from ragdefender.embedders import load_embedder` |
| (no equivalent) | `from ragdefender.similarity import top_similar_pairs, n_pairs_for` |
| (no equivalent) | `from ragdefender import setup_logging` |

The `ragdefender.core` and `ragdefender.core.{defender,evaluator}` modules
still import successfully but emit `DeprecationWarning`.

### CLI

| 0.1.1 | 0.2.0 |
|---|---|
| `ragdefender defend --mode singlehop` | `ragdefender defend --task-type single_hop` |
| `ragdefender defend --mode multihop` | `ragdefender defend --task-type multi_hop` |
| `ragdefender evaluate --attack blind` | `ragdefender evaluate --attack tan-et-al` |
| `ragdefender defend --device cuda` | unchanged (default also accepts `auto`) |
| *(no equivalent)* | `ragdefender reproduce {claim1\|claim2\|claim3}` |
| *(no equivalent)* | `ragdefender -v ...` / `-vv` for INFO/DEBUG logging |

The legacy spellings still work and emit `DeprecationWarning`.

### Filesystem

| 0.1.1 path | 0.2.0 path |
|---|---|
| `claims/claim1/`               | `claims/claim1_poisonedrag/` |
| `claims/claim2/`               | `claims/claim2_tan/` |
| `claims/claim3/`               | `claims/claim3_garag/` |
| `artifacts/blind/`             | `artifacts/tan/` |
| `artifacts/blind/blind_*.json` | `artifacts/tan/tan_*.json` |
| `artifacts/run_blind.py`       | `artifacts/run_tan.py` |
| `claims/*/run.sh: cd /content/RAGDefender/artifacts/` | `claims/*/run.sh` resolves `${REPO_ROOT}` from `${BASH_SOURCE}` |

## Behavior changes

These are real semantic differences. Same input → different output. The first
two are *breaking* — they have no transparent fallback and require you to edit
your code.

### 0. `task_type` is now required *(BREAKING — no shim)*

v0.1.1 silently defaulted to `mode='multihop'`. That was the wrong choice on
single-hop datasets (NQ, MS MARCO) and silently degraded results without
warning. v0.2.0 forces the choice: `RAGDefender()` defaults `task_type` to
`"auto"`, and `defend()` raises `ValueError` if neither layer supplies a
concrete value.

```python
# v0.1.1 (worked, often wrong)
defender = RAGDefender()
defender.defend(q, docs)                                 # silently used multi-hop

# v0.2.0 (raises)
defender = RAGDefender()
defender.defend(q, docs)
# ValueError: task_type='auto' requires an explicit override...

# v0.2.0 (correct)
defender = RAGDefender(task_type="single_hop")           # NQ / MS MARCO
defender.defend(q, docs)
# or
defender = RAGDefender()
defender.defend(q, docs, task_type="multi_hop")          # HotpotQA
```

There is no `DeprecationWarning` shim for this — silently picking a default
is exactly what v0.1.1 did wrong. Audit your codebase with
`PYTHONWARNINGS=error::DeprecationWarning` *and* run your tests; missing
`task_type` shows up as a hard `ValueError`, not a warning.

### 1. `defend()` now actually runs Stage 2 (paper §4.2)

This is a **correctness fix**.

In v0.1.1, after Stage 1 estimated $N_{adv}$, `defend()` returned
`R[:len(R) - N_adv]` — i.e. it dropped the *last* $N_{adv}$ entries of `R`.
That works only if the caller pre-sorts `R` so the most-likely-adversarial
passages are at the end. Otherwise, the wrong indices get dropped.

In v0.2.0, `defend()` runs the pair-frequency TopK ranking from paper §4.2
(`IdentifyAdversarial.select`) to identify *which* passage indices are
adversarial, then drops those.

On the captured legacy fixtures:

| Fixture | v0.1.1 F1 | v0.2.0 F1 |
|---|---|---|
| `tests/fixtures/legacy_v011_singlehop.json` | 0.67 | 1.00 |
| `tests/fixtures/legacy_v011_multihop.json`  | 0.50 | 1.00 |

If you were relying on the old behavior (e.g. you pre-sorted `R` and expected
the last $N_{adv}$ to be removed), you now get a different — but algorithmically
correct — result. There is no opt-out for this change in v0.2.0.

### 2. Default embedder semantics

v0.2.0 keeps the v0.1.1 default checkpoint
(`sentence-transformers/all-MiniLM-L6-v2`, exposed as the preset `"minilm-all"`)
so existing callers see identical embeddings. `device="auto"` is the new
constructor default and selects CUDA if available, else CPU; passing
`device="cuda"` explicitly still works the same way.

The paper uses **Stella**, not MiniLM. `RAGDefender(embedder="stella")` works
today but downloads ~1.5 GB and is gated behind `RAGDEFENDER_TEST_HEAVY=1` in
the test suite. The default will flip to Stella in Phase 6 (a separate release)
along with regenerated `claims/*/expected/result.txt` numbers.

### 3. `Evaluator` no longer recreates `RAGDefender` on every call

v0.1.1's `Evaluator.evaluate()` and `evaluate_asr()` each instantiated a
fresh `RAGDefender()` if `self.defender` was `None`, so calling both methods
loaded the embedder twice. v0.2.0 caches the lazily-created defender on
`self.defender`. Pass an explicit `Evaluator(defender=...)` if you want
deterministic instantiation.

## What did NOT change

- Paper §4.1 concentration-grouping algorithm. The implementation differs
  from the paper text in three ways (OR/threshold/flip — see
  [`docs/algorithm.md`](algorithm.md#-implementation-note-the-code-does-not-match-the-paper-text-byte-for-byte)).
  v0.2.0 preserves the v0.1.1 computation byte-for-byte so existing
  `claims/*/expected/result.txt` numbers remain valid. A paper-faithful
  rewrite is Phase 6.
- `Evaluator.evaluate()`, `Evaluator.evaluate_asr()`, `Evaluator.save_results()`,
  `Evaluator.load_results()` signatures (modulo the `defense_mode → task_type`
  alias).
- `RAGDefender.get_metrics()` signature.

## Quickly auditing your code

Run your code with warnings as errors against v0.2.0 to find every callsite
that needs updating:

```bash
PYTHONWARNINGS=error::DeprecationWarning python your_script.py
```

Fix each `DeprecationWarning` it raises and you are done.
