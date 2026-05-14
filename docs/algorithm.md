# RAGDefender — Algorithm Reference

This document mirrors the paper's §4 in code-friendly form. It is the canonical
explanation of what `ragdefender.RAGDefender` actually does, and is the place
where any divergence between the implementation and the paper text is recorded.

> **Citation.** Kim, M., Lee, H., Koo, H. *Rescuing the Unpoisoned: Efficient
> Defense against Knowledge Corruption Attacks on RAG Systems.* ACSAC 2025.
> [DOI: 10.1109/ACSAC67867.2025.00093](https://doi.org/10.1109/ACSAC67867.2025.00093)

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| $q$ | User query | `query` |
| $\mathcal{R} = \{r_1, \ldots, r_k\}$ | Retrieved passages | `R` |
| $\mathcal{R}_{adv}$ | Adversarial subset of $\mathcal{R}$ | indices removed by `defend()` |
| $\mathcal{R}_{safe} = \mathcal{R} \setminus \mathcal{R}_{adv}$ | Surviving passages | return value of `defend()` |
| $N_{adv} = \|\mathcal{R}_{adv}\|$ | Number of adversarial passages | `n_adv` |
| $T_{top}$ | Top-$m$ TF-IDF terms across $\mathcal{R}$ | computed in `_n_tfidf` |
| $m$ | Top-TF-IDF term count | `RAGDefender(m=…)`, default `5` |
| $p$ | Frequency-score weighting exponent | `RAGDefender(p=…)`, default `2` |
| $s^{mean}_i$, $s^{median}_i$ | Per-passage mean/median pairwise cosine similarity | computed in `concentration.py` |

## Overview

RAGDefender is a two-stage post-retrieval filter:

```
                      ┌─────────────────────┐
       R = {r_1, …}   │  Stage 1: Grouping  │   N_adv ∈ [0, |R|]
   ───────────────►   │   (paper §4.1)      │  ───────────────►
                      └─────────────────────┘
                                                          │
                                                          ▼
                                       ┌─────────────────────────────────┐
                                       │   Stage 2: Identification       │
                                       │       (paper §4.2)              │
                                       │  pair-frequency TopK ranking    │
                                       └─────────────────────────────────┘
                                                          │
                                                          ▼
                                                 R_adv (indices)
                                                          │
                                                          ▼
                                          R_safe = R \ R_adv  ──►  generator
```

Stage 1 picks one of two grouping strategies based on the query's task type:
**clustering-based** for single-hop QA (NQ, MS MARCO), **concentration-based**
for multi-hop QA (HotpotQA). Both produce a single integer $N_{adv}$. Stage 2
then ranks individual passages and returns the indices of the top $N_{adv}$.

## Stage 1 — Grouping Retrieved Passages (paper §4.1)

### Clustering-based Grouping (single-hop QA)

Code: [`ragdefender/grouping/clustering.py`](../ragdefender/grouping/clustering.py)
→ `ClusteringBasedGrouping.estimate_n_adv`.

1. Embed $\mathcal{R}$ with the configured embedder.
2. Hierarchical agglomerative clustering with `n_clusters=2` over the
   embeddings; let $n_{min}$ be the smaller cluster size.
3. **Eq. 1** — top-$m$ TF-IDF vote, with English stopwords removed:
   ```math
   N_{TF-IDF} = \sum_{i=1}^{|\mathcal{R}|}
                \mathbf{1}\!\left(
                    \sum_{j=1}^{m} \mathbf{1}(t_j \in r_i) > \lfloor m/2 \rfloor
                \right)
   ```
   `m = 5` by default (paper §5).
4. **Eq. 2** — the term-frequency vote tells us whether the adversarial
   passages form the minority or the majority cluster:
   ```math
   N_{adv} =
   \begin{cases}
     n_{min}                 & \text{if } N_{TF-IDF} \leq \lfloor |\mathcal{R}|/2 \rfloor \\
     |\mathcal{R}| - n_{min} & \text{otherwise}
   \end{cases}
   ```

### Concentration-based Grouping (multi-hop QA)

Code: [`ragdefender/grouping/concentration.py`](../ragdefender/grouping/concentration.py)
→ `ConcentrationBasedGrouping.estimate_n_adv`.

1. Embed $\mathcal{R}$ and compute the $|\mathcal{R}| \times |\mathcal{R}|$
   pairwise cosine similarity matrix.
2. For each passage compute its concentration factors:
   ```math
   s^{mean}_i = \frac{1}{|\mathcal{R}| - 1}
                \sum_{j \neq i} \mathrm{sim}(r_i, r_j),
   \qquad
   s^{median}_i = \mathrm{median}_{j \neq i}\,\mathrm{sim}(r_i, r_j)
   ```
3. **Paper Eq. 3** *(text)* — count passages whose mean *and* median exceed
   the global mean and median:
   ```math
   N_{adv}^{paper} = \sum_i
       \mathbf{1}(s^{mean}_i > \bar s)\cdot\mathbf{1}(s^{median}_i > \tilde s)
   ```
   where $\bar s$ and $\tilde s$ are the global mean of $s^{mean}_i$ and
   global median of $s^{median}_i$.

#### ⚠️ Implementation note: the code does *not* match the paper text byte-for-byte

`ConcentrationBasedGrouping.estimate_n_adv` (preserved from
`ragdefender.core.defender._find_num_adversarial` in v0.1.1) differs from
Eq. 3 in three ways:

1. **OR rather than AND**: a passage is counted if its mean or median
   concentration is high.
2. **Different median threshold**: the cutoff is
   `(global_median + global_mean) / 2` rather than the paper's
   `median(s^{median})`.
3. **Result-flipping branch**: when `global_mean ≥ global_median`, the result
   is replaced with $|\mathcal{R}| - \text{sum}$.

The legacy implementation is kept verbatim because the published
`claims/*/expected/result.txt` were produced by it, and a silent rewrite would
invalidate the artifact-evaluation reproducibility numbers. A paper-faithful
implementation lands as part of Phase 6 (a separate behavior-change release)
along with regenerated expected outputs.

The contract is locked by
[`tests/test_concentration_grouping.py::test_pin_or_threshold_legacy_behavior_singlehop`](../tests/test_concentration_grouping.py)
— any future paper-faithful rewrite **must** intentionally update that test
and the legacy fixtures, not silently change the computation.

## Stage 2 — Identifying Adversarial Passages (paper §4.2)

Code: [`ragdefender/identification/topk.py`](../ragdefender/identification/topk.py)
→ `IdentifyAdversarial.select`.

Given $N_{adv}$ from Stage 1:

1. **Eq. 4** — number of pairs to consider:
   ```math
   N_{pairs} = \max\!\left(1,\, \binom{N_{adv}}{2}\right)
   ```
2. **Eq. 5** — take the top-$N_{pairs}$ most-similar passage pairs in $\mathcal{R}$:
   ```math
   \mathcal{P}_{top} = \mathrm{TopK}\!\left(
       \{(r_i, r_j) \in \mathcal{R} \times \mathcal{R} \mid i \neq j\},\,
       N_{pairs},\, \mathrm{sim}(r_i, r_j)
   \right)
   ```
3. **Eq. 6** — frequency score for each passage:
   ```math
   f_i = \sum_{(r_i, r_j) \in \mathcal{P}_{top}}
         \mathrm{sgn}(\mathrm{sim}(r_i, r_j)) \cdot |\mathrm{sim}(r_i, r_j)|^p
   ```
   $p = 2$ by default (paper §5; ablation in Table 7).
4. **Eq. 7** — the adversarial set is the top-$N_{adv}$ by frequency score:
   ```math
   \mathcal{R}_{adv} = \mathrm{TopK}(\{r_i \mid r_i \in \mathcal{R}\},\, N_{adv},\, f_i)
   ```

The remaining passages form $\mathcal{R}_{safe} = \mathcal{R} \setminus \mathcal{R}_{adv}$
and are returned to the generator.

> **v0.2.0 behavior change.** v0.1.1's `RAGDefender.defend()` did **not** run
> Stage 2. It just returned `R[:|R| - N_adv]`, which is correct only if the
> caller already sorted $\mathcal{R}$ so adversarial passages were at the end.
> v0.2.0 routes through `IdentifyAdversarial.select` and so handles
> arbitrarily-ordered $\mathcal{R}$. On the captured single-hop fixture this
> raises F1 from 0.67 to 1.0; on the multi-hop fixture from 0.50 to 1.0.
> See [`docs/migration-0.1-to-0.2.md`](migration-0.1-to-0.2.md).

## Mapping back to the public API

| Paper concept | Public API |
|---|---|
| Whole pipeline | `RAGDefender(task_type=…).defend(query, R)` |
| Stage 1, single-hop | `ClusteringBasedGrouping(embedder, m=5).estimate_n_adv(R)` |
| Stage 1, multi-hop | `ConcentrationBasedGrouping(embedder).estimate_n_adv(R)` |
| Stage 2 | `IdentifyAdversarial(embedder, p=2).select(R, n_adv)` |
| Embedder | `load_embedder("minilm-all" | "minilm-paraphrase" | "stella")` |

## Hyperparameter defaults (paper §5)

| Symbol | Default | Set via |
|---|---|---|
| $m$ | 5 | `RAGDefender(m=…)` |
| $p$ | 2 | `RAGDefender(p=…)` |
| $k$ (top-$k$ retrieval) | 5 (NQ, MS MARCO), 2 (HotpotQA) | upstream of RAGDefender; set in `artifacts/main.py` |
| Embedder | Stella (paper) | v0.2.0 default is `minilm-all`; flips to Stella in Phase 6 (see [migration](migration-0.1-to-0.2.md)) |
