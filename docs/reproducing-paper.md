# Reproducing the Paper

This page maps each table and figure in the ACSAC 2025 paper to the script(s)
in this repository that produce it. The three ACSAC artifact-evaluation
"claims" cover Figure 4 / Figure 5 numbers across the three attack baselines;
several other tables are produced as side effects, and a couple are not yet
fully automated. Those gaps are flagged.

> **Hardware.** The paper used one Quadro RTX 8000 (~48 GB). 8-bit quantization
> in the artifact lets it run on a single 16 GB GPU but produces slightly
> different numbers (same trends).

> **Conda env.** All `artifacts/*` scripts assume the `artifact_acsac` env from
> [`artifacts/env.yml`](../artifacts/env.yml). See [`README.md`](../README.md#installation)
> for setup. Override with `CONDA_ENV=<name>` in your environment.

## The three ACSAC claims

Each claim runs one attack baseline end-to-end across NQ, HotpotQA, MS MARCO
on Contriever + LLaMA-7B + Vicuna-7B (8-bit), then aggregates ASR / accuracy.

| Claim | Attack baseline | Paper section | Script | Approximate runtime |
|---|---|---|---|---|
| `claim1_poisonedrag` | PoisonedRAG (Zou et al., USENIX'25) | §6.2, Fig. 4 col. 1 | [`claims/claim1_poisonedrag/run.sh`](../claims/claim1_poisonedrag/run.sh) | 4–5 h on a single GPU |
| `claim2_tan`         | Tan et al. (ACL'24) — formerly "Blind" | §6.2, Fig. 4 col. 3 | [`claims/claim2_tan/run.sh`](../claims/claim2_tan/run.sh) | 1–2 h |
| `claim3_garag`       | GARAG (Cho et al., EMNLP'24) | §6.2, Fig. 4 col. 2 | [`claims/claim3_garag/run.sh`](../claims/claim3_garag/run.sh) | 1–2 h |

Run any claim either via the CLI:

```bash
ragdefender reproduce claim1   # claim1 | claim2 | claim3
```

…or directly:

```bash
bash claims/claim1_poisonedrag/run.sh
```

Outputs land under `artifacts/logs/main_logs_<METHOD>_12/` and `artifacts/results/`.
Compare each `claims/<claim>/expected/result.txt` against the script's stdout
to validate.

## Tables and figures coverage

| Paper artifact | What it shows | Reproducibility |
|---|---|---|
| **Figure 4** | ASR + Accuracy across attacks (PoisonedRAG, GARAG, Tan et al.) × datasets (NQ, HotpotQA, MS MARCO) at perturbation ratios 1× / 2× / 4× / 6×, GPT-4o generator. | Partially: claims 1–3 cover the open-source-LLM rows. The GPT-4o row needs API access; replace `gpt4o` in `model_configs/` with your key and re-run the same scripts. |
| **Figure 5** | ASR + Accuracy across LLMs (LLaMA-7B/13B, Vicuna-7B/13B, GPT-4o, Gemini-1.5-pro) × datasets, PoisonedRAG only. | Partially: open-source LLM cells produced by `claim1_poisonedrag` with the appropriate `model_name` swapped in `artifacts/run_poisonedrag.py`. Commercial-LLM cells need API keys. |
| **Table 2** (cost & speed) | $/iter and seconds/iter for RAGDefender vs. RobustRAG. | Manual: timing instrumentation lives in `artifacts/main.py`'s loop. Not currently aggregated by `eval.py` — see open task in CHANGELOG. |
| **Table 3** (GPU memory) | Memory footprint at fine-tuning / inference. | Manual: collect with `nvidia-smi --query-gpu=memory.used` while a claim runs. RAGDefender row should report 0 (no GPU usage). |
| **Table 4** (different RAG architectures) | RAGDefender on BlendedRAG, REPLUG, SELF-RAG. | **Not in the artifact** — the upstream RAG implementations are external. The paper points to their respective repos; integrate them with the package as `from ragdefender import RAGDefender; defender.defend(...)` in their retrieval loop. |
| **Table 5** (different retrievers) | RAGDefender across Contriever, DPR, ANCE × LLMs × datasets, MS MARCO focus. | Re-run claim 1 with `eval_model_code` in [`artifacts/run_poisonedrag.py`](../artifacts/run_poisonedrag.py) set to `dpr` and `ance`. Currently the artifact ships only the Contriever rows because the paper notes this is sufficient to demonstrate retriever-agnosticism. |
| **Table 6** (clustering algorithm ablation) | Agglomerative vs. K-Means vs. DBSCAN. | Set `--method tan` then patch `ragdefender/grouping/clustering.py::ClusteringBasedGrouping` to swap the clustering call. Not exposed via CLI — requires a one-line code edit and one re-run. |
| **Table 7** ($p$, $m$ ablation) | Frequency-score exponent and TF-IDF top-$m$ ablation. | Pass `--p P --m M` to `artifacts/main_abl.py`; ranges `p ∈ {1,2,3}`, `m ∈ {3,5,7}` per the paper. |
| **Table 8** (Stage 1 vs Stage 2 vs combined) | Demonstrates value of running both stages. | Stage 1-only baseline: call `ClusteringBasedGrouping.estimate_n_adv(R)` and drop `R[:N_adv]` (matches v0.1.1 behavior). Stage 2-only baseline: call `IdentifyAdversarial.select(R, k)` with a fixed $k$. Combined: default `RAGDefender.defend`. Combine the three numbers manually. |

## What's NOT automated

These need follow-up work; tracked as ideas in [`CHANGELOG.md`](../CHANGELOG.md#unreleased)
under "Future work":

- Per-figure orchestration script (`ragdefender reproduce fig4` etc.) — the
  three `claims/*` cover the bulk of Figure 4 but a single command-per-figure
  experience would be friendlier.
- Commercial-LLM rows (GPT-4o, Gemini-1.5-pro) require user-supplied API keys;
  a documented workflow with `.env`-style key handling is missing.
- Table 4 (BlendedRAG, REPLUG, SELF-RAG) requires integration with three
  external repos; the package supports it (the API is intentionally generic),
  but no glue is shipped.
- Table 6 (clustering algorithm ablation) needs a CLI knob to swap clustering
  algorithms without editing source.

## A note on the algorithm divergence

The `ConcentrationBasedGrouping` implementation differs from the paper's
Eq. 3 in three documented ways (OR vs AND, threshold definition, result flip).
See [`docs/algorithm.md` § Implementation note](algorithm.md#-implementation-note-the-code-does-not-match-the-paper-text-byte-for-byte).
The artifact's `expected/result.txt` files were generated by the legacy
implementation, so the numbers you reproduce will match the legacy
implementation, **not** a paper-faithful one. A paper-faithful concentration
grouping is gated to a future release (Phase 6) with regenerated expectations.
