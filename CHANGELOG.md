# Changelog

All notable changes to RAGDefender are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once 1.0 lands.

## [0.2.0] — 2026-05-12

This release re-aligns the package with the paper's terminology and structure
("Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption
Attacks on RAG Systems" — ACSAC 2025). It is a deliberate breaking change.
Most v0.1.1 entry points still work in this release with a `DeprecationWarning`,
but **two changes have no transparent fallback** and require a code edit:
the new mandatory `task_type` argument and the Stage-2 behavior fix. Both
are called out under "Breaking changes" below.

A complete migration table lives in
[`docs/migration-0.1-to-0.2.md`](docs/migration-0.1-to-0.2.md).

### Breaking changes (no shim, requires a code edit)

- **`task_type` is now required.** `RAGDefender()` defaults `task_type` to
  `"auto"`, and `defend()` raises `ValueError` when called without an explicit
  `task_type` (or `mode`). v0.1.1 silently defaulted to `mode='multihop'`,
  which gave wrong results on single-hop datasets (NQ, MS MARCO).
  Migrate `RAGDefender().defend(q, docs)` →
  `RAGDefender(task_type='multi_hop').defend(q, docs)` (or `'single_hop'`).
- **`defend()` now actually runs Stage 2.** Same input → different `R_safe`.
  See "Behavior changes" in
  [`docs/migration-0.1-to-0.2.md`](docs/migration-0.1-to-0.2.md#1-defend-now-actually-runs-stage-2-paper-42).
  This is a correctness fix; F1 rises from 0.67 → 1.00 on the captured
  single-hop fixture and 0.50 → 1.00 on the multi-hop one. There is no opt-out.

### Added
- **Paper-aligned subpackages** mirroring §4.1 / §4.2 of the paper:
  - `ragdefender.grouping` (Stage 1) — `ClusteringBasedGrouping` (single-hop QA),
    `ConcentrationBasedGrouping` (multi-hop QA), `Grouping` ABC.
  - `ragdefender.identification` (Stage 2) — `IdentifyAdversarial` performs the
    pair-frequency TopK ranking from §4.2 that v0.1.1 silently skipped.
- New utility modules: `ragdefender.embedders` (`load_embedder` factory with
  `minilm-all` / `minilm-paraphrase` / `stella` presets), `ragdefender.similarity`
  (`cos_sim_matrix`, `top_similar_pairs`, `n_pairs_for`), `ragdefender._logging`
  (`setup_logging`, `get_logger`).
- New `RAGDefender` constructor knobs exposing paper hyperparameters:
  `task_type`, `p` (frequency-score exponent), `m` (top-$m$ TF-IDF terms),
  and the new `device="auto"` default.
- New `defend(..., return_indices=True)` returns the dropped indices alongside
  the surviving passages.
- New CLI subcommand `ragdefender reproduce {claim1|claim2|claim3}` that
  delegates to `claims/<claim>/run.sh`.
- New CLI flag `-v / --verbose` (count) wires `setup_logging` to INFO/DEBUG.
- New documentation under [`docs/`](docs/):
  - [`docs/algorithm.md`](docs/algorithm.md) — paper §4 walkthrough with the
    equations and the implementation-vs-paper note.
  - [`docs/reproducing-paper.md`](docs/reproducing-paper.md) — Fig. 4 / 5 +
    Tables 2–8 → script/CLI map, with explicit gaps flagged.
  - [`docs/migration-0.1-to-0.2.md`](docs/migration-0.1-to-0.2.md) — full
    code-level before/after for every renamed API.
  - [`docs/RELEASING.md`](docs/RELEASING.md) — moved from
    `PACKAGE_RELEASE_GUIDE.md`.
- Comprehensive test suite under [`tests/`](tests/) — 8 files, 55 tests
  including byte-equivalence regression tests for both grouping algorithms
  against captured v0.1.1 outputs.
- `.github/workflows/test.yml` — pytest matrix on Python 3.9 / 3.10 / 3.11
  with CPU-only torch.
- New example: `examples/integration_with_retriever.py` — end-to-end
  retriever → defender → generator wiring with mocked components.

### Changed
- **Behavior**: `defend()` now runs Stage 2 to identify *which* indices to
  drop, rather than truncating `R[:|R| - N_adv]` from the end. Same `R` →
  different `R_safe` (closer to ground truth — F1 rises from 0.67 → 1.00 on
  the single-hop fixture, 0.50 → 1.00 on the multi-hop fixture). See the
  [migration doc](docs/migration-0.1-to-0.2.md#1-defend-now-actually-runs-stage-2-paper-42).
- `RAGDefender(similarity_model=…)` → `RAGDefender(embedder=…)` (alias retained).
- `defend(query, retrieved_docs, mode='multihop')` →
  `defend(query, R, task_type='multi_hop')` (aliases retained).
- `Evaluator` lazy-instantiates `RAGDefender` once instead of three times.
- Module flattening: `ragdefender.core.defender` → `ragdefender.defender`,
  `ragdefender.core.evaluator` → `ragdefender.evaluator` (back-compat shims kept).
- The paper now refers to the ACL 2024 attack family as "Tan et al." rather
  than "Blind". All directory / file / CLI / config strings updated:
  - `claims/claim2/`            → `claims/claim2_tan/`
  - `claims/claim1/`            → `claims/claim1_poisonedrag/`
  - `claims/claim3/`            → `claims/claim3_garag/`
  - `artifacts/blind/`          → `artifacts/tan/`
  - `artifacts/run_blind.py`    → `artifacts/run_tan.py`
  - CLI `--attack blind`        → `--attack tan-et-al` (alias retained)
  - `eval.py --method Blind`    → `--method Tan` (alias retained)
- `claims/*/run.sh` now resolve `REPO_ROOT` from `${BASH_SOURCE}` instead of
  hardcoding the Colab `/content/RAGDefender/artifacts/` path. They work
  unchanged on Colab, locally, and in CI.
- `setup.py` removed; `pyproject.toml` is the single source of build metadata.
- `MANIFEST.in` rewritten to match what is actually shipped (was referencing
  several nonexistent paths).
- Bumped to **0.2.0**.

### Deprecated
- `from ragdefender.core import …` and `from ragdefender.core.{defender,evaluator}`
  imports — use `from ragdefender import …` instead.
- `RAGDefender(similarity_model=…)`, `defend(retrieved_docs=…, mode=…, top_k=…)`
  kwargs.
- `Evaluator(defense_mode=…)` kwarg.
- CLI `--mode {singlehop|multihop}`, `--attack blind`.
- `eval.py --method Blind`.

All deprecated paths emit `DeprecationWarning` and route to the new equivalent
in this release; they are removed in 0.3.0.

### Removed
- Empty stub subpackages: `ragdefender/{attacks,datasets,defenses,models}/`.
- `setup.py` — superseded by `pyproject.toml`.
- `PACKAGE_SUMMARY.md` — stale (described removed `method='isolation'`).
- Pre-built distributions (`dist/`) and `ragdefender.egg-info/` from the repo;
  both regenerated by `python -m build` and `pip install -e .` respectively.

### Fixed
- `pyproject.toml` no longer references the `YOUR_PAPER_URL` placeholder; the
  ACSAC 2025 DOI (`10.1109/ACSAC67867.2025.00093`) is now the canonical link.
- `pyproject.toml`'s `[tool.setuptools.package-data]` no longer references
  `datasets/`, `model_configs/`, `templates/`, `resources/` — none of which
  existed inside the package.
- `.gitignore` no longer ignores `claims/`, `PACKAGE_RELEASE_GUIDE.md`,
  `PACKAGE_SUMMARY.md`, or `publish.sh` (which silently kept these tracked
  files invisible to `git status`).
- `artifacts/run_garag.py` had a leftover `# … and blind` comment from the
  copy-paste it was forked from.

### Known follow-ups (not in 0.2.0)
- **Phase 3 — artifacts/main.py dedup**. The defense algorithm is implemented
  twice today: once in `ragdefender/grouping/*` and once in
  `artifacts/main.py:23-65`. The plan is for `artifacts/main.py` to import
  from the package; this requires running the byte-equivalence acceptance
  gate (old vs new `main.py` on the same poisoned-corpus seed) which needs
  the `artifact_acsac` conda env. Deferred until that env is set up.
- **Phase 6 — paper-faithful defaults**.
  - The default embedder will flip from `minilm-all` to `stella` (paper §5).
  - The `ConcentrationBasedGrouping` formula will be rewritten to match the
    paper text (§4.1: AND, not OR; threshold = `median(s^median)`, not
    `(median+mean)/2`; no result-flip branch). Both invalidate the existing
    `claims/*/expected/result.txt`; expected outputs need regenerating and
    the version bumps to 0.3.0.

## [0.1.1] — 2025-10-25

Last release before the v0.2.0 restructure. See `git log v0.1.1` for the
detailed history. Notable points:

- Single defense method exposed via `defender.defend(query, docs, mode=…)`
  with `mode ∈ {'singlehop', 'multihop'}`.
- Replaced an earlier short-lived API that exposed three separate methods
  (`isolation` / `aggregation` / `filtering`) — those did not match the paper.
- CLI: `ragdefender {defend, evaluate, info}`.

## [0.1.0] — initial release

Initial PyPI release.
