#!/bin/bash
# PoisonedRAG Defense Evaluation (ACSAC 2025 artifact, claim 1).
# Runs RAGDefender against PoisonedRAG-poisoned passages on NQ, HotpotQA, MS MARCO.
#
# Resolves paths relative to this script so it works locally, on Colab, or anywhere.
# Override REPO_ROOT or CONDA_ENV via env vars if your layout differs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ARTIFACTS="${REPO_ROOT}/artifacts"
CONDA_ENV="${CONDA_ENV:-artifact_acsac}"

cd "${ARTIFACTS}"

echo "Running PoisonedRAG defense evaluation..."
echo "  REPO_ROOT = ${REPO_ROOT}"
echo "  CONDA_ENV = ${CONDA_ENV}"
echo "Note: Due to GPU memory limits, runs use 8-bit quantization and a limited model set"
echo "(LLaMA-7B, Vicuna-7B). Numbers may differ slightly from the paper but show the same trends."

conda run -n "${CONDA_ENV}" python run_poisonedrag.py
conda run -n "${CONDA_ENV}" python eval.py --method PoisonedRAG
