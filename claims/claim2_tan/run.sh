#!/bin/bash
# Tan et al. Defense Baseline Evaluation (ACSAC 2025 artifact, claim 2).
# Runs the Tan et al. (ACL 2024) baseline so its numbers can be compared against
# RAGDefender. Formerly called the "Blind" baseline in v0.1.1.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ARTIFACTS="${REPO_ROOT}/artifacts"
CONDA_ENV="${CONDA_ENV:-artifact_acsac}"

cd "${ARTIFACTS}"

echo "Running Tan et al. defense baseline evaluation..."
echo "  REPO_ROOT = ${REPO_ROOT}"
echo "  CONDA_ENV = ${CONDA_ENV}"
echo "Note: GPU-memory-limited variant (LLaMA-7B, Vicuna-7B, 8-bit quant)."

conda run -n "${CONDA_ENV}" python run_tan.py
conda run -n "${CONDA_ENV}" python eval.py --method Tan
