# RAGDefender Artifact - ACSAC 2025

Artifacts for paper "Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems" (ACSAC 2025).

## Overview

This artifact contains the implementation and evaluation code for RAGDefender, a novel defense system against knowledge corruption attacks on Retrieval-Augmented Generation (RAG) systems. The artifact includes defense methods and baseline comparisons across multiple datasets and retrieval models.

The evaluation includes comparisons against established attack and defense methods:
- **PoisonedRAG**: Knowledge poisoning attack method from W. Zou, R. Geng, B. Wang, and J. Jia, "PoisonedRAG: Knowledge poisoning attacks to retrieval-augmented generation of large language models," USENIX Security 2025.
- **Blind Defense**: Baseline defense method from H. Tan, F. Sun, W. Yang, Y. Wang, Q. Cao, and X. Cheng, "Blinded by generated contexts: How language models merge generated and retrieved contexts when knowledge conflicts?" ACL 2024.
- **GARAG**: Genetic attack method from S. Cho, S. Jeong, J. Seo, T. Hwang, and J. C. Park, "Typos that broke the RAG's back: Genetic attack on RAG pipeline by simulating documents in the wild via low-level perturbations," EMNLP 2024, pp. 2826–2844.

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (13GB+ VRAM recommended)
- 12GB+ system RAM

## Installation

Run the installation script to set up all dependencies:

```bash
./install.sh
```

This script will:
- Install conda environment with required Python packages
- Download pre-trained models and datasets
- Set up evaluation infrastructure

## Quick Start

The artifact contains three main reproducibility claims that can be evaluated:

### Claim 1: PoisonedRAG Defense Effectiveness
```bash
cd claims/claim1
./run.sh
```

### Claim 2: Blind Defense Method Baseline
```bash
cd claims/claim2
./run.sh
```

### Claim 3: GARAG Defense Method Baseline
```bash
cd claims/claim3
./run.sh
```

## Reproducibility Claims

For each major paper result evaluated under the "Results Reproduced" badge:

```
claims/claim1/
    |------ claim.txt    # Brief description of the paper claim
    |------ run.sh       # Script to produce result
    |------ expected/    # Expected output or validation info
claims/claim2/
    |------ claim.txt    # Brief description of the paper claim
    |------ run.sh       # Script to produce result
    |------ expected/    # Expected output or validation info
claims/claim3/
    |------ claim.txt    # Brief description of the paper claim
    |------ run.sh       # Script to produce result
    |------ expected/    # Expected output or validation info
```

## Expected Results

Each claim generates evaluation results showing:
- Model performance across datasets (NQ, HotpotQA, MS MARCO)
- Accuracy and Attack Success Rate (ASR) metrics
- Comparison across different models (LLaMA-7B, Vicuna-7B)
- Performance with different retrieval models (Contriever, DPR, ANCE)

Expected outputs are provided in `claims/claim*/expected/result.txt` for comparison.

## Technical Notes

Due to computational constraints for artifact evaluation:
- Models are quantized to 8-bit precision to reduce memory usage
- Only LLaMA-7B and Vicuna-7B models are included (vs. larger variants in paper)
- RAGDefender itself does not consume GPU memory; only model loading requires GPU resources
- Results may show slight numerical differences from paper but demonstrate the same performance trends

## Directory Structure

```
artifacts/                  # Main implementation code
   run_poisonedrag.py      # PoisonedRAG evaluation script
   run_blind.py            # Blind defense evaluation script
   run_garag.py            # GARAG defense evaluation script
   eval.py                 # Main evaluation script
   main.py                 # Core evaluation script
   src/                    # Source code modules
   datasets/               # Evaluation datasets
   model_configs/          # Model configuration files
   results/                # Evaluation results
   logs/                   # Execution logs
   poisoned_corpus/        # Poisoned document datasets
   blind/                  # Blind defense results
   GARAG/                  # GARAG defense results

claims/                     # Reproducibility claims
   claim1/                 # PoisonedRAG defense evaluation
   claim2/                 # Blind defense baseline
   claim3/                 # GARAG defense baseline

infrastructure/             # Infrastructure requirements/setup
install.sh                  # Installation script
LICENSE                     # MIT License
use.txt                     # Usage guidelines and limitations
```

## Running Individual Experiments

You can also run evaluations directly using:

```bash
cd artifacts
# PoisonedRAG evaluation
python run_poisonedrag.py
python eval.py --method PoisonedRAG

# Blind defense baseline
python run_blind.py
python eval.py --method Blind

# GARAG defense baseline
python run_garag.py
python eval.py --method GARAG
```

## Evaluation Time

Each claim evaluation takes approximately:
- Claim 1 (PoisonedRAG): 2-4 hours on single GPU
- Claim 2 (Blind): 1-2 hours on single GPU
- Claim 3 (GARAG): 1-2 hours on single GPU

Times may vary based on hardware configuration.

## Troubleshooting

If you encounter memory issues:
- Reduce batch sizes in model configuration files
- Ensure sufficient GPU memory is available
- Try running claims sequentially rather than in parallel

For installation issues:
- Verify CUDA installation and compatibility
- Check Python version requirements
- Ensure sufficient disk space for models and datasets

## Contact

For questions about this artifact, please refer to the paper or contact the authors through the conference proceedings.
