#!/bin/bash

# GARAG Defense Method Baseline Evaluation
# This script runs the evaluation for the GARAG defense method as another baseline comparison

cd /content/RAGDefender/artifacts/

echo "Running GARAG defense method evaluation..."
echo "Note: Due to GPU memory limitations, using 8-bit quantization and limited model variants (llama-7b, vicuna-7b)"
echo "This provides another baseline performance metric for comparison with RAGDefender."

conda run -n artifact_acsac python run_garag.py
conda run -n artifact_acsac python eval.py --method GARAG