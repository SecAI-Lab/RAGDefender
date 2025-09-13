#!/bin/bash

# Blind Defense Method Baseline Evaluation
# This script runs the evaluation for the Blind defense method as a baseline comparison

cd /content/RAGDefender/artifacts/

echo "Running Blind defense method evaluation..."
echo "Note: Due to GPU memory limitations, using 8-bit quantization and limited model variants (llama-7b, vicuna-7b)"
echo "This provides baseline performance metrics for comparison with RAGDefender."

conda run -n artifact_acsac python run_blind.py
conda run -n artifact_acsac python eval.py --method Blind