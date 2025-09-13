#!/bin/bash

# PoisonedRAG Defense Evaluation
# This script runs the evaluation for RAGDefender against PoisonedRAG attacks

cd /content/RAGDefender/artifacts/

echo "Running PoisonedRAG defense evaluation..."
echo "Note: Due to GPU memory limitations, using 8-bit quantization and limited model variants (llama-7b, vicuna-7b)"
echo "This may cause slight discrepancies from reported results but demonstrates the same trends."

conda run -n artifact_acsac python run_poisonedrag.py
conda run -n artifact_acsac python eval.py --method PoisonedRAG