#!/bin/bash

# install.sh - Setup script for ACSAC Artifact Evaluation on Google Colab
# This script installs condacolab and sets up the conda environment
#
# PREREQUISITES: 
# - RAGDefender repository should already be cloned before running this script
# - This script assumes you're running in Google Colab

set -e  # Exit on any error

echo "=== ACSAC Artifact Installation for Google Colab ==="
echo ""

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if RAGDefender directory exists
if [ ! -d "RAGDefender" ]; then
    echo "Error: RAGDefender directory not found!"
    echo "Please clone the repository first:"
    echo "git clone https://github.com/SecAI-Lab/RAGDefender.git"
    exit 1
fi

# Check if artifacts/env.yml exists
if [ ! -f "RAGDefender/artifacts/env.yml" ]; then
    echo "Error: RAGDefender/artifacts/env.yml not found!"
    echo "Please ensure the repository is properly cloned."
    exit 1
fi

# Phase detection
if check_conda; then
    echo "Conda detected - proceeding with environment setup"
    PHASE=2
else
    echo "Conda not detected - installing condacolab first"
    PHASE=1
fi

if [ "$PHASE" -eq 1 ]; then
    echo "PHASE 1: Installing condacolab..."
    echo ""
    
    pip install condacolab
    
    echo ""
    echo "=========================================="
    echo "CONDACOLAB INSTALLED"
    echo "=========================================="
    echo "IMPORTANT: You need to manually restart the runtime now."
    echo ""
    echo "In Google Colab:"
    echo "1. Go to Runtime -> Restart runtime"
    echo "2. After restart, run this script again"
    echo ""
    echo "Alternatively, run this Python code in a new cell:"
    echo "import condacolab"
    echo "condacolab.install()  # This will restart automatically"
    echo "=========================================="
    
elif [ "$PHASE" -eq 2 ]; then
    echo "PHASE 2: Setting up RAGDefender conda environment..."
    echo ""
    
    echo "Found RAGDefender directory ✓"
    echo "Found artifacts/env.yml ✓"
    echo ""
    
    echo "Creating conda environment from RAGDefender/artifacts/env.yml..."
    conda env create -f RAGDefender/artifacts/env.yml
    echo "Conda environment created successfully ✓"
    
    # Try to extract environment name from yml file
    ENV_NAME=$(grep "name:" RAGDefender/artifacts/env.yml | head -1 | cut -d: -f2 | tr -d ' ')
    if [ ! -z "$ENV_NAME" ]; then
        echo ""
        echo "=========================================="
        echo "INSTALLATION COMPLETE!"
        echo "=========================================="
        echo "Environment name: $ENV_NAME"
        echo "To activate the environment: conda activate $ENV_NAME"
        echo ""
        echo "Next steps:"
        echo "1. conda activate $ENV_NAME"
        echo "2. cd RAGDefender"
        echo "3. Follow the project's usage instructions"
        echo "=========================================="
    else
        echo "Environment created but name could not be determined."
        echo "Check 'conda env list' to see available environments."
    fi
fi