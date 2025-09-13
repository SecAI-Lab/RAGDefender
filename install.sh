#!/bin/bash

# install.sh - Setup script for ACSAC Artifact Evaluation on Google Colab
# This script installs condacolab, clones the RAGDefender repository, and sets up the environment
#
# IMPORTANT: Due to Colab runtime restart after condacolab installation,
# this script should be run in two phases or follow the manual steps below.

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

# Phase detection
if check_conda; then
    echo "Conda detected - proceeding with Phase 2 (post-restart)"
    PHASE=2
else
    echo "Conda not detected - starting Phase 1 (pre-restart)"
    PHASE=1
fi

if [ "$PHASE" -eq 1 ]; then
    echo "PHASE 1: Installing condacolab..."
    echo "This will restart the Colab runtime automatically."
    echo ""
    
    pip install condacolab
    
    python3 << 'EOF'
import condacolab
print("Installing condacolab - runtime will restart after this...")
condacolab.install()
EOF

    echo ""
    echo "=========================================="
    echo "PHASE 1 COMPLETE"
    echo "The runtime has been restarted."
    echo "Please run this script again to continue with Phase 2."
    echo "=========================================="
    
elif [ "$PHASE" -eq 2 ]; then
    echo "PHASE 2: Setting up RAGDefender environment..."
    echo ""
    
    echo "Step 1: Cloning RAGDefender repository..."
    if [ -d "RAGDefender" ]; then
        echo "RAGDefender directory already exists, removing..."
        rm -rf RAGDefender
    fi
    
    git clone https://github.com/SecAI-Lab/RAGDefender.git
    echo "Repository cloned successfully."
    
    echo ""
    echo "Step 2: Creating conda environment from yml file..."
    
    # Check if environment file exists
    if [ -f "RAGDefender/artifacts/env.yml" ]; then
        conda env create -f RAGDefender/artifacts/env.yml
        echo "Conda environment created successfully."
        
        # Try to extract environment name from yml file
        ENV_NAME=$(grep "name:" RAGDefender/artifacts/env.yml | head -1 | cut -d: -f2 | tr -d ' ')
        if [ ! -z "$ENV_NAME" ]; then
            echo ""
            echo "Environment name: $ENV_NAME"
            echo "To activate: conda activate $ENV_NAME"
        fi
    else
        echo "Error: artifacts/env.yml not found in RAGDefender directory"
        echo "Please check the repository structure."
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "INSTALLATION COMPLETE!"
    echo "=========================================="
fi