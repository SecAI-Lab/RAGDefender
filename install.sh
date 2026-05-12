#!/bin/bash
# install.sh — entry point router for RAGDefender installation
#
# RAGDefender is published on PyPI as `ragdefender`. Library users do NOT need this
# script — they should `pip install ragdefender`. This script exists only to route
# ACSAC artifact reviewers to the correct Colab/conda installer.

cat <<'EOF'
RAGDefender installation entry point
====================================

For most users (library + CLI):

    pip install ragdefender                        # release from PyPI
    pip install -e ".[dev]"                        # editable install from this checkout

For ACSAC 2025 artifact reviewers running on Google Colab:

    bash install_colab.sh                          # installs condacolab + artifact_acsac env

For ACSAC 2025 artifact reviewers on a local machine with conda:

    conda env create -f artifacts/env.yml          # creates the artifact_acsac env
    conda activate artifact_acsac
    pip install -e .                               # installs ragdefender into that env

See README.md and docs/reproducing-paper.md for the full reproduction workflow.
EOF
exit 0
