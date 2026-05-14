#!/bin/bash
# Script to publish RAGDefender to PyPI
#
# Usage:
#   ./publish.sh         # Publish to PyPI
#   ./publish.sh test    # Publish to TestPyPI (for testing)

set -e  # Exit on error

PUBLISH_TO=${1:-pypi}  # Default to PyPI, or use "test" for TestPyPI

echo "========================================"
echo "RAGDefender Package Publishing Script"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.py or pyproject.toml not found!"
    echo "Please run this script from the RAGDefender root directory."
    exit 1
fi

# Check if build tools are installed
echo "Checking build tools..."
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build tools..."
    pip install --upgrade build twine
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info ragdefender.egg-info

# Build the package
echo ""
echo "Building package..."
python -m build

# Check if build was successful
if [ ! -d "dist" ]; then
    echo "Error: Build failed - dist/ directory not created"
    exit 1
fi

echo ""
echo "Build complete! Generated files:"
ls -lh dist/

# Check the distribution
echo ""
echo "Checking package..."
twine check dist/*

# Upload to PyPI or TestPyPI
echo ""
if [ "$PUBLISH_TO" == "test" ]; then
    echo "Publishing to TestPyPI..."
    echo "You can test installation with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ ragdefender"
    twine upload --repository testpypi dist/*
else
    echo "Publishing to PyPI..."
    echo ""
    echo "WARNING: This will publish to the REAL PyPI!"
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        twine upload dist/*
        echo ""
        echo "Successfully published to PyPI!"
        echo "Users can now install with: pip install ragdefender"
    else
        echo "Publishing cancelled."
        exit 0
    fi
fi

echo ""
echo "========================================"
echo "Publishing complete!"
echo "========================================"
