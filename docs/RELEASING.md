# RAGDefender Package Release Guide

Quick reference for building and publishing the RAGDefender pip package.

## Pre-Release Checklist

1. Update version in: `ragdefender/__init__.py`, `setup.py`, `pyproject.toml`
2. Ensure dependencies are listed in `requirements.txt` and `pyproject.toml`
3. Test that examples run: `python examples/basic_usage.py`
4. Verify CLI works: `ragdefender --version`

## Building the Package

```bash
# Install tools
pip install --upgrade build twine

# Clean and build
rm -rf build/ dist/ *.egg-info
python -m build

# Verify
twine check dist/*
```

## Testing Locally

```bash
pip install dist/*.whl
python -c "from ragdefender import RAGDefender; print('Success!')"
ragdefender --version
```

## Publishing to PyPI

```bash
# Using publish script
./publish.sh test   # TestPyPI (for testing)
./publish.sh        # PyPI (production)

# Or manually
twine upload --repository testpypi dist/*  # Test
twine upload dist/*                         # Production
```

## PyPI Setup

1. Create account at https://pypi.org/
2. Generate API token at Account Settings → API tokens
3. Store in `~/.pypirc` or enter when prompted

## GitHub Auto-Publishing

1. Add secrets to GitHub: `PYPI_API_TOKEN`, `TEST_PYPI_API_TOKEN`
2. Create and push tag: `git tag v0.1.0 && git push origin v0.1.0`
3. Create GitHub release → Auto-publishes to PyPI

## Common Issues

- **Build fails**: `pip install --upgrade build`
- **Upload rejected**: Increment version, cannot overwrite
- **Import fails**: Check `__init__.py` exports

## Version Numbering

- `0.1.0` - Initial release
- `0.1.1` - Bug fix
- `0.2.0` - New feature
- `1.0.0` - Production ready

---

For help: for8821@g.skku.edu
