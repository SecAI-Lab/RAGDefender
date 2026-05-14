"""Phase-1 smoke tests — verify the package still imports after the cleanup.

Phase 2 adds the grouping/, identification/, embedders, similarity test files;
this file stays as the trivial sanity check.
"""
import importlib

import pytest


def test_package_importable():
    """`import ragdefender` works and exposes a __version__ string."""
    mod = importlib.import_module("ragdefender")
    assert isinstance(mod.__version__, str)


def test_public_api_exists():
    """The two pre-0.2.0 public classes are still re-exported from the top level."""
    from ragdefender import RAGDefender, Evaluator
    assert RAGDefender is not None
    assert Evaluator is not None


@pytest.mark.parametrize("removed_subpkg", ["attacks", "datasets", "defenses", "models"])
def test_empty_stub_subpackages_removed(removed_subpkg):
    """The four empty stubs deleted in Phase 1 must NOT come back."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"ragdefender.{removed_subpkg}")


def test_legacy_fixtures_present():
    """Phase-0 captured fixtures are required by the Phase-2 byte-equivalence tests."""
    from pathlib import Path
    fixtures = Path(__file__).parent / "fixtures"
    assert (fixtures / "legacy_v011_singlehop.json").exists()
    assert (fixtures / "legacy_v011_multihop.json").exists()
