"""Shared test fixtures."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
# v0.1.1 default embedder; legacy_v011_*.json fixtures were captured under it.
_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Seed Python and (lazily) numpy/torch for every test."""
    random.seed(42)
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass
    yield


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Tests must not touch CUDA — legacy fixtures were captured on CPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="session")
def tiny_embedder():
    """Cached SentenceTransformer used by every grouping/identification test.

    Same checkpoint that produced the legacy_v011_*.json fixtures so byte-equivalence
    tests can compare apples to apples. Downloaded on first use; ~80MB.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBEDDER_NAME)


@pytest.fixture
def legacy_singlehop():
    """v0.1.1 single-hop algorithm outputs captured under all-MiniLM-L6-v2."""
    return json.loads((FIXTURES / "legacy_v011_singlehop.json").read_text())


@pytest.fixture
def legacy_multihop():
    """v0.1.1 multi-hop algorithm outputs captured under all-MiniLM-L6-v2."""
    return json.loads((FIXTURES / "legacy_v011_multihop.json").read_text())


@pytest.fixture
def single_hop_R(legacy_singlehop):
    """The single-hop retrieved set ``R`` plus its ground-truth adversarial indices."""
    return {
        "query": legacy_singlehop["query"],
        "R": legacy_singlehop["docs"],
        "adversarial_indices": legacy_singlehop["ground_truth_adversarial_indices"],
    }


@pytest.fixture
def multi_hop_R(legacy_multihop):
    """The multi-hop retrieved set ``R`` plus its ground-truth adversarial indices."""
    return {
        "query": legacy_multihop["query"],
        "R": legacy_multihop["docs"],
        "adversarial_indices": legacy_multihop["ground_truth_adversarial_indices"],
    }


def pytest_configure(config):
    """Register the ``heavy`` marker for tests that need RAGDEFENDER_TEST_HEAVY=1."""
    config.addinivalue_line(
        "markers",
        "heavy: requires RAGDEFENDER_TEST_HEAVY=1 (large model download, e.g. Stella)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``heavy`` tests unless explicitly enabled."""
    if os.environ.get("RAGDEFENDER_TEST_HEAVY"):
        return
    skip_heavy = pytest.mark.skip(reason="set RAGDEFENDER_TEST_HEAVY=1 to enable")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
