"""Shared test fixtures.

Phase-1 conftest only sets up determinism; richer fixtures (tiny_embedder,
single_hop_R, multi_hop_R) are added in Phase 2 alongside the grouping/ and
identification/ subpackages they exercise.
"""
import json
import os
import random
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


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


@pytest.fixture
def legacy_singlehop():
    """v0.1.1 single-hop algorithm outputs captured under all-MiniLM-L6-v2."""
    return json.loads((FIXTURES / "legacy_v011_singlehop.json").read_text())


@pytest.fixture
def legacy_multihop():
    """v0.1.1 multi-hop algorithm outputs captured under all-MiniLM-L6-v2."""
    return json.loads((FIXTURES / "legacy_v011_multihop.json").read_text())


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Tests must not touch CUDA — legacy fixtures were captured on CPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
