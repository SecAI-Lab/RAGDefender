"""Fixtures shared by the framework-adapter tests.

Every adapter test skips itself when its framework is not installed
(``pytest.importorskip``), so this directory is safe to collect in the core
test job. Install the extras to actually run them::

    pip install -e ".[dev,integrations]"
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def shared_defender(tiny_embedder):
    """One CPU RAGDefender reusing the session-cached MiniLM checkpoint."""
    from ragdefender import RAGDefender

    return RAGDefender(embedder=tiny_embedder, task_type="single_hop", device="cpu")


def assert_removed_ground_truth(kept_positions, fixture):
    """Shared assertion: survivors are in retrieval order and exactly the clean ones."""
    kept_positions = list(kept_positions)
    assert kept_positions == sorted(kept_positions), "retrieval order must be preserved"
    removed = set(range(len(fixture["R"]))) - set(kept_positions)
    assert removed == set(fixture["adversarial_indices"])
