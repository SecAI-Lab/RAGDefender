"""Tests for Stage-1 ClusteringBasedGrouping (single-hop QA, paper §4.1)."""
from __future__ import annotations

import pytest

from ragdefender.grouping import ClusteringBasedGrouping


def test_empty_R_returns_zero(tiny_embedder):
    g = ClusteringBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv([]) == 0


def test_default_m_is_5():
    """Paper §5: m = 5 top-TF-IDF terms by default."""
    g = ClusteringBasedGrouping(embedder=None)
    assert g.m == 5


def test_byte_equivalent_with_v011(tiny_embedder, legacy_singlehop):
    """Phase-2 acceptance gate: new ClusteringBasedGrouping must match v0.1.1's
    _find_num_adversarial_agg byte-for-byte on the captured fixture.
    """
    expected = legacy_singlehop["v0_1_1_outputs"]["n_adv_clustering"]
    g = ClusteringBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv(legacy_singlehop["docs"]) == expected


def test_estimate_returns_3_on_single_hop_fixture(tiny_embedder, single_hop_R):
    """Single-hop fixture has 3 ground-truth adversarial passages."""
    g = ClusteringBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv(single_hop_R["R"]) == len(single_hop_R["adversarial_indices"])


def test_n_tfidf_helper_runs(tiny_embedder, single_hop_R):
    """The TF-IDF helper exists and returns a non-negative count <= |R|."""
    g = ClusteringBasedGrouping(tiny_embedder)
    n_tfidf = g._n_tfidf(single_hop_R["R"])
    assert 0 <= n_tfidf <= len(single_hop_R["R"])
