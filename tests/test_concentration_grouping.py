"""Tests for Stage-1 ConcentrationBasedGrouping (multi-hop QA, paper §4.1).

Includes a regression test that *pins* the current OR/threshold/flip behaviour
so a future paper-faithful rewrite (Phase 6) cannot land silently.
"""
from __future__ import annotations

import pytest

from ragdefender.grouping import ConcentrationBasedGrouping


def test_empty_R_returns_zero(tiny_embedder):
    g = ConcentrationBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv([]) == 0


def test_byte_equivalent_with_v011(tiny_embedder, legacy_multihop):
    """Phase-2 acceptance gate: new ConcentrationBasedGrouping must match v0.1.1's
    _find_num_adversarial byte-for-byte on the captured fixture.
    """
    expected = legacy_multihop["v0_1_1_outputs"]["n_adv_concentration"]
    g = ConcentrationBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv(legacy_multihop["docs"]) == expected


def test_estimate_returns_2_on_multi_hop_fixture(tiny_embedder, multi_hop_R):
    """Multi-hop fixture has 2 ground-truth adversarial passages."""
    g = ConcentrationBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv(multi_hop_R["R"]) == len(multi_hop_R["adversarial_indices"])


def test_singleton_R_does_not_crash(tiny_embedder):
    """torch.median on a 1-element tensor is the corner case the advisor flagged."""
    g = ConcentrationBasedGrouping(tiny_embedder)
    n = g.estimate_n_adv(["only one passage"])
    assert isinstance(n, int)
    assert 0 <= n <= 1


def test_two_element_R_does_not_crash(tiny_embedder):
    g = ConcentrationBasedGrouping(tiny_embedder)
    n = g.estimate_n_adv(["passage one about cats", "passage two about dogs"])
    assert isinstance(n, int)
    assert 0 <= n <= 2


def test_pin_or_threshold_legacy_behavior_singlehop(tiny_embedder, legacy_singlehop):
    """Regression: even on the single-hop fixture (where this multi-hop strategy
    isn't the right choice), the output must equal what v0.1.1 produced. Future
    paper-faithful rewrites must change the legacy fixtures *intentionally*.
    """
    expected = legacy_singlehop["v0_1_1_outputs"]["n_adv_concentration"]
    g = ConcentrationBasedGrouping(tiny_embedder)
    assert g.estimate_n_adv(legacy_singlehop["docs"]) == expected
