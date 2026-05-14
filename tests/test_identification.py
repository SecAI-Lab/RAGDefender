"""Tests for Stage-2 IdentifyAdversarial (paper §4.2)."""
from __future__ import annotations

import pytest

from ragdefender.identification import IdentifyAdversarial


def test_n_adv_zero_returns_empty(tiny_embedder):
    sel = IdentifyAdversarial(tiny_embedder).select(["a", "b", "c"], n_adv=0)
    assert sel == []


def test_n_adv_one_returns_one_index(tiny_embedder):
    """Paper Eq. 4 clamp: max(1, C(1,2)) = 1; we must still produce one index."""
    sel = IdentifyAdversarial(tiny_embedder).select(
        ["cat", "kitten", "tower"], n_adv=1
    )
    assert len(sel) == 1
    assert sel[0] in (0, 1, 2)


def test_n_adv_geq_len_returns_all(tiny_embedder):
    sel = IdentifyAdversarial(tiny_embedder).select(["a", "b"], n_adv=5)
    assert sorted(sel) == [0, 1]


def test_select_picks_ground_truth_singlehop(tiny_embedder, single_hop_R):
    """On the single-hop fixture (3 Lyon-poisoned docs), Stage 2 with the right
    n_adv must surface exactly those indices.
    """
    n_adv = len(single_hop_R["adversarial_indices"])
    sel = IdentifyAdversarial(tiny_embedder).select(single_hop_R["R"], n_adv=n_adv)
    assert sorted(sel) == sorted(single_hop_R["adversarial_indices"])


def test_select_picks_ground_truth_multihop(tiny_embedder, multi_hop_R):
    n_adv = len(multi_hop_R["adversarial_indices"])
    sel = IdentifyAdversarial(tiny_embedder).select(multi_hop_R["R"], n_adv=n_adv)
    assert sorted(sel) == sorted(multi_hop_R["adversarial_indices"])


def test_p_parameter_changes_score_weighting(tiny_embedder, single_hop_R):
    """p=1 vs p=2 may produce a different ranking even when n_adv coincides."""
    n_adv = len(single_hop_R["adversarial_indices"])
    sel_p1 = IdentifyAdversarial(tiny_embedder, p=1).select(single_hop_R["R"], n_adv=n_adv)
    sel_p4 = IdentifyAdversarial(tiny_embedder, p=4).select(single_hop_R["R"], n_adv=n_adv)
    # Both must return n_adv indices; rankings may differ
    assert len(sel_p1) == n_adv == len(sel_p4)


def test_empty_R_returns_empty(tiny_embedder):
    sel = IdentifyAdversarial(tiny_embedder).select([], n_adv=3)
    assert sel == []
