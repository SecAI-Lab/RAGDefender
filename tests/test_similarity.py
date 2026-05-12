"""Tests for ragdefender.similarity helpers."""
from __future__ import annotations

import math

import pytest


def test_n_pairs_for_zero():
    from ragdefender.similarity import n_pairs_for
    assert n_pairs_for(0) == 0
    assert n_pairs_for(-3) == 0


def test_n_pairs_for_one_clamps_to_one():
    """Paper Eq. 4: max(1, C(N_adv, 2)) — C(1,2)=0 must be clamped to 1."""
    from ragdefender.similarity import n_pairs_for
    assert n_pairs_for(1) == 1


def test_n_pairs_for_matches_binomial():
    from ragdefender.similarity import n_pairs_for
    assert n_pairs_for(2) == 1
    assert n_pairs_for(3) == math.comb(3, 2)
    assert n_pairs_for(5) == math.comb(5, 2)


def test_top_similar_pairs_returns_k_pairs(tiny_embedder):
    from ragdefender.similarity import top_similar_pairs
    docs = ["cat", "kitten", "skyscraper", "feline pet", "tall building"]
    embeddings = tiny_embedder.encode(docs, convert_to_tensor=True)
    pairs = top_similar_pairs(embeddings, k=3)
    assert len(pairs) == 3
    sims = [p[1] for p in pairs]
    assert sims == sorted(sims, reverse=True)


def test_top_similar_pairs_indices_are_valid(tiny_embedder):
    from ragdefender.similarity import top_similar_pairs
    docs = ["a", "b", "c", "d"]
    embeddings = tiny_embedder.encode(docs, convert_to_tensor=True)
    pairs = top_similar_pairs(embeddings, k=4)
    n = len(docs)
    for (i, j), _ in pairs:
        assert 0 <= i < j < n


def test_top_similar_pairs_empty_or_singleton(tiny_embedder):
    from ragdefender.similarity import top_similar_pairs
    embeddings = tiny_embedder.encode(["solo"], convert_to_tensor=True)
    assert top_similar_pairs(embeddings, k=5) == []
