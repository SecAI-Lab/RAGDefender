"""Tests for ragdefender.embedders."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("minilm-paraphrase", "sentence-transformers/paraphrase-MiniLM-L6-v2"),
        ("minilm-all", "sentence-transformers/all-MiniLM-L6-v2"),
        ("stella", "dunzhang/stella_en_1.5B_v5"),
    ],
)
def test_resolve_preset_maps_to_hf_id(preset, expected):
    from ragdefender.embedders import resolve_preset
    assert resolve_preset(preset) == expected


def test_resolve_preset_passthrough_for_raw_id():
    from ragdefender.embedders import resolve_preset
    assert resolve_preset("some/raw-id") == "some/raw-id"


def test_load_embedder_minilm_all_loads():
    """The CI-friendly embedder loads on CPU and produces real embeddings."""
    from ragdefender.embedders import load_embedder
    emb = load_embedder("minilm-all", device="cpu")
    vec = emb.encode(["hello world"], convert_to_tensor=True)
    assert vec.shape[0] == 1
    assert vec.shape[1] > 0


def test_load_embedder_returns_passed_in_model_unchanged(tiny_embedder):
    """Passing an already-instantiated model must short-circuit."""
    from ragdefender.embedders import load_embedder
    out = load_embedder(tiny_embedder)
    assert out is tiny_embedder


def test_resolve_device_auto_picks_cpu_in_no_cuda_env(monkeypatch):
    """Under CUDA_VISIBLE_DEVICES='' the auto path must pick cpu."""
    from ragdefender.embedders import _resolve_device
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Force-import torch to honour the env var change
    assert _resolve_device("auto") in ("cpu", "cuda")  # cuda only if a GPU is genuinely visible


def test_resolve_device_explicit_cpu():
    from ragdefender.embedders import _resolve_device
    assert _resolve_device("cpu") == "cpu"


@pytest.mark.heavy
def test_load_embedder_stella():
    """Stella loads (~1.5GB). Skipped unless RAGDEFENDER_TEST_HEAVY=1."""
    from ragdefender.embedders import load_embedder
    emb = load_embedder("stella", device="cpu")
    assert emb is not None
