"""Tests for the ragdefender CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CLI = [sys.executable, "-m", "ragdefender.cli"]


def _run(*args, **kwargs):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        CLI + list(args),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        **kwargs,
    )


def test_info_returns_zero_and_shows_version():
    res = _run("info")
    assert res.returncode == 0, res.stderr
    assert "0.2.0" in res.stdout
    assert "ACSAC" in res.stdout


def test_help_lists_subcommands():
    res = _run("--help")
    assert res.returncode == 0
    for sub in ("defend", "evaluate", "info", "reproduce"):
        assert sub in res.stdout


def test_defend_smoke(tmp_path: Path, single_hop_R):
    """End-to-end CLI invocation against the single-hop fixture."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(single_hop_R["R"]))
    out = tmp_path / "out.json"
    res = _run(
        "defend",
        "--query", single_hop_R["query"],
        "--corpus", str(corpus),
        "--task-type", "single_hop",
        "--device", "cpu",
        "--output", str(out),
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["query"] == single_hop_R["query"]
    assert payload["num_removed"] == len(single_hop_R["adversarial_indices"])


def test_attack_alias_blind_warns(tmp_path: Path, single_hop_R):
    """--attack blind still works but emits a DeprecationWarning to stderr."""
    test_data = tmp_path / "td.json"
    test_data.write_text(
        json.dumps(
            [
                {
                    "query": single_hop_R["query"],
                    "retrieved_docs": single_hop_R["R"],
                    "poisoned_indices": single_hop_R["adversarial_indices"],
                }
            ]
        )
    )
    out = tmp_path / "ev.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONWARNINGS": "always"}
    res = subprocess.run(
        CLI
        + [
            "evaluate",
            "--test-data", str(test_data),
            "--attack", "blind",
            "--task-type", "single_hop",
            "--device", "cpu",
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert "DeprecationWarning" in res.stderr or "blind" in res.stderr.lower()


def test_task_type_alias_singlehop_warns(tmp_path: Path, single_hop_R):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(single_hop_R["R"]))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONWARNINGS": "always"}
    res = subprocess.run(
        CLI
        + [
            "defend",
            "--query", single_hop_R["query"],
            "--corpus", str(corpus),
            "--task-type", "singlehop",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    # The deprecation warning is emitted via warnings module → goes to stderr.
    assert "v0.1.1 spelling" in res.stderr or "singlehop" in res.stderr


def test_reproduce_unknown_claim_rejected():
    res = _run("reproduce", "claim42")
    assert res.returncode != 0
    # argparse error
    assert "invalid choice" in res.stderr.lower()
