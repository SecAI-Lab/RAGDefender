"""``ragdefender.integrations`` must import without any framework installed."""

from __future__ import annotations

import importlib
import re

import pytest

import ragdefender.integrations as integrations

CASES = [
    ("RAGDefenderCompressor", "langchain_core", "langchain"),
    ("RAGDefenderPostprocessor", "llama_index.core", "llama-index"),
    ("RAGDefenderFilter", "haystack", "haystack"),
    ("RAGDefenderValidator", "guardrails", "guardrails"),
]


def _installed(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def test_package_lists_all_adapters():
    assert set(integrations.__all__) == {name for name, _, _ in CASES}
    assert set(integrations.__all__) <= set(dir(integrations))


@pytest.mark.parametrize("name,framework,extra", CASES)
def test_lazy_attribute_resolves_or_names_the_extra(name, framework, extra):
    if _installed(framework):
        cls = getattr(integrations, name)
        assert cls.__name__ == name
    else:
        with pytest.raises(ImportError, match=re.escape(f"ragdefender[{extra}]")):
            getattr(integrations, name)


def test_unknown_attribute_is_attribute_error():
    with pytest.raises(AttributeError):
        integrations.DoesNotExist  # noqa: B018


def test_core_import_does_not_pull_frameworks():
    """Importing ragdefender must not import any adapter framework as a side effect."""
    import subprocess
    import sys

    code = (
        "import sys, ragdefender, ragdefender.integrations;"
        "frameworks=('langchain_core','llama_index','haystack','guardrails');"
        "bad=[m for m in frameworks if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], timeout=120).returncode == 0
