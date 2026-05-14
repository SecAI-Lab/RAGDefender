"""Embedding model factory for RAGDefender.

The paper (§5 Implementation) uses Sentence Transformers with the **Stella**
embedding model. Two non-paper checkpoints have a history in the codebase:

* v0.1.1 ``RAGDefender`` defaulted to ``all-MiniLM-L6-v2``  (this package).
* ``artifacts/main.py:142`` uses                  ``paraphrase-MiniLM-L6-v2``  (research code).

v0.2.0 keeps the **v0.1.1** default (``minilm-all``) so existing callers see
identical filtering after upgrading. The artifact research code keeps its
explicit ``embedder='minilm-paraphrase'`` override unchanged. Switching the
default to Stella is sequenced as Phase 6 (post-0.2.0) and requires
regenerating ``claims/*/expected/result.txt``.

Use::

    >>> from ragdefender.embedders import load_embedder
    >>> emb = load_embedder("minilm-all", device="cpu")     # v0.1.1 default (current)
    >>> emb = load_embedder("minilm-paraphrase")            # what artifacts/main.py uses
    >>> emb = load_embedder("stella")                       # paper-faithful (heavy)
    >>> emb = load_embedder("sentence-transformers/all-MiniLM-L6-v2")  # raw HF id
"""
from __future__ import annotations

from typing import Union

# Friendly preset → Hugging Face model id
PRESETS = {
    "minilm-all": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-paraphrase": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "stella": "dunzhang/stella_en_1.5B_v5",
}
DEFAULT_EMBEDDER = "minilm-all"   # preserves v0.1.1 behavior; Phase 6 will flip to "stella"


def resolve_preset(name: str) -> str:
    """Map a friendly preset (``"stella"``) to the Hugging Face id."""
    return PRESETS.get(name, name)


def load_embedder(name_or_model: Union[str, "object"], device: str = "auto", gpu_id: int = 0):
    """Return a SentenceTransformer ready for ``.encode``.

    Accepts a preset name (``"minilm-paraphrase"``, ``"minilm-all"``, ``"stella"``),
    a raw Hugging Face id (``"sentence-transformers/all-MiniLM-L6-v2"``), or an
    already-instantiated model (which is returned unchanged).

    ``device='auto'`` selects ``cuda`` if available else ``cpu``.
    """
    # Lazy import — keeps `ragdefender` importable even before sentence_transformers
    # finishes its own slow import chain when a caller only needs other utilities.
    if not isinstance(name_or_model, str):
        return name_or_model

    from sentence_transformers import SentenceTransformer

    full_name = resolve_preset(name_or_model)
    needs_remote_code = "stella" in full_name.lower()
    model = SentenceTransformer(full_name, trust_remote_code=needs_remote_code)

    resolved_device = _resolve_device(device)
    if resolved_device == "cuda":
        import torch
        torch.cuda.set_device(gpu_id)
        model = model.to(resolved_device)
    return model


def _resolve_device(device: str) -> str:
    """Map ``"auto"`` to ``"cuda"`` if available, else ``"cpu"``; pass others through."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
