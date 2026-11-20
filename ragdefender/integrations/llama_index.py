"""LlamaIndex adapter — :class:`RAGDefenderPostprocessor` is a ``BaseNodePostprocessor``.

LlamaIndex runs node postprocessors on the retrieved ``NodeWithScore`` list
before the response synthesizer sees it, which is exactly where RAGDefender
belongs.

Install::

    pip install 'ragdefender[llama-index]'

Use::

    from ragdefender.integrations import RAGDefenderPostprocessor

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[RAGDefenderPostprocessor(task_type="single_hop")],
    )
    response = query_engine.query("What is the capital of France?")

    # or on a retriever's output directly:
    nodes = retriever.retrieve(question)
    postprocessor = RAGDefenderPostprocessor(task_type="single_hop")
    safe_nodes = postprocessor.postprocess_nodes(nodes, query_str=question)

Nodes keep their scores and metadata; retrieval order is preserved.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ragdefender._logging import get_logger
from ragdefender.defender import RAGDefender
from ragdefender.embedders import DEFAULT_EMBEDDER
from ragdefender.integrations._common import (
    DEFAULT_MIN_PASSAGES,
    build_defender,
    missing_dependency,
    partition,
)

try:
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
except ImportError as exc:  # pragma: no cover — exercised only without the extra
    raise missing_dependency("llama-index-core", "llama-index") from exc

logger = get_logger(__name__)

__all__ = ["RAGDefenderPostprocessor"]


class RAGDefenderPostprocessor(BaseNodePostprocessor):
    """Drop adversarial nodes from a retrieved ``List[NodeWithScore]``.

    Configure it like :class:`ragdefender.RAGDefender` (``task_type``,
    ``embedder``, ``device``, ``p``, ``m``) or pass a pre-built instance via
    ``defender=`` to share one embedder across several query engines.

    Args:
        task_type: ``"single_hop"`` (NQ / MS MARCO-style fact lookup) or
            ``"multi_hop"`` (HotpotQA-style). Required unless ``defender`` is given.
        embedder: Preset (``"minilm-all"``, ``"minilm-paraphrase"``, ``"stella"``)
            or Hugging Face id of the sentence-embedding model.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        gpu_id: GPU index when ``device="cuda"``.
        p: Stage-2 frequency-score exponent (paper Eq. 6).
        m: Stage-1 top-``m`` TF-IDF terms (paper Eq. 1).
        min_passages: Retrieved sets smaller than this pass through unfiltered.
        metadata_mode: Which text view of each node to hand to the detector.
            ``MetadataMode.NONE`` (default) uses the bare passage text, matching
            how the paper treats passages; ``EMBED`` includes embed-visible metadata.
        defender: Pre-built :class:`ragdefender.RAGDefender`; when given, the
            constructor knobs above are ignored except ``task_type`` (which, if
            set, overrides the defender's own per call) and ``min_passages``.
    """

    task_type: Optional[str] = Field(
        default=None,
        description="'single_hop' or 'multi_hop' (required unless defender= is given).",
    )
    embedder: str = Field(
        default=DEFAULT_EMBEDDER, description="Embedder preset or Hugging Face id."
    )
    device: str = Field(default="auto", description="'auto', 'cpu' or 'cuda'.")
    gpu_id: int = 0
    p: int = 2
    m: int = 5
    min_passages: int = Field(default=DEFAULT_MIN_PASSAGES, ge=0)
    metadata_mode: MetadataMode = Field(
        default=MetadataMode.NONE, description="Text view of each node passed to the detector."
    )

    _defender: Any = PrivateAttr(default=None)

    def __init__(self, defender: Optional[RAGDefender] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._defender = build_defender(
            defender,
            task_type=self.task_type,
            embedder=self.embedder,
            device=self.device,
            gpu_id=self.gpu_id,
            p=self.p,
            m=self.m,
        )

    @property
    def defender(self) -> RAGDefender:
        """The underlying :class:`ragdefender.RAGDefender` instance."""
        return self._defender

    @classmethod
    def class_name(cls) -> str:
        return "RAGDefenderPostprocessor"

    # ----------------------------------------------------------- LlamaIndex API
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        query = query_bundle.query_str if query_bundle is not None else ""
        texts = [node.node.get_content(metadata_mode=self.metadata_mode) for node in nodes]
        kept, _removed = partition(
            self._defender,
            query,
            texts,
            task_type=self.task_type,
            min_passages=self.min_passages,
        )
        return [nodes[i] for i in kept]
