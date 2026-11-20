"""LangChain adapter — :class:`RAGDefenderCompressor` is a ``BaseDocumentCompressor``.

LangChain's post-retrieval hook is the *document compressor*: anything that
takes ``(documents, query)`` and returns a subset. Wrapping a retriever in a
``ContextualCompressionRetriever`` runs the compressor on every retrieval, so
poisoned passages never reach the prompt.

Install::

    pip install 'ragdefender[langchain]'

Use::

    from ragdefender.integrations import RAGDefenderCompressor

    compressor = RAGDefenderCompressor(task_type="single_hop")   # or "multi_hop"

    # 1. Wrap any retriever (needs `langchain-classic` or `langchain`):
    safe_retriever = compressor.wrap_retriever(vectorstore.as_retriever(search_kwargs={"k": 10}))
    docs = safe_retriever.invoke("What is the capital of France?")

    # 2. Or call it directly on a list of Documents:
    safe_docs = compressor.compress_documents(docs, query)

The compressor keeps LangChain ``Document`` objects intact (metadata, ids) and
preserves retrieval order. ``acompress_documents`` is inherited and runs the
filter in a worker thread.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

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
    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from pydantic import Field, PrivateAttr
except ImportError as exc:  # pragma: no cover — exercised only without the extra
    raise missing_dependency("langchain-core", "langchain") from exc

logger = get_logger(__name__)

__all__ = ["RAGDefenderCompressor"]


class RAGDefenderCompressor(BaseDocumentCompressor):
    """Drop adversarial passages from retrieved LangChain ``Document`` objects.

    Configure it like :class:`ragdefender.RAGDefender` (``task_type``,
    ``embedder``, ``device``, ``p``, ``m``) or pass a pre-built instance via
    ``defender=`` to share one embedder across several chains.

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

    # ------------------------------------------------------------ LangChain API
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Return the documents RAGDefender considers legitimate, in retrieval order."""
        docs = list(documents)
        kept, _removed = partition(
            self._defender,
            query,
            [doc.page_content for doc in docs],
            task_type=self.task_type,
            min_passages=self.min_passages,
        )
        return [docs[i] for i in kept]

    # ------------------------------------------------------------ convenience
    def wrap_retriever(self, base_retriever: Any) -> Any:
        """Return a ``ContextualCompressionRetriever`` that filters ``base_retriever``.

        ``ContextualCompressionRetriever`` lives outside ``langchain-core``; this
        tries ``langchain_classic`` (LangChain >= 1.0) then ``langchain`` (0.x).
        """
        try:
            from langchain_classic.retrievers import ContextualCompressionRetriever
        except ImportError:
            try:
                from langchain.retrievers import ContextualCompressionRetriever  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "ContextualCompressionRetriever needs the `langchain-classic` "
                    "(LangChain >= 1.0) or `langchain` (0.x) package:\n\n"
                    "    pip install langchain-classic\n"
                ) from exc
        return ContextualCompressionRetriever(base_compressor=self, base_retriever=base_retriever)
