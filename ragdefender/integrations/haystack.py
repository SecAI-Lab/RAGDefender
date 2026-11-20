"""Haystack 2.x adapter — :class:`RAGDefenderFilter` is a ``@component``.

It takes ``documents`` and ``query`` and emits the surviving ``documents`` (plus
``removed_documents`` for logging/auditing), so it slots between the retriever
and the prompt builder::

    retriever ──► RAGDefenderFilter ──► PromptBuilder ──► Generator

Install::

    pip install 'ragdefender[haystack]'

Use::

    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    from ragdefender.integrations import RAGDefenderFilter

    pipe = Pipeline()
    pipe.add_component("retriever", retriever)
    pipe.add_component("defender", RAGDefenderFilter(task_type="single_hop"))
    pipe.add_component("prompt", PromptBuilder(template=TEMPLATE))
    pipe.connect("retriever.documents", "defender.documents")
    pipe.connect("defender.documents", "prompt.documents")

    pipe.run({"retriever": {"query": q}, "defender": {"query": q}, "prompt": {"question": q}})

The embedder is loaded in ``warm_up()`` (Haystack calls it before the first
run), so building and serialising a pipeline stays cheap.
"""

from typing import Any, Dict, List, Optional

from ragdefender._logging import get_logger
from ragdefender.defender import RAGDefender
from ragdefender.embedders import DEFAULT_EMBEDDER
from ragdefender.integrations._common import (
    DEFAULT_MIN_PASSAGES,
    build_defender,
    ensure_task_type,
    missing_dependency,
    partition,
)

try:
    from haystack import Document, component, default_from_dict, default_to_dict
except ImportError as exc:  # pragma: no cover — exercised only without the extra
    raise missing_dependency("haystack-ai", "haystack") from exc

logger = get_logger(__name__)

__all__ = ["RAGDefenderFilter"]


@component
class RAGDefenderFilter:
    """Drop adversarial passages from a list of Haystack ``Document`` objects.

    Configure it like :class:`ragdefender.RAGDefender` (``task_type``,
    ``embedder``, ``device``, ``p``, ``m``) or pass a pre-built instance via
    ``defender=`` to share one embedder across several pipelines.

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
            A component built this way cannot be serialised with ``to_dict()``.
    """

    def __init__(
        self,
        task_type: Optional[str] = None,
        embedder: str = DEFAULT_EMBEDDER,
        device: str = "auto",
        gpu_id: int = 0,
        p: int = 2,
        m: int = 5,
        min_passages: int = DEFAULT_MIN_PASSAGES,
        defender: Optional[RAGDefender] = None,
    ) -> None:
        ensure_task_type(defender, task_type)
        self.task_type = task_type
        self.embedder = embedder
        self.device = device
        self.gpu_id = gpu_id
        self.p = p
        self.m = m
        self.min_passages = min_passages
        self._defender: Optional[RAGDefender] = defender

    @property
    def defender(self) -> Optional[RAGDefender]:
        """The underlying :class:`ragdefender.RAGDefender` (``None`` until ``warm_up``)."""
        return self._defender

    def warm_up(self) -> None:
        """Load the embedder. Haystack calls this before the pipeline's first run."""
        if self._defender is None:
            self._defender = build_defender(
                None,
                task_type=self.task_type,
                embedder=self.embedder,
                device=self.device,
                gpu_id=self.gpu_id,
                p=self.p,
                m=self.m,
            )

    # -------------------------------------------------------------- Haystack API
    @component.output_types(documents=List[Document], removed_documents=List[Document])
    def run(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """Filter ``documents`` for ``query``.

        :param documents: Retrieved documents (the paper's :math:`\\mathcal{R}`).
        :param query: The user query these documents were retrieved for.
        :returns: ``documents`` — survivors in retrieval order;
            ``removed_documents`` — the passages RAGDefender flagged as adversarial.
        """
        if self._defender is None:
            self.warm_up()
        assert self._defender is not None  # for type checkers

        docs = list(documents)
        kept, removed = partition(
            self._defender,
            query,
            [doc.content or "" for doc in docs],
            task_type=self.task_type,
            min_passages=self.min_passages,
        )
        return {
            "documents": [docs[i] for i in kept],
            "removed_documents": [docs[i] for i in removed],
        }

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary (``defender=`` instances are not serialisable)."""
        return default_to_dict(
            self,
            task_type=self.task_type,
            embedder=self.embedder,
            device=self.device,
            gpu_id=self.gpu_id,
            p=self.p,
            m=self.m,
            min_passages=self.min_passages,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGDefenderFilter":
        """Deserialise from a dictionary produced by :meth:`to_dict`."""
        return default_from_dict(cls, data)
