"""Guardrails AI adapter — :class:`RAGDefenderValidator` is a Guardrails ``Validator``.

Guardrails validates values flowing through an LLM app and lets you decide
what happens on failure (``on_fail="fix" | "exception" | "noop" | ...``). This
validator treats the *retrieved context* as the value: it passes when no
passage looks adversarial and otherwise fails with a ``fix_value`` holding the
filtered context, so ``on_fail="fix"`` turns it into a transparent filter and
``on_fail="exception"`` into a hard gate.

Install::

    pip install 'ragdefender[guardrails]'

Use::

    from guardrails import Guard
    from ragdefender.integrations import RAGDefenderValidator

    guard = Guard().use(RAGDefenderValidator(task_type="single_hop", on_fail="fix"))

    # Context is the retrieved passages joined by `separator` (default: blank line)
    outcome = guard.validate("\\n\\n".join(passages), metadata={"query": question})
    safe_context = outcome.validated_output

You can also call the validator directly on a list of passages::

    result = RAGDefenderValidator(task_type="single_hop").validate(passages, {"query": question})
    safe_passages = result.fix_value if result.outcome == "fail" else passages

Guardrails Hub: this class follows the Hub validator contract
(``@register_validator`` + ``validate()``), so the same code can be published
as a standalone Hub package. See ``docs/integrations.md``.
"""

from typing import Any, Callable, Dict, List, Optional, Union

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
    from guardrails.validator_base import (
        FailResult,
        PassResult,
        ValidationResult,
        Validator,
        register_validator,
    )
except ImportError as exc:  # pragma: no cover — exercised only without the extra
    raise missing_dependency("guardrails-ai", "guardrails") from exc

logger = get_logger(__name__)

__all__ = ["RAGDefenderValidator"]

#: Default separator between passages when the context arrives as one string.
DEFAULT_SEPARATOR = "\n\n"


@register_validator(name="secai-lab/ragdefender", data_type=["string", "list"])
class RAGDefenderValidator(Validator):
    """Validates that retrieved RAG context contains no adversarial (poisoned) passages.

    **Key Properties**

    | Property                      | Description                                   |
    | ----------------------------- | --------------------------------------------- |
    | Name for `format` attribute   | `secai-lab/ragdefender`                        |
    | Supported data types          | `string` (passages joined by `separator`), `list` (of passages) |
    | Programmatic fix              | The context with the flagged passages removed  |

    The query the passages were retrieved for is read from
    ``metadata[query_key]`` (default key: ``"query"``). It is optional today —
    the current detectors are query-agnostic — but pass it so future
    query-conditioned strategies keep working.

    Args:
        task_type: ``"single_hop"`` (NQ / MS MARCO-style fact lookup) or
            ``"multi_hop"`` (HotpotQA-style). Required unless ``defender`` is given.
        embedder: Preset (``"minilm-all"``, ``"minilm-paraphrase"``, ``"stella"``)
            or Hugging Face id of the sentence-embedding model.
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        gpu_id: GPU index when ``device="cuda"``.
        p: Stage-2 frequency-score exponent (paper Eq. 6).
        m: Stage-1 top-``m`` TF-IDF terms (paper Eq. 1).
        min_passages: Retrieved sets smaller than this pass validation untouched.
        separator: Delimiter used to split a string value into passages and to
            re-join the fix value.
        query_key: Metadata key holding the user query.
        on_fail: Guardrails on-fail action (``"fix"`` recommended for filtering).
        defender: Pre-built :class:`ragdefender.RAGDefender`; when given, the
            constructor knobs above are ignored except ``task_type`` (which, if
            set, overrides the defender's own per call), ``min_passages``,
            ``separator`` and ``query_key``.
    """  # noqa: E501

    def __init__(
        self,
        task_type: Optional[str] = None,
        embedder: str = DEFAULT_EMBEDDER,
        device: str = "auto",
        gpu_id: int = 0,
        p: int = 2,
        m: int = 5,
        min_passages: int = DEFAULT_MIN_PASSAGES,
        separator: str = DEFAULT_SEPARATOR,
        query_key: str = "query",
        on_fail: Optional[Union[Callable, str]] = None,
        defender: Optional[RAGDefender] = None,
    ) -> None:
        super().__init__(
            on_fail=on_fail,
            task_type=task_type,
            embedder=embedder,
            device=device,
            gpu_id=gpu_id,
            p=p,
            m=m,
            min_passages=min_passages,
            separator=separator,
            query_key=query_key,
        )
        self._task_type = task_type
        self._min_passages = min_passages
        self._separator = separator
        self._query_key = query_key
        self._defender = build_defender(
            defender,
            task_type=task_type,
            embedder=embedder,
            device=device,
            gpu_id=gpu_id,
            p=p,
            m=m,
        )

    @property
    def defender(self) -> RAGDefender:
        """The underlying :class:`ragdefender.RAGDefender` instance."""
        return self._defender

    # ------------------------------------------------------------ Guardrails API
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Pass when no passage is flagged; otherwise fail with the filtered context as fix."""
        metadata = metadata or {}
        query = str(metadata.get(self._query_key, ""))

        if isinstance(value, str):
            passages: List[str] = [part for part in value.split(self._separator) if part.strip()]
            as_string = True
        elif isinstance(value, (list, tuple)):
            passages = [str(part) for part in value]
            as_string = False
        else:
            return FailResult(
                error_message=(
                    "RAGDefenderValidator expects the retrieved context as a string "
                    f"(passages joined by {self._separator!r}) or a list of strings, "
                    f"got {type(value).__name__}."
                )
            )

        kept, removed = partition(
            self._defender,
            query,
            passages,
            task_type=self._task_type,
            min_passages=self._min_passages,
        )
        if not removed:
            return PassResult()

        safe: List[str] = [passages[i] for i in kept]
        fix_value: Any = self._separator.join(safe) if as_string else safe
        return FailResult(
            error_message=(
                f"RAGDefender flagged {len(removed)} of {len(passages)} retrieved "
                f"passages as adversarial (indices {removed})."
            ),
            fix_value=fix_value,
        )
