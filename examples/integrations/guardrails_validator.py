"""RAGDefender as a Guardrails AI validator on the retrieved context.

    pip install "ragdefender[guardrails]"

``on_fail="fix"`` turns the validator into a transparent filter;
``on_fail="exception"`` makes it a hard gate.
"""
from guardrails import Guard
from guardrails.errors import ValidationError

from ragdefender.integrations import RAGDefenderValidator

from _passages import PASSAGES, QUERY, report


def main() -> None:
    context = "\n\n".join(PASSAGES)

    # 1. Filter mode: validated_output is the context minus the flagged passages
    guard = Guard().use(RAGDefenderValidator(task_type="single_hop", device="cpu", on_fail="fix"))
    outcome = guard.validate(context, metadata={"query": QUERY})
    kept = [PASSAGES.index(p) for p in outcome.validated_output.split("\n\n")]
    report('Guard(on_fail="fix")', kept)

    # 2. Gate mode: raise when anything is flagged
    gate = Guard().use(RAGDefenderValidator(task_type="single_hop", device="cpu", on_fail="exception"))
    try:
        gate.validate(context, metadata={"query": QUERY})
        print('Guard(on_fail="exception"): passed')
    except ValidationError as exc:
        print(f'Guard(on_fail="exception"): blocked → {exc}')


if __name__ == "__main__":
    main()
