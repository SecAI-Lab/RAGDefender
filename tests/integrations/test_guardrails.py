"""Tests for :class:`ragdefender.integrations.guardrails.RAGDefenderValidator`."""

from __future__ import annotations

import pytest

pytest.importorskip("guardrails")

from guardrails.validator_base import FailResult, PassResult  # noqa: E402

from ragdefender.integrations.guardrails import RAGDefenderValidator  # noqa: E402


def _expected_safe(fixture):
    bad = set(fixture["adversarial_indices"])
    return [t for i, t in enumerate(fixture["R"]) if i not in bad]


def test_registered_under_hub_style_name():
    assert RAGDefenderValidator.rail_alias == "secai-lab/ragdefender"


def test_list_value_fails_with_filtered_fix(shared_defender, single_hop_R):
    validator = RAGDefenderValidator(defender=shared_defender, on_fail="fix")
    result = validator.validate(list(single_hop_R["R"]), {"query": single_hop_R["query"]})
    assert isinstance(result, FailResult)
    assert result.fix_value == _expected_safe(single_hop_R)
    assert str(len(single_hop_R["adversarial_indices"])) in result.error_message


def test_string_value_splits_on_separator_and_rejoins(shared_defender, single_hop_R):
    validator = RAGDefenderValidator(defender=shared_defender, on_fail="fix", separator="\n---\n")
    context = "\n---\n".join(single_hop_R["R"])
    result = validator.validate(context, {"query": single_hop_R["query"]})
    assert isinstance(result, FailResult)
    assert result.fix_value == "\n---\n".join(_expected_safe(single_hop_R))


def test_task_type_override_per_call(shared_defender, multi_hop_R):
    validator = RAGDefenderValidator(defender=shared_defender, task_type="multi_hop", on_fail="fix")
    result = validator.validate(list(multi_hop_R["R"]), {"query": multi_hop_R["query"]})
    assert isinstance(result, FailResult)
    assert result.fix_value == _expected_safe(multi_hop_R)


def test_task_type_required_without_defender():
    with pytest.raises(ValueError, match="task_type"):
        RAGDefenderValidator()


def test_builds_defender_from_config(single_hop_R):
    validator = RAGDefenderValidator(task_type="single_hop", embedder="minilm-all", device="cpu")
    result = validator.validate(list(single_hop_R["R"]), {"query": single_hop_R["query"]})
    assert isinstance(result, FailResult) and result.fix_value == _expected_safe(single_hop_R)


def test_small_sets_pass(shared_defender):
    validator = RAGDefenderValidator(defender=shared_defender)
    assert isinstance(validator.validate(["only", "two"], {"query": "q"}), PassResult)
    assert isinstance(validator.validate("", {}), PassResult)


def test_unsupported_type_fails_without_fix(shared_defender):
    result = RAGDefenderValidator(defender=shared_defender).validate(42, {})
    assert isinstance(result, FailResult) and result.fix_value is None


def test_guard_on_fail_fix_returns_filtered_context(shared_defender, single_hop_R):
    from guardrails import Guard

    guard = Guard().use(RAGDefenderValidator(defender=shared_defender, on_fail="fix"))
    outcome = guard.validate(
        "\n\n".join(single_hop_R["R"]), metadata={"query": single_hop_R["query"]}
    )
    assert outcome.validation_passed
    assert outcome.validated_output == "\n\n".join(_expected_safe(single_hop_R))


def test_guard_on_fail_exception_blocks(shared_defender, single_hop_R):
    from guardrails import Guard

    guard = Guard().use(RAGDefenderValidator(defender=shared_defender, on_fail="exception"))
    with pytest.raises(Exception, match="RAGDefender flagged"):
        guard.validate("\n\n".join(single_hop_R["R"]), metadata={"query": single_hop_R["query"]})
