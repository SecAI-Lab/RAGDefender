"""Evaluator — batch precision/recall/F1 and ASR measurement for a RAGDefender."""
from __future__ import annotations

import json
import warnings
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from ragdefender._logging import get_logger
from ragdefender.defender import RAGDefender

logger = get_logger(__name__)

# Map deprecated v0.1.1 mode strings to v0.2.0 task_type values.
_DEPRECATED_MODE_MAP = {"singlehop": "single_hop", "multihop": "multi_hop"}


class Evaluator:
    """Batch-evaluate a :class:`RAGDefender` on labelled test data.

    Behaviour change from v0.1.1: a single ``RAGDefender`` is created at
    construction time (or passed in) and reused across all examples. v0.1.1
    re-instantiated the defender on every call.
    """

    def __init__(self, defender: Optional[RAGDefender] = None):
        """
        Args:
            defender: Pre-built RAGDefender. If ``None``, a default instance is
                created lazily on first use.
        """
        self.defender = defender

    def _ensure_defender(self) -> RAGDefender:
        """Create the default defender once if the caller did not supply one."""
        if self.defender is None:
            logger.info("No defender provided; instantiating RAGDefender() with defaults.")
            self.defender = RAGDefender()
        return self.defender

    @staticmethod
    def _resolve_task_type(task_type: Optional[str], defense_mode: Optional[str]) -> str:
        """Resolve task_type, accepting the v0.1.1 ``defense_mode=`` alias."""
        if defense_mode is not None:
            warnings.warn(
                "defense_mode= is deprecated; pass task_type='single_hop'/'multi_hop'.",
                DeprecationWarning,
                stacklevel=3,
            )
            task_type = _DEPRECATED_MODE_MAP.get(defense_mode, defense_mode)
        if task_type is None:
            raise ValueError(
                "task_type must be specified (use 'single_hop' or 'multi_hop')."
            )
        return task_type

    # ----------------------------------------------------------- evaluate
    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        attack_method: str = "poisonedrag",
        task_type: Optional[str] = None,
        verbose: bool = True,
        # --- deprecated v0.1.1 kwarg ---
        defense_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate Precision / Recall / F1 of the defender over ``test_data``.

        Each example needs ``query``, ``retrieved_docs`` (or ``R``), and
        ``poisoned_indices``. ``ground_truth`` is optional and unused here.
        """
        defender = self._ensure_defender()
        effective_task = self._resolve_task_type(task_type, defense_mode)

        total_tp = total_fp = total_fn = 0
        iterator = tqdm(test_data) if verbose else test_data

        for example in iterator:
            query = example["query"]
            R = example.get("R") or example["retrieved_docs"]
            poisoned_indices = example.get("poisoned_indices", [])

            defended = defender.defend(query=query, R=R, task_type=effective_task)
            metrics = defender.get_metrics(R, defended, poisoned_indices)

            total_tp += metrics["true_positives"]
            total_fp += metrics["false_positives"]
            total_fn += metrics["false_negatives"]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "attack_method": attack_method,
            "task_type": effective_task,
            "num_examples": len(test_data),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }

    # ----------------------------------------------------------- evaluate_asr
    def evaluate_asr(
        self,
        test_data: List[Dict[str, Any]],
        llm_response_fn: Callable[[str, List[str]], str],
        task_type: Optional[str] = None,
        verbose: bool = True,
        # --- deprecated v0.1.1 kwarg ---
        defense_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attack Success Rate (ASR) before vs after defense.

        ``llm_response_fn(query, docs) -> str``. ASR counts a success when
        the example's ``target_answer`` (case-insensitive) appears in the
        LLM response.
        """
        defender = self._ensure_defender()
        effective_task = self._resolve_task_type(task_type, defense_mode)

        successful_before = successful_after = 0
        total = len(test_data)
        iterator = tqdm(test_data) if verbose else test_data

        for example in iterator:
            query = example["query"]
            R = example.get("R") or example["retrieved_docs"]
            target = example.get("target_answer", "")

            response_before = llm_response_fn(query, R)
            if target.lower() in response_before.lower():
                successful_before += 1

            defended = defender.defend(query=query, R=R, task_type=effective_task)
            response_after = llm_response_fn(query, defended)
            if target.lower() in response_after.lower():
                successful_after += 1

        asr_before = successful_before / total if total > 0 else 0.0
        asr_after = successful_after / total if total > 0 else 0.0
        asr_reduction = (
            (asr_before - asr_after) / asr_before * 100 if asr_before > 0 else 0.0
        )
        return {
            "asr_before_defense": asr_before,
            "asr_after_defense": asr_after,
            "asr_reduction_percent": asr_reduction,
            "attacks_successful_before": successful_before,
            "attacks_successful_after": successful_after,
            "total_examples": total,
        }

    # ----------------------------------------------------------- I/O
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str) -> None:
        """Write evaluation results to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def load_results(input_path: str) -> Dict[str, Any]:
        """Read evaluation results from a JSON file."""
        with open(input_path, "r") as f:
            return json.load(f)
