"""Command-line interface for RAGDefender.

Subcommands:
  ragdefender info                           — version + paper + citation
  ragdefender defend ...                     — defend a single query
  ragdefender evaluate ...                   — batch evaluation on a labelled JSON file
  ragdefender reproduce {claim1,claim2,claim3}  — re-run an ACSAC reproducibility claim

Global flags:
  --verbose / -v   (count) → setup_logging(WARNING - 10*verbose)
  --version
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

from ragdefender import Evaluator, RAGDefender, __version__, setup_logging
from ragdefender._logging import get_logger

logger = get_logger(__name__)

# Deprecated aliases preserved for one minor.
_TASK_ALIASES = {"singlehop": "single_hop", "multihop": "multi_hop"}
_ATTACK_ALIASES = {"blind": "tan-et-al"}
_TASK_CHOICES = ["single_hop", "multi_hop"]
_TASK_CLI_CHOICES = _TASK_CHOICES + list(_TASK_ALIASES)
_ATTACK_CHOICES = ["poisonedrag", "tan-et-al", "garag"]
_ATTACK_CLI_CHOICES = _ATTACK_CHOICES + list(_ATTACK_ALIASES)


def _normalize_task(value: str) -> str:
    """Map deprecated ``singlehop``/``multihop`` to ``single_hop``/``multi_hop``."""
    if value in _TASK_ALIASES:
        warnings.warn(
            f"--task-type {value!r} is the v0.1.1 spelling; use {_TASK_ALIASES[value]!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _TASK_ALIASES[value]
    return value


def _normalize_attack(value: str) -> str:
    """Map deprecated ``blind`` to ``tan-et-al``."""
    if value in _ATTACK_ALIASES:
        warnings.warn(
            f"--attack {value!r} is the v0.1.1 spelling; use {_ATTACK_ALIASES[value]!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ATTACK_ALIASES[value]
    return value


# --------------------------------------------------------------- subcommands
def defend_command(args: argparse.Namespace) -> int:
    """Defend a single query against a corpus file (JSON or one-doc-per-line text)."""
    documents = _load_corpus(args.corpus)
    if documents is None:
        return 1

    defender = RAGDefender(device=args.device, gpu_id=args.gpu_id)
    task_type = _normalize_task(args.task_type)
    safe = defender.defend(query=args.query, R=documents, task_type=task_type)

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "query": args.query,
                    "original_docs": documents,
                    "defended_docs": safe,
                    "num_removed": len(documents) - len(safe),
                },
                indent=2,
            )
        )
        logger.info("Results saved to %s", args.output)
    else:
        # User-facing output stays on stdout, not via logger.
        print(f"\n=== Query ===\n{args.query}")
        print(f"\n=== Clean Documents ({len(safe)}/{len(documents)}) ===")
        for i, doc in enumerate(safe, 1):
            display = doc if len(doc) <= 200 else doc[:200] + "..."
            print(f"\n{i}. {display}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Aggregate Precision/Recall/F1 over a labelled JSON file."""
    test_data = json.loads(Path(args.test_data).read_text())
    defender = RAGDefender(device=args.device, gpu_id=args.gpu_id)
    evaluator = Evaluator(defender)
    task_type = _normalize_task(args.task_type)
    attack = _normalize_attack(args.attack)

    logger.info("Evaluating %d examples (attack=%s, task_type=%s)…", len(test_data), attack, task_type)
    results = evaluator.evaluate(
        test_data=test_data,
        attack_method=attack,
        task_type=task_type,
        verbose=True,
    )

    output_file = args.output or f"eval_results_{attack}_{task_type}.json"
    evaluator.save_results(results, output_file)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Attack Method:  {results['attack_method']}")
    print(f"Task Type:      {results['task_type']}")
    print(f"Examples:       {results['num_examples']}")
    print(f"Precision:      {results['precision']:.4f}")
    print(f"Recall:         {results['recall']:.4f}")
    print(f"F1 Score:       {results['f1_score']:.4f}")
    print(f"\nResults saved to {output_file}")
    return 0


def info_command(_args: argparse.Namespace) -> int:
    """Print version + paper + citation."""
    print(f"RAGDefender v{__version__}")
    print("Efficient defense against knowledge corruption attacks on RAG systems")
    print()
    print("Paper:    Rescuing the Unpoisoned: Efficient Defense against")
    print("          Knowledge Corruption Attacks on RAG Systems (ACSAC 2025)")
    print("DOI:      https://doi.org/10.1109/ACSAC67867.2025.00093")
    print("Authors:  Minseok Kim, Hankook Lee, Hyungjoon Koo (Sungkyunkwan University)")
    print("License:  MIT")
    print("Repo:     https://github.com/SecAI-Lab/RAGDefender")
    return 0


def reproduce_command(args: argparse.Namespace) -> int:
    """Run one of the three ACSAC artifact reproducibility claims.

    Delegates to ``claims/<claim_dir>/run.sh`` so the actual orchestration stays
    in one place. Reviewers wanting fine-grained control should run those
    scripts directly.
    """
    repo_root = _resolve_repo_root()
    if repo_root is None:
        print(
            "error: cannot locate the repository root (claims/ not found near the package). "
            "Run the claim's run.sh directly: bash claims/<claim_dir>/run.sh",
            file=sys.stderr,
        )
        return 2

    claim_map = {
        "claim1": "claim1_poisonedrag",
        "claim2": "claim2_tan",
        "claim3": "claim3_garag",
    }
    claim_dir = repo_root / "claims" / claim_map[args.claim]
    run_sh = claim_dir / "run.sh"
    if not run_sh.exists():
        # Fall back to the v0.1.1 directory names if Phase-4 rename hasn't landed yet.
        legacy_map = {"claim1": "claim1", "claim2": "claim2", "claim3": "claim3"}
        run_sh = repo_root / "claims" / legacy_map[args.claim] / "run.sh"
    if not run_sh.exists():
        print(f"error: run.sh not found at {run_sh}", file=sys.stderr)
        return 2

    logger.info("Executing %s", run_sh)
    return subprocess.call(["bash", str(run_sh)])


# --------------------------------------------------------------- helpers
def _load_corpus(path: str) -> Optional[list]:
    """Read a JSON list / dict-with-'documents' / one-doc-per-line text file."""
    if path.endswith(".json"):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: failed to read JSON corpus {path}: {exc}", file=sys.stderr)
            return None
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "documents" in data:
            return data["documents"]
        print("error: JSON corpus must be a list or a dict with 'documents' key", file=sys.stderr)
        return None
    try:
        return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    except OSError as exc:
        print(f"error: failed to read corpus {path}: {exc}", file=sys.stderr)
        return None


def _resolve_repo_root() -> Optional[Path]:
    """Walk up from the package looking for claims/ — the repo-root marker."""
    here = Path(__file__).resolve()
    for parent in [here.parent.parent, *here.parents]:
        if (parent / "claims").is_dir():
            return parent
    return None


def _configure_logging(verbose: int) -> None:
    """``-v`` → INFO, ``-vv`` → DEBUG; default WARNING (silent for normal runs)."""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    setup_logging(level=levels[min(verbose, len(levels) - 1)])


# --------------------------------------------------------------- argparse
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdefender",
        description="RAGDefender — defense against knowledge corruption attacks on RAG systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ragdefender info\n"
            "  ragdefender defend --query 'What is the capital?' --corpus docs.json --task-type single_hop\n"
            "  ragdefender evaluate --test-data test.json --attack poisonedrag --task-type multi_hop\n"
            "  ragdefender reproduce claim1\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"RAGDefender {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for INFO, -vv for DEBUG (default: WARNING-only)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- defend
    p_defend = subparsers.add_parser("defend", help="Defend a single query")
    p_defend.add_argument("--query", type=str, required=True, help="Query string")
    p_defend.add_argument(
        "--corpus", type=str, required=True, help="JSON list / dict / one-doc-per-line text file"
    )
    p_defend.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=_TASK_CLI_CHOICES,
        metavar="{single_hop,multi_hop}",
        help="single_hop (NQ, MS MARCO) or multi_hop (HotpotQA)",
    )
    p_defend.add_argument("--output", type=str, default=None, help="JSON output path (default: stdout)")
    p_defend.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_defend.add_argument("--gpu-id", type=int, default=0)

    # --- evaluate
    p_eval = subparsers.add_parser("evaluate", help="Batch-evaluate on labelled JSON")
    p_eval.add_argument("--test-data", type=str, required=True, help="Path to test data JSON")
    p_eval.add_argument(
        "--attack",
        type=str,
        default="poisonedrag",
        choices=_ATTACK_CLI_CHOICES,
        metavar="{poisonedrag,tan-et-al,garag}",
    )
    p_eval.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=_TASK_CLI_CHOICES,
        metavar="{single_hop,multi_hop}",
    )
    p_eval.add_argument("--output", type=str, default=None)
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_eval.add_argument("--gpu-id", type=int, default=0)

    # --- info
    subparsers.add_parser("info", help="Show package info")

    # --- reproduce
    p_repro = subparsers.add_parser(
        "reproduce", help="Run an ACSAC artifact reproducibility claim"
    )
    p_repro.add_argument("claim", choices=["claim1", "claim2", "claim3"])

    return parser


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "defend":
        return defend_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    if args.command == "info":
        return info_command(args)
    if args.command == "reproduce":
        return reproduce_command(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
