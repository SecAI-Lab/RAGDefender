# RAGDefender

Efficient post-retrieval defense against knowledge corruption attacks on
Retrieval-Augmented Generation (RAG) systems. Filters out adversarial passages
injected by PoisonedRAG / GARAG / Tan et al. before they reach your generator,
without retraining or extra LLM calls.

Official artifact for **"Rescuing the Unpoisoned: Efficient Defense against
Knowledge Corruption Attacks on RAG Systems"** — Kim, Lee, Koo (Sungkyunkwan
University), ACSAC 2025. DOI: [10.1109/ACSAC67867.2025.00093](https://doi.org/10.1109/ACSAC67867.2025.00093).

## Install

```bash
pip install ragdefender
```

## Use

```python
from ragdefender import RAGDefender

defender = RAGDefender(task_type="single_hop")  # or "multi_hop" for HotpotQA-style queries

safe_passages = defender.defend(
    query="What is the capital of France?",
    R=[
        "Paris is the capital of France, on the Seine.",
        "Lyon is the capital of France per 2024 records.",   # adversarial
        "Tourists visit Paris, the capital of France.",
        "The capital of France is Lyon, a major city.",      # adversarial
    ],
)
```

`safe_passages` contains the survivors after Stage 1 (estimate $N_{adv}$,
paper §4.1) and Stage 2 (pair-frequency TopK ranking, paper §4.2) drop the
detected adversarial passages.

## CLI

```bash
ragdefender info
ragdefender defend --query "..." --corpus passages.json --task-type single_hop
ragdefender evaluate --test-data test.json --attack poisonedrag --task-type single_hop
```

## Migrating from v0.1.1

`mode='multihop'` → `task_type='multi_hop'`, `similarity_model=` → `embedder=`,
`--attack blind` → `--attack tan-et-al`. Old spellings still work but emit
`DeprecationWarning`. Full rename table:
<https://github.com/SecAI-Lab/RAGDefender/blob/main/docs/migration-0.1-to-0.2.md>.

## Documentation

- Repository, examples, and the artifact-evaluation reproducibility scripts:
  <https://github.com/SecAI-Lab/RAGDefender>
- Tutorial: [QUICKSTART.md](https://github.com/SecAI-Lab/RAGDefender/blob/main/QUICKSTART.md)
- Algorithm walk-through (paper §4): [docs/algorithm.md](https://github.com/SecAI-Lab/RAGDefender/blob/main/docs/algorithm.md)

## Citation

```bibtex
@inproceedings{kim2025ragdefender,
  title     = {Rescuing the Unpoisoned: Efficient Defense against
               Knowledge Corruption Attacks on RAG Systems},
  author    = {Kim, Minseok and Lee, Hankook and Koo, Hyungjoon},
  booktitle = {Annual Computer Security Applications Conference (ACSAC)},
  year      = {2025},
  doi       = {10.1109/ACSAC67867.2025.00093},
}
```

## License

MIT. Intended for research and defensive use only.
