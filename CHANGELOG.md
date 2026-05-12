# Changes to RAGDefender Package

## ✅ Updated: Single Defense with Two Modes

### What Changed

**Before (Incorrect):**
- Had 3 separate methods: `isolation`, `aggregation`, `filtering`
- Did not match the paper's implementation

**After (Correct):**
- **One defense method** with **two modes**:
  - `mode='singlehop'` - For NQ, MSMARCO (uses `find_num_adv_agg`)
  - `mode='multihop'` - For HotpotQA (uses `find_num_adv`)

### API Changes

```python
# OLD (Wrong)
defender.defend(query, docs, method='isolation')

# NEW (Correct)
defender.defend(query, docs, mode='singlehop')   # For NQ, MSMARCO
defender.defend(query, docs, mode='multihop')    # For HotpotQA
```

### CLI Changes

```bash
# OLD
ragdefender defend --method isolation

# NEW
ragdefender defend --mode singlehop
ragdefender defend --mode multihop
```

### Implementation Details

**Single-Hop Mode** (`mode='singlehop'`):
- Uses `_find_num_adversarial_agg()`
- Aggregation clustering + TF-IDF validation
- Best for simple factual questions

**Multi-Hop Mode** (`mode='multihop'`):
- Uses `_find_num_adversarial()`
- Similarity-based outlier detection
- Best for complex multi-document reasoning

### Files Updated

1. **ragdefender/core/defender.py**
   - Removed: `_isolation_defense()`, `_aggregation_defense()`, `_filtering_defense()`
   - Updated: `defend()` method to use `mode` parameter
   - Kept: `_find_num_adversarial()` and `_find_num_adversarial_agg()`

2. **ragdefender/core/evaluator.py**
   - Changed parameter from `defense_method` to `defense_mode`
   - Updated all method signatures

3. **ragdefender/cli.py**
   - Changed `--method` to `--mode`
   - Choices: `['singlehop', 'multihop']`

4. **Documentation**
   - Updated: README.md, QUICKSTART.md
   - Updated: examples/basic_usage.py

### Why This Matters

This now correctly implements the paper's approach where:
- NQ and MSMARCO use aggregation-based detection (single-hop)
- HotpotQA uses similarity-based detection (multi-hop)

The key insight is that single-hop and multi-hop questions have different document similarity patterns, requiring different detection strategies.
