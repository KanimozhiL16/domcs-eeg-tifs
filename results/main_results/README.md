# Main Result Evidence

This folder separates the submitted manuscript table from the raw NVIDIA evidence.

## Files

| File | Meaning |
|---|---|
| `original_20260406_multi_seed_summary.csv` | Original Brev/NVIDIA A100 60-epoch B2T run created on 2026-04-06 09:08:26 UTC. |
| `rerun_20260428_multi_seed_summary.csv` | Independent rerun on Brev/NVIDIA A100 using the same uploaded code/data/protocol on 2026-04-28. |

The submitted manuscript table is stored separately at:

```text
experiments/results/TABLE_III_main_results.csv
```

A backward-compatible alias with the same values is retained at:

```text
experiments/results/TABLE_IV_main_results.csv
```

The raw rerun is expected to be close to the submitted manuscript values, not bit-identical, because CUDA training contains nondeterministic operations unless a stricter deterministic pipeline is enforced.

## Recomputed Summary

| Evidence | Mean EER (%) | Std EER (%) | Mean AUC | Std AUC |
|---|---:|---:|---:|---:|
| Submitted paper Table III | 3.76 | 0.21 | 0.9931 | 0.0007 |
| Original NVIDIA A100 run | 3.76 | 0.21 | 0.9931 | 0.0007 |
| Independent NVIDIA A100 rerun | 3.82 | 0.29 | 0.9929 | 0.0011 |

Use `python scripts/verify_reproducibility.py` to recompute and compare these values.
