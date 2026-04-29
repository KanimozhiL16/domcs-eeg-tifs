# Submitted Paper Implementation Map

This map shows where each major implementation/evidence area of the submitted IEEE TIFS paper belongs in the repository. It is intended for reviewers who need to connect the manuscript claims to code and result files.

## Main B2T Model

| Paper Item | Repository Location |
|---|---|
| DOMCS-EEG model definition | `scripts/train_60ep.py`, `scripts/core_framework.py`, `domcs_eeg/model.py` |
| Main 60-epoch training pipeline | `scripts/train_60ep.py` |
| Main configuration | `experiments/configs/main_60ep.yaml` |
| Submitted main paper table | `experiments/results/TABLE_III_main_results.csv` |
| Backward-compatible table alias | `experiments/results/TABLE_IV_main_results.csv` |
| Paper result lock | `experiments/results/PAPER_RESULTS_LOCK.csv` |
| Original raw NVIDIA evidence | `results/main_results/original_20260406_multi_seed_summary.csv` |
| Independent NVIDIA rerun evidence | `results/main_results/rerun_20260428_multi_seed_summary.csv` |

## Dataset and Protocol

| Paper Item | Repository Location |
|---|---|
| Dataset placement guide | `data/README.md` |
| B2T protocol metadata | `experiments/configs/main_60ep.yaml` |
| Paper alignment verifier | `scripts/verify_paper_alignment.py` |
| Reproducibility verifier | `scripts/verify_reproducibility.py` |
| Full audit | `REPRODUCIBILITY_AUDIT.md` |

## Supporting Experiments

| Paper Section / Evidence Type | Repository Location |
|---|---|
| Baseline comparison | `results/baseline_comparison/` when present, plus baseline notebooks/scripts |
| Loss ablation | `results/ablation/`, `scripts/experiment_groups.py`, `scripts/core_framework.py` |
| Protocol comparison | `scripts/experiment_groups.py`, protocol result tables/figures where present |
| Epoch study, including 200-epoch diagnostic | `experiments/`, `results/ablation/`, and raw NVIDIA `run_200EP_SEED1` evidence where archived |
| Enrollment/prototype studies | `scripts/experiment_groups.py`, corresponding result CSVs where present |
| Interpretability | `interpretability/` and paper figures |
| Security/adversarial analysis | `results/security_analysis/`, `results/adversarial_t5/`, and attack notebooks where present |
| Realtime authentication prototype | `realtime_auth/`, `run_realtime_auth.py`, `start_realtime_auth.bat` |

## What Reviewers Should Run First

```bash
python scripts/verify_reproducibility.py
```

This verifies the submitted paper table, the corrected hyperparameter configuration, the original NVIDIA A100 evidence, and the independent NVIDIA rerun evidence.

## What Should Not Be Used as the Main Table III Source

The following are useful historical or diagnostic artifacts, but they should not replace the 5-seed 60-epoch B2T submitted Table III source:

| Artifact Type | Reason |
|---|---|
| 40-epoch notebooks | Earlier/shorter diagnostic runs |
| 200-epoch single-seed logs | Epoch-study diagnostic, not the 5-seed main claim |
| Older Q1 backup folders | Some contain different draft values |
| Notebook checkpoint copies | May contain stale generated text or draft tables |

## Ethical Reproducibility Statement

This repository supports statistical reproducibility. It preserves the submitted paper table, the original NVIDIA A100 run, and an independent NVIDIA A100 rerun. It does not claim that every GPU rerun will be bit-identical to the manuscript table.
