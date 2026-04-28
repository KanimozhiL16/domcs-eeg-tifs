# DOMCS-EEG v11 Reproducibility Audit

This audit documents the relationship between the final v11 manuscript, the GitHub repository, and the NVIDIA A100 implementation evidence.

## Scope

Final manuscript package:

```text
DOMCS_EEG_TIFS_20260428_FINAL_v11_FULLY_CONSISTENT_MAIN.pdf
```

Public code URL cited in the manuscript:

```text
https://github.com/KanimozhiL16/domcs-eeg-tifs
```

Zenodo DOI cited in the manuscript:

```text
https://doi.org/10.5281/zenodo.19666452
```

## Primary Protocol

| Field | Value |
|---|---|
| Dataset | PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) |
| Subjects | 109 |
| Sampling rate | 128 Hz |
| Window | 2 s |
| Step | 1 s |
| Input tensor | 64 channels x 256 samples |
| Enrollment runs | R01-R02 |
| Verification runs | R03-R14 |
| Protocol | Baseline-to-Task (B2T) |
| Gallery | K-means prototypes, K = 3 per subject |
| Metrics | EER, AUC, CRR |
| Seeds | 1, 2, 3, 4, 5 |
| Epochs | 60 |

## Model and Training Configuration

| Item | Value |
|---|---|
| Backbone | 1D CNN: 64 -> 128 -> 256 filters with adaptive temporal pooling |
| Identity embedding | 128-D L2-normalized embedding |
| State/domain branch | Domain/state head from identity embedding |
| Complementary embedding | 128-D L2-normalized complementary-state embedding |
| ArcFace scale | 30.0 |
| ArcFace margin | 0.5 rad |
| SupCon temperature | 0.07 |
| Loss weights | ArcFace 1.0, SupCon 0.5, State 0.5, Orthogonal 0.1 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 256 |

Source files:

```text
scripts/train_60ep.py
experiments/configs/main_60ep.yaml
```

## Evidence Files

| Evidence | Repository File | Original Local/Brev Source |
|---|---|---|
| Manuscript Table IV | `experiments/results/TABLE_IV_main_results.csv` | v11 paper table |
| Original NVIDIA raw 60-epoch B2T run | `results/main_results/original_20260406_multi_seed_summary.csv` | `/home/nvidia/24PHD1237/FILES_1/EEGMMIDB/experiments/run_20260406_090214_60EP_FINAL/multi_seed_summary.csv` |
| Independent NVIDIA rerun | `results/main_results/rerun_20260428_multi_seed_summary.csv` | `/home/nvidia/24PHD1237/FILES_1/DOMCS_EEG_RERUN_20260428/results/run_20260428_015200_60EP_FINAL/multi_seed_summary.csv` |

Original final GitHub evidence archive:

```text
DOMCS_EEG_GITHUB_FINAL_20260406_2305.zip
created: 2026-04-06 23:05:52 UTC
```

Original raw multi-seed evidence timestamp:

```text
2026-04-06 09:08:26 UTC
```

Independent rerun evidence timestamp:

```text
2026-04-28 01:52:00 UTC
```

## Numeric Comparison

| Source | Seed Values | Mean EER (%) | Std EER (%) | Mean AUC | Std AUC | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Manuscript Table IV | Paper-rounded values | 3.75 | 0.20 | 0.9928 | 0.0008 | Locked manuscript table |
| Original NVIDIA A100 run | Seeds 1-5 | 3.76 | 0.21 | 0.9931 | 0.0007 | Raw implementation evidence |
| Independent NVIDIA A100 rerun | Seeds 1-5 | 3.82 | 0.29 | 0.9929 | 0.0011 | Reproducibility check |

The raw NVIDIA evidence and the independent rerun are close to the manuscript values, but they are not expected to be bit-identical. CUDA training, DataLoader ordering, K-means initialization, floating-point kernels, and library versions can cause small run-to-run differences even with fixed seeds.

## Ethical Reporting Position

The repository should not claim that every GPU rerun will exactly reproduce `3.75`, `0.9928`, and `86.95` bit-for-bit. The correct claim is:

```text
The repository preserves the final manuscript table and the raw NVIDIA A100 evidence. Independent reruns using the same code, preprocessed EEGMMIDB archive, seeds, protocol, and hyperparameters reproduce comparable B2T performance within the reported experimental tolerance.
```

## Reviewer Commands

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the preprocessed archive at one of the documented locations, for example:

```text
/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz
```

Run the verification script:

```bash
python scripts/verify_reproducibility.py
```

Run the full 60-epoch experiment:

```bash
python scripts/train_60ep.py
```

The full experiment creates a new folder under the configured EEGMMIDB experiment directory and writes `multi_seed_summary.csv` plus per-seed logs and summaries.
