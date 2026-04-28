# DOMCS-EEG

Reproducibility package for **Domain-Orthogonal Multi-Component Supervised EEG (DOMCS-EEG)**, an EEG biometric verification framework evaluated under a strict Baseline-to-Task (B2T) protocol.

This repository is the code URL cited in the final v11 IEEE TIFS manuscript:

```text
DOMCS_EEG_TIFS_20260428_FINAL_v11_FULLY_CONSISTENT_MAIN.pdf
```

Cited archive DOI:

```text
https://doi.org/10.5281/zenodo.19666452
```

## What This Repository Contains

| Area | Contents |
|---|---|
| Main model | DOMCS-EEG 1D CNN embedding model with ArcFace, SupCon, state/domain, and orthogonality losses |
| Main experiment | 60-epoch 5-seed B2T training/evaluation pipeline |
| Paper tables | Paper-locked Table IV values and result-lock metadata |
| Raw evidence | Original NVIDIA A100 multi-seed result and independent Apr 28 rerun |
| Supporting experiments | Baseline, ablation, protocol, security, interpretability, and realtime-authentication assets where included in the repository |
| Reviewer checks | Verification scripts and audit files for paper/result alignment |

## Primary Paper Result

The v11 manuscript reports the following headline B2T result:

| Source | EER (%) | AUC | CRR (%) |
|---|---:|---:|---:|
| Manuscript Table IV | 3.75 +/- 0.20 | 0.9928 +/- 0.0008 | 86.95 +/- 0.40 |

Paper table source:

```text
experiments/results/TABLE_IV_main_results.csv
experiments/results/PAPER_RESULTS_LOCK.csv
```

Raw NVIDIA evidence is preserved separately:

| Evidence | Mean EER (%) | Std EER (%) | Mean AUC | Std AUC | File |
|---|---:|---:|---:|---:|---|
| Original Brev/NVIDIA A100 run, 2026-04-06 | 3.76 | 0.21 | 0.9931 | 0.0007 | `results/main_results/original_20260406_multi_seed_summary.csv` |
| Independent Brev/NVIDIA A100 rerun, 2026-04-28 | 3.82 | 0.29 | 0.9929 | 0.0011 | `results/main_results/rerun_20260428_multi_seed_summary.csv` |

These values are comparable but not bit-identical. GPU training can vary slightly because of CUDA kernels, floating-point ordering, DataLoader behavior, K-means initialization, and library versions. The ethical reproducibility claim is that the code reproduces comparable B2T performance within the reported experimental tolerance.

## Protocol

| Field | Value |
|---|---|
| Dataset | PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) |
| Subjects | 109 |
| Sampling rate | 128 Hz |
| EEG window | 2 seconds |
| Step | 1 second |
| Enrollment | R01-R02 resting-state baseline |
| Verification probes | R03-R14 task-state EEG |
| Gallery | K-means prototypes, K = 3 per subject |
| Metrics | EER, AUC, CRR, ROC/DET |
| Seeds | 1, 2, 3, 4, 5 |

## Model Configuration

The reviewer-facing configuration is stored at:

```text
experiments/configs/main_60ep.yaml
```

Main settings:

```text
epochs = 60
batch_size = 256
learning_rate = 3e-4
weight_decay = 1e-4
ArcFace scale = 30.0
ArcFace margin = 0.5
SupCon temperature = 0.07
loss weights = ArcFace 1.0, SupCon 0.5, State 0.5, Orthogonal 0.1
```

## Repository Structure

```text
domcs_eeg/                 Core model and biometric evaluation utilities
scripts/                   Training, framework, and verification scripts
experiments/configs/       Paper/reviewer hyperparameter configuration
experiments/results/       Paper-aligned result tables
results/main_results/      Raw original and rerun NVIDIA evidence
results/                   Additional baseline, ablation, security, and protocol results when present
figures/                   Paper/result figures
interpretability/          Grad-CAM, Integrated Gradients, occlusion, and UMAP figures
realtime_auth/             FastAPI-style real-time authentication prototype
data/                      Dataset placement instructions
checkpoints/               Checkpoint manifest and demo/evaluation checkpoint
```

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The dataset is not stored in GitHub. See:

```text
data/README.md
```

Expected preprocessed archive:

```text
EEGMMIDB_win2s_step1s_fs128.npz
```

Expected shape:

```text
X: (173198, 64, 256) float32
```

## Verify Repository Evidence

Run:

```bash
python scripts/verify_reproducibility.py
```

This recomputes the summary statistics for:

- manuscript Table IV,
- original NVIDIA A100 run,
- independent NVIDIA A100 rerun.

The legacy paper-table lock can also be checked with:

```bash
python scripts/verify_paper_alignment.py
```

## Full 60-Epoch Rerun

After placing the preprocessed dataset at the expected path, run:

```bash
python scripts/train_60ep.py
```

The script writes a timestamped experiment directory containing:

```text
multi_seed_summary.csv
seed_*/summary.json
seed_*/train_log.csv
seed_*/model_best.pt
```

A rerun should be interpreted statistically, not as an exact bitwise equality test.

## Realtime Authentication Direction

The `realtime_auth/` module provides a path for a live EEG biometric demo:

- enroll resting-state EEG windows,
- compute normalized identity embeddings,
- store K = 3 subject prototypes,
- verify task-state probes by maximum cosine similarity,
- expose accept/reject decisions through a web API.

## Citation

```bibtex
@article{kanimozhi2026domcs,
  title={DOMCS-EEG: Domain-Orthogonal Multi-Component Supervised EEG for Robust Cross-Task Biometric Verification},
  author={Kanimozhi, L. and Shridevi, S.},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026}
}
```
