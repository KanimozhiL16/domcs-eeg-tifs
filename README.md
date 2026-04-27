# DOMCS-EEG

Paper-aligned implementation for **Domain-Orthogonal Multi-Component Supervised EEG (DOMCS-EEG)**, an EEG biometric verification framework evaluated under a strict Baseline-to-Task (B2T) protocol.

This repository is aligned with the manuscript:

**DOMCS-EEG: Domain-Orthogonal Multi-Component Supervised EEG for Robust Cross-Task Biometric Verification**  
Target venue: IEEE Transactions on Information Forensics and Security (TIFS), 2026.

## Paper-Locked Headline Results

The following values are locked to the final manuscript package `DOMCS_EEG_TIFS_20260427_FINAL_v7_COMPLETE_MAIN.pdf`.

| Setting | EER (%) | AUC | CRR (%) | Source |
|---|---:|---:|---:|---|
| DOMCS-EEG, B2T, 5 seeds | 3.75 +/- 0.20 | 0.9928 +/- 0.0008 | 86.95 +/- 0.40 | `experiments/results/TABLE_IV_main_results.csv` |
| Random split | 1.54 | 0.9986 | - | manuscript Table V |
| Hybrid protocol | 2.71 | 0.9951 | - | manuscript Table V |
| ArcFace-only ablation (E1) | 3.94 +/- 0.31 | - | - | manuscript Table VI |
| Full DOMCS-EEG ablation (E5) | 3.86 +/- 0.13 | - | - | manuscript Table VI |

Primary protocol:

- Dataset: PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)
- Subjects: 109
- Sampling rate: 128 Hz
- EEG window: 2 seconds, 1 second step
- Enrollment: R01-R02 resting-state baseline
- Verification probes: R03-R14 task-state EEG
- Gallery: K-means prototypes, K = 3 per subject
- Metrics: Equal Error Rate (EER), AUC, Correct Recognition Rate (CRR), ROC/DET

## Repository Structure

```text
domcs_eeg/                 Core model and biometric evaluation utilities
scripts/                   Training and experiment framework scripts
experiments/configs/       Paper-locked hyperparameter configuration
experiments/results/       Paper-aligned result tables
figures/                   Paper/result figures
interpretability/          Grad-CAM, Integrated Gradients, occlusion and UMAP figures
realtime_auth/             FastAPI-style real-time authentication prototype
runtime_store/             Portable runtime registry template
checkpoints/               Seed-1 checkpoint for demo/evaluation
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility

The main locked configuration is:

```text
experiments/configs/main_60ep.yaml
```

The paper-aligned result table is:

```text
experiments/results/TABLE_IV_main_results.csv
```

For a full rerun, update the dataset path in `scripts/core_framework.py` or adapt it through your local experiment launcher, then run:

```bash
python scripts/train_60ep.py
```

The expected 5-seed B2T result is:

```text
EER = 3.75 +/- 0.20 %
AUC = 0.9928 +/- 0.0008
CRR = 86.95 +/- 0.40 %
```

## Dataset

Download EEGMMIDB from PhysioNet:

https://physionet.org/content/eegmmidb/1.0.0/

Expected preprocessed archive:

```text
EEGMMIDB_win2s_step1s_fs128.npz
```

The repository does not include the full dataset.

## Deployment Direction

The `realtime_auth/` module provides a practical path for a live EEG biometric demo:

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
