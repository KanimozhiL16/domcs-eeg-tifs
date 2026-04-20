# DOMCS-EEG: Security-Aware Cross-State EEG Biometric Verification

> **Paper:** Security-Aware Cross-State EEG Biometric Verification via Protocol Realism and Representation Disentanglement  
> **Journal:** IEEE Transactions on Information Forensics and Security (T-IFS)  
> **Authors:** Kanimozhi L · S. Shridevi — Vellore Institute of Technology, Chennai  
> **Hardware:** NVIDIA A100-SXM4-80GB (Brev.dev, NVIDIA Academic Hardware Grant)

---

## Overview

DOMCS-EEG is a cross-state EEG biometric verification framework that addresses the fundamental challenge of **cognitive-state variability**: a user enrolled at rest (eyes closed/open) must be reliably verified during active motor tasks — a mismatch no prior method evaluates under a strict protocol.

**Key contributions:**
1. **B2T Protocol** — Baseline-to-Task: enroll on resting-state (R01+R02), verify across 12 motor-task runs (R03–R14). The only deployment-realistic cross-state evaluation on EEGMMIDB.
2. **Disentangled Identity Embedding** — A squared-cosine decorrelation constraint forces the identity head `z_id` and the cognitive-state head `z_cs` to occupy independent subspaces.
3. **Six-Category Security Analysis (T0–T5)** — Forgery, enrollment leakage, noise robustness, mimicry, and raw-input FGSM/PGD adversarial attacks.

---

## Results

| Metric | Value | Protocol |
|--------|-------|----------|
| **EER** | **3.75% ± 0.20%** | B2T (5 seeds) |
| **AUC** | **0.9928 ± 0.0008** | B2T (5 seeds) |
| **CRR** | **86.95% ± 0.40%** | B2T (5 seeds) |
| 95% CI (EER) | [3.35%, 4.88%] | Bootstrapped |
| Protocol inflation | +143% EER vs. random-split | B2T vs. random |
| State leakage reduction | 92.48% → 52.9% | After decorrelation |

**Per-seed results:**

| Seed | EER (%) | AUC |
|------|---------|-----|
| 1 | 3.69 | 0.9925 |
| 2 | 3.81 | 0.9931 |
| 3 | 4.01 | 0.9926 |
| 4 | 3.45 | 0.9943 |
| 5 | 3.87 | 0.9931 |
| **Mean ± Std** | **3.75 ± 0.20** | **0.9928 ± 0.0008** |

---

## Repository Structure

```
eeg-biometric-disentangle/
│
├── domcs_eeg/                  # Core Python package
│   ├── model.py                # DOMCS-EEG model (EEGBackbone, DOMCSEEG, ArcFaceLoss)
│   ├── evaluate.py             # compute_eer(), build_prototypes(), evaluate_b2t()
│   ├── core_framework.py       # Experiment framework utilities
│   └── experiment_groups.py    # Multi-group experiment runner
│
├── scripts/
│   └── train_60ep.py           # Full training script (as run on Brev A100)
│
├── configs/
│   └── main_60ep.yaml          # Hyperparameter config for reproduction
│
├── results/
│   ├── main_results/           # Per-seed EER/AUC/CRR (raw + paper tables)
│   ├── security_analysis/      # T0–T4 security CSVs + LaTeX tables
│   ├── adversarial_t5/         # FGSM, PGD, targeted false-accept, 50Hz noise
│   ├── baseline_comparison/    # Comparison vs. CNN+Softmax, Triplet, SupCon, ArcFace
│   └── ablation/               # E1–E5 ablation + epoch/embedding sweep
│
├── figures/                    # All 33 paper figures (PNG)
│
├── checkpoints/
│   └── CHECKPOINT_MANIFEST.md  # Checkpoint locations and loading instructions
│
├── requirements.txt
├── setup.py
└── reproduce.py                # One-command reproduction check
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and preprocess the dataset
Download **PhysioNet EEGMMIDB** from https://physionet.org/content/eegmmidb/1.0.0/

```python
# Preprocess to NPZ format:
# X.shape = (N, 64, 256)  — N windows, 64 channels, 256 time-steps (2s @ 128Hz)
# Y = subject IDs (0–108)
# session = run labels (r01–r14)
# Save as: data/eegmmidb_preprocessed.npz
```

### 3. Train
```bash
python scripts/train_60ep.py
```
Update `ROOT` and `NPZ_PATH` in `train_60ep.py` to point to your preprocessed data.

**Expected output (5 seeds):**
```
EER:  3.75% ± 0.20%
AUC:  0.9928 ± 0.0008
CRR:  86.95% ± 0.40%
```

### 4. Evaluate a saved checkpoint
```python
from domcs_eeg.model import DOMCSEEG
from domcs_eeg.evaluate import evaluate_b2t

model = DOMCSEEG.from_checkpoint("checkpoints/seed_1/model_best.pt")
results = evaluate_b2t(model, X_enroll, Y_enroll, X_test, Y_test, runs_test)
print(f"EER: {results['eer']:.4f}  AUC: {results['auc']:.4f}")
```

---

## Model Architecture

```
Input: (B, 64, 256)  — batch × channels × time

Backbone (shared):
  Conv1d(64→64,  k=7) → BatchNorm → ELU
  Conv1d(64→128, k=5) → BatchNorm → ELU
  Conv1d(128→256,k=3) → BatchNorm → ELU → AdaptiveAvgPool1d(1)
  → f ∈ ℝ²⁵⁶

Identity head:   Linear(256→128) + LayerNorm + L2-norm  → z_id  ∈ ℝ¹²⁸
State head:      Linear(256→128) + LayerNorm + L2-norm  → z_cs  ∈ ℝ¹²⁸
State classifier (training only): Linear(128→64) → ReLU → Linear(64→2)

At inference: only z_id and enrolled K-means prototypes are used.
```

## Loss Function

```
L = L_arcface(×1.0) + L_supcon(×0.5) + L_state(×0.5) + L_orth(×0.1)

L_orth = (1/B) Σᵢ (z_id⁽ⁱ⁾ · z_cs⁽ⁱ⁾)²   ← squared cosine similarity penalty
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| ArcFace (s=30, m=0.50) | 1.0 | Inter-subject angular margin discrimination |
| Supervised Contrastive (τ=0.07) | 0.5 | Intra-class compactness across states |
| State classification | 0.5 | Forces z_cs to encode cognitive state |
| Decorrelation (L_orth) | 0.1 | Suppresses identity–state entanglement |

---

## B2T Protocol

```
Enrollment:  R01 + R02  (resting state, eyes open/closed)  →  K-means (K=3) prototypes
Verification: R03 – R14 (12 motor task runs)               →  cosine similarity scoring
```

This is the **only** protocol that reflects real deployment: enrollment state ≠ verification state.

---

## Security Analysis Summary

| Threat | Setting | Result |
|--------|---------|--------|
| T0 Random forgery | Baseline | EER = 4.17% |
| T1 Skilled forgery | Resting-state EEG probe | EER = 3.41% (Δ = −0.76 pp) |
| T2 Enrollment leakage | 50% gallery exposed | EER = 3.96% |
| T3 AWGN noise | 30 dB SNR | EER = 4.17% (unchanged) |
| T3 AWGN noise | 10 dB SNR | EER = 8.00% |
| T4 Black-box mimicry | 50 query steps | EER = 5.06% |
| T5 FGSM | ε = 0.005 | EER = 5.90%, ASR = 12.72% |
| T5 PGD | ε = 0.010, 7 steps | EER = 8.97%, ASR = 26.50% |
| T5b Targeted false-accept | ε = 0.005, 30 pairs | ASR = 12.77% |
| 50 Hz line noise | amp = 0.05 | EER ≈ 3.35% (negligible) |

**Key finding:** Genuine scores remain stable under all T5 adversarial attacks (genuine_score_drop = 0.0). Attacks raise impostor FAR — they cannot degrade the real user's identity embedding.

---

## Checkpoints

Pre-trained model checkpoints (Seeds 1–5) are hosted separately due to file size.  
See [`checkpoints/CHECKPOINT_MANIFEST.md`](checkpoints/CHECKPOINT_MANIFEST.md) for download instructions and loading code.

> Checkpoints trained on: NVIDIA A100-SXM4-80GB (Brev.dev, NVIDIA Academic Hardware Grant)

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 3×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 256 |
| Epochs | 60 (early stop, patience=10) |
| Embedding dim | 128 |
| ArcFace scale s | 30.0 |
| ArcFace margin m | 0.50 |
| SupCon temperature τ | 0.07 |
| λ_sup | 0.5 |
| λ_state | 0.5 |
| λ_orth | 0.1 |
| K (prototypes) | 3 |
| Seeds | {1, 2, 3, 4, 5} |

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)**
- 109 subjects · 64 channels · 128 Hz (downsampled from 160 Hz)
- 14 runs per subject: R01–R02 (resting), R03–R14 (motor tasks)
- 173,198 total windows after 2s/50%-overlap segmentation

```bibtex
@article{schalk2004bci2000,
  title={{BCI2000}: A general-purpose brain-computer interface system},
  author={Schalk, G. and McFarland, D.J. and Hinterberger, T. and Birbaumer, N. and Wolpaw, J.R.},
  journal={IEEE Trans. Biomed. Eng.}, volume={51}, number={6}, pages={1034--1043}, year={2004}
}
@article{goldberger2000physionet,
  title={{PhysioBank}, {PhysioToolkit}, and {PhysioNet}},
  author={Goldberger, A.L. and others},
  journal={Circulation}, volume={101}, number={23}, pages={e215--e220}, year={2000}
}
```

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{kanimozhi2026domcseeg,
  title={Security-Aware Cross-State {EEG} Biometric Verification via Protocol Realism
         and Representation Disentanglement},
  author={Kanimozhi, L. and Shridevi, S.},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  note={Under review}
}
```

---

## License

Code: MIT License.  
Dataset: PhysioNet Restricted Health Data License — see https://physionet.org/content/eegmmidb/1.0.0/

---

*VIT Chennai · School of Computer Science and Engineering (SCOPE) · Centre for Neuroinformatics*
