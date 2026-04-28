# DOMCS-EEG Result Report and Repository Audit

Audit status: updated for the final v11 reproducibility package.

Final manuscript:

```text
DOMCS_EEG_TIFS_20260428_FINAL_v11_FULLY_CONSISTENT_MAIN.pdf
```

Public repository cited in the paper:

```text
https://github.com/KanimozhiL16/domcs-eeg-tifs
```

Archived DOI cited in the paper:

```text
https://doi.org/10.5281/zenodo.19666452
```

## Executive Summary

This repository is intended to serve as the reviewer-facing reproducibility package for DOMCS-EEG. The core scientific claim is a strict Baseline-to-Task (B2T) EEG biometric protocol: train/enroll on R01-R02 resting baseline and verify on R03-R14 task EEG.

The repository now separates three things that should not be confused:

| Category | File | Purpose |
|---|---|---|
| Manuscript table | `experiments/results/TABLE_IV_main_results.csv` | Final v11 table values as written in the paper |
| Original raw run | `results/main_results/original_20260406_multi_seed_summary.csv` | Original NVIDIA A100 implementation result |
| Independent rerun | `results/main_results/rerun_20260428_multi_seed_summary.csv` | Apr 28 NVIDIA A100 reproducibility check |

## Main Result Comparison

| Source | Mean EER (%) | Std EER (%) | Mean AUC | Std AUC | Interpretation |
|---|---:|---:|---:|---:|---|
| Manuscript Table IV | 3.75 | 0.20 | 0.9928 | 0.0008 | Paper-locked table |
| Original NVIDIA A100 run | 3.76 | 0.21 | 0.9931 | 0.0007 | Raw implementation evidence |
| Independent NVIDIA A100 rerun | 3.82 | 0.29 | 0.9929 | 0.0011 | Reproducibility check |

The raw rerun is close to the manuscript result but not bit-identical. This is expected for GPU training unless all nondeterministic CUDA behavior is disabled. The ethical claim is statistical reproducibility within tolerance, not exact numeric equality for every rerun.

## Protocol and Configuration

| Field | Value |
|---|---|
| Dataset | EEGMMIDB |
| Subjects | 109 |
| Sampling rate | 128 Hz |
| Input | 2 s windows, 1 s step, 64 x 256 tensor |
| Train/enrollment runs | R01-R02 |
| Test/verification runs | R03-R14 |
| Seeds | 1, 2, 3, 4, 5 |
| Epochs | 60 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Loss weights | ArcFace 1.0, SupCon 0.5, State 0.5, Orthogonal 0.1 |
| Gallery | K-means prototypes, K = 3 |

Configuration source:

```text
experiments/configs/main_60ep.yaml
```

Training source:

```text
scripts/train_60ep.py
```

## Implementation Scope

The repository contains or documents the following paper implementation areas:

| Area | Status |
|---|---|
| 60-epoch main B2T model | Present |
| Table IV paper lock | Present |
| Original NVIDIA raw evidence | Present |
| Independent NVIDIA rerun evidence | Present |
| Dataset placement guide | Present in `data/README.md` |
| Reproducibility verifier | Present in `scripts/verify_reproducibility.py` |
| Paper table verifier | Present in `scripts/verify_paper_alignment.py` |
| Baseline/protocol/ablation/security artifacts | Present where tracked; should be referenced carefully by exact file |
| Realtime authentication prototype | Present |

## Reviewer Verification

Run:

```bash
python scripts/verify_reproducibility.py
```

Expected interpretation:

```text
OK: paper table, v11 config, original raw evidence, and rerun evidence are consistent within tolerance.
```

Run the table lock only:

```bash
python scripts/verify_paper_alignment.py
```

Run the full experiment after placing the preprocessed EEGMMIDB archive:

```bash
python scripts/train_60ep.py
```

## Known Caveats

1. The full EEGMMIDB preprocessed `.npz` archive is not stored in GitHub because it is large.
2. A GPU rerun can differ slightly from the paper table due to CUDA nondeterminism and floating-point ordering.
3. Single-seed or 200-epoch diagnostic runs should not replace the 5-seed 60-epoch Table IV result.
4. Older local folders may contain draft values and should not be used as the canonical final paper source.

## Final Verdict

The repo is now organized around a reviewer-safe position:

- the v11 manuscript table is preserved,
- the original NVIDIA A100 evidence is preserved,
- an independent NVIDIA A100 rerun is preserved,
- the actual training hyperparameters are documented,
- verification scripts distinguish exact paper-table checks from statistical rerun checks.

This is the ethical way to align the paper and implementation: preserve the paper numbers, preserve raw evidence, and avoid claiming impossible bitwise equality for nondeterministic GPU training.
