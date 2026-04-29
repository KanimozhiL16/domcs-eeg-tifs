# Paper Alignment Manifest

This repository is aligned to the submitted IEEE TIFS manuscript package:

```text
finalDOMCS_EEG_TIFS_Submission.pdf
```

The manuscript cites this repository as:

```text
https://github.com/KanimozhiL16/domcs-eeg-tifs
```

and cites the archived package DOI:

```text
https://doi.org/10.5281/zenodo.19666452
```

## Submitted Paper Table III Result

The primary B2T claim in the submitted manuscript is:

```text
DOMCS-EEG under strict Baseline-to-Task (B2T):
EER = 3.76 +/- 0.21 %
AUC = 0.9931 +/- 0.0007
CRR = 86.95 +/- 0.40 %
```

Repository sources:

```text
experiments/results/TABLE_III_main_results.csv
experiments/results/PAPER_RESULTS_LOCK.csv
```

`experiments/results/TABLE_IV_main_results.csv` is retained as a backward-compatible alias and contains the same submitted-paper values.

## Raw NVIDIA Evidence

The submitted paper table is preserved, and raw implementation evidence is also included so reviewers can see the numerical provenance.

| Evidence | File | Mean EER (%) | Std EER (%) | Mean AUC | Std AUC |
|---|---|---:|---:|---:|---:|
| Original NVIDIA A100 run, 2026-04-06 | `results/main_results/original_20260406_multi_seed_summary.csv` | 3.76 | 0.21 | 0.9931 | 0.0007 |
| Independent NVIDIA A100 rerun, 2026-04-28 | `results/main_results/rerun_20260428_multi_seed_summary.csv` | 3.82 | 0.29 | 0.9929 | 0.0011 |

The independent rerun is close to, but not exactly identical to, the submitted paper table. This is expected for GPU training unless a stricter deterministic pipeline is enforced. The repository therefore documents both the submitted paper table and the raw rerun evidence.

## Protocol Lock

The repository follows the protocol described in the manuscript:

```text
Dataset: EEGMMIDB
Subjects: 109
Sampling rate: 128 Hz
Window length: 2 s
Step: 1 s
Enrollment runs: R01, R02
Verification runs: R03-R14
Enrollment prototypes: K = 3
Classifier/metric head: ArcFace, scale = 30, margin = 0.5 rad
Contrastive loss temperature: 0.07
Independent seeds: 1, 2, 3, 4, 5
Epochs: 60
```

Configuration source:

```text
experiments/configs/main_60ep.yaml
```

Actual submitted-paper training hyperparameters:

```text
learning rate: 3e-4
weight decay: 1e-4
batch size: 256
loss weights: ArcFace 1.0, SupCon 0.5, State 0.5, Orthogonal 0.1
```

## Reviewer Verification

Run:

```bash
python scripts/verify_reproducibility.py
```

This recomputes means and standard deviations for the submitted paper table, original NVIDIA evidence, and independent NVIDIA rerun.

For the submitted paper table lock only, run:

```bash
python scripts/verify_paper_alignment.py
```

## Do Not Replace With Older Backups

Older local folders and notebooks can contain different values due to earlier drafts, partial protocol studies, 40-epoch runs, 200-epoch single-seed diagnostics, or mismatched backup copies. The final public repository should use the files documented in this manifest and in `REPRODUCIBILITY_AUDIT.md`.

Known non-canonical examples:

- `DOMCS_EEG_Q1_PAPER_BACKUP_3` is not the final submitted paper-aligned source.
- Older `TABLE_IV_final.csv` and notebook checkpoint values can contain draft numbers and should not replace submitted Table III unless the paper is revised.
- `run_200EP_SEED1` is an epoch-study/single-seed diagnostic, not the main 5-seed Table III result.

Canonical final GitHub evidence archive found on NVIDIA:

```text
DOMCS_EEG_GITHUB_FINAL_20260406_2305.zip
created: 2026-04-06 23:05:52 UTC
```
