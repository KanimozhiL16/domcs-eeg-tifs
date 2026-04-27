# Paper Alignment Manifest

This repository was prepared to match the final manuscript package:

`DOMCS_EEG_TIFS_20260427_FINAL_v7_COMPLETE_MAIN.pdf`

## Canonical Paper Result

The primary claim in the manuscript is:

```text
DOMCS-EEG under strict Baseline-to-Task (B2T):
EER = 3.75 +/- 0.20 %
AUC = 0.9928 +/- 0.0008
CRR = 86.95 +/- 0.40 %
```

Repository source:

```text
experiments/results/TABLE_IV_main_results.csv
```

## Protocol Lock

The repository follows the same protocol described in the manuscript:

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
```

Configuration source:

```text
experiments/configs/main_60ep.yaml
```

## Files Corrected for Public Release

The original local candidate folder already contained the paper-matching result CSV, model code, figures, and runtime demo. For this GitHub-ready copy, the public metadata was corrected so that it does not contradict the manuscript:

- README venue changed from IEEE Access to IEEE TIFS.
- README B2T result fixed to `3.75 +/- 0.20`, `0.9928 +/- 0.0008`, `86.95 +/- 0.40`.
- README protocol comparison fixed to manuscript values where listed.
- Broken mojibake characters replaced with ASCII.
- `runtime_store/registry.json` changed from an absolute local path to a portable relative path.
- Git ignore rules extended for datasets, caches, virtual environments, and generated run outputs.

## Do Not Replace With These Local Backups

The folder `DOMCS_EEG_Q1_PAPER_BACKUP_3` is not the final paper-aligned version because its multi-seed summary reports about `4.51%` EER, not the final manuscript value of `3.75% +/- 0.20%`.

The canonical local source for this repository is:

```text
DOMCS_EEG_GITHUB_FINAL_20260406_2305/DOMCS-EEG
```

with the public-facing corrections listed above.
