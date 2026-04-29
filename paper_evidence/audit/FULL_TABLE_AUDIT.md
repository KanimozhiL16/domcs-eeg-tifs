# Full Table Evidence Audit

This folder maps the submitted DOMCS-EEG TIFS manuscript values to repository evidence files.

## Canonical Main Result

The repository lock is:

- B2T EER: 3.76 +/- 0.21 %
- B2T AUC: 0.9931 +/- 0.0007
- CRR: 86.95 +/- 0.40 %
- Seeds: 1, 2, 3, 4, 5
- Protocol: baseline-to-task

Files:

- `experiments/results/TABLE_IV_main_results.csv`
- `experiments/results/PAPER_RESULTS_LOCK.csv`
- `paper_evidence/master/MASTER_RESULTS_TABLE.csv`

## Evidence Map

| Manuscript area | Evidence file | Evidence type |
|---|---|---|
| Main B2T performance | `experiments/results/TABLE_IV_main_results.csv` | repository lock |
| Master experiment summary | `paper_evidence/master/MASTER_RESULTS_TABLE.csv` | raw exported CSV |
| Baseline comparison | `paper_evidence/baselines/comparison_table.csv` | raw exported CSV |
| Security summary | `paper_evidence/security/TIFS_security_summary.csv` | raw exported CSV |
| Forgery evaluation | `paper_evidence/security/forgery_results.csv` | raw exported CSV |
| Noise/security attacks | `paper_evidence/security/attack_noise.csv` | raw exported CSV |
| Leakage/security attacks | `paper_evidence/security/attack_leakage.csv` | raw exported CSV |
| Mimicry/security attacks | `paper_evidence/security/attack_mimicry.csv` | raw exported CSV |
| White-box attacks | `paper_evidence/security/attack_whitebox_corrected.csv`, `paper_evidence/security/attack_whitebox_embedding.csv` | raw exported CSV |
| Merged security attack table | `paper_evidence/security/attack_summary_merged.csv` | raw exported CSV |
| Targeted false accept | `paper_evidence/security/targeted_false_accept_summary.csv` | raw exported CSV |
| Interpretability channel table | `paper_evidence/interpretability/TABLE_INTERPRETABILITY.csv` | raw exported CSV |
| Representation/disentanglement summary | `paper_evidence/interpretability/TABLE_IV_representation_summary_paper_values.csv` | paper-extracted lock |
| Protocol comparison | `paper_evidence/protocol/TABLE_V_protocol_comparison_paper_values.csv` | paper-extracted lock |
| BED validation | `paper_evidence/bed/TABLE_VIII_bed_validation_paper_values.csv` | paper-extracted lock |
| Supplementary constants/statistics | `paper_evidence/supplementary/SUPPLEMENTARY_VALUES_LOCK.csv` | paper-extracted lock |

## Important Note

Some values were found as raw structured result CSVs. Other values were present in the submitted paper/notebook text but not as standalone raw CSV result files. Those values are kept as `paper-extracted lock` CSVs so the repository does not silently mix raw logs with manually reported manuscript values.

This is intentional for research integrity: the repository distinguishes reproducible raw exports from paper-table locks.
