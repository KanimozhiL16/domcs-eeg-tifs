# DOMCS-EEG Result Report and Repository Audit

Prepared from the repository snapshot at:
`C:\Users\L.KANIMOZHI\OneDrive\Documents\Q1 PAPER1 31MAR26\10apr26\brev\DOMCS_EEG_GITHUB_FINAL_20260406_2305\DOMCS-EEG`

Date of audit: 2026-04-20

## 1. Executive Summary

This repository snapshot supports a strong Q1-style EEG biometric paper centered on protocol realism under cross-task evaluation. The core claim implemented in the codebase is that a baseline-to-task (B2T) evaluation protocol is substantially harder and more realistic than random-split or same-task protocols, and that a disentanglement-driven metric-learning model preserves strong verification performance under this harder setting.

From the tracked artefacts, the main reported performance is:

| Setting | EER (%) | AUC | Notes |
|---|---:|---:|---|
| Random split | 1.12 | 0.9991 | optimistic protocol |
| Same-task | 2.28 | 0.9969 | still easier than realistic deployment |
| B2T / proposed protocol | 3.75 ± 0.20 | 0.9928 ± 0.0008 | main result |
| Closed-set rank-1 style recognition rate (CRR) | 86.95 ± 0.40 | - | 109 subjects, 12 task runs |

The repository snapshot does contain security experimentation work. Specifically, it includes four Colab notebooks dedicated to attack analysis:

- `colab_domcs_attack_exhaustive_full.ipynb`
- `colab_domcs_attack_followup.ipynb`
- `colab_domcs_attack_lowram.ipynb`
- `colab_domcs_attack_paper_final.ipynb`

These notebooks implement adversarial/security evaluations for FGSM, PGD, targeted false-accept style attacks, and 50 Hz line-noise robustness. However, the repository snapshot does not preserve a clean tracked CSV/JSON security summary equivalent to the main biometric results table. Therefore:

- Yes, the repository does include the security experiments you did.
- No, the repository snapshot does not preserve them as cleanly as the main biometric results.
- The security work is present mainly as notebook-based experimental code, not as a finalized canonical results table.

One more critical audit finding: this is not a live git checkout. The folder contains `.gitignore`, but no `.git` directory, no commit history, and no remote metadata. So I can verify what is present in this repository snapshot, but I cannot certify what the online GitHub repo currently contains without the remote URL.

## 2. Research Idea Captured by the Implementation

The implemented research idea is scientifically sound and publication-relevant:

1. Train on baseline/resting-state EEG only.
2. Test on task-state EEG only.
3. Learn identity embeddings that remain discriminative across cognitive state shift.
4. Suppress state leakage through explicit auxiliary state supervision and embedding disentanglement.
5. Evaluate with biometric metrics, especially EER and ROC-AUC, rather than only classification accuracy.

This is exactly the right framing for EEG biometric authentication in a Q1 paper, because it attacks the main criticism of inflated protocols: many EEG biometric papers overestimate performance by using random splits or same-task splits that leak task/state structure into both train and test.

## 3. Verified Repository Contents

### 3.1 Main research artefacts present

- Main result table: `experiments/results/TABLE_IV_main_results.csv`
- Main config: `experiments/configs/main_60ep.yaml`
- Best checkpoint: `checkpoints/seed_1/model_best.pt`
- Main paper figures: `figures/FIG_05_ROC_DET_60ep.png`, `FIG_10_protocol_comparison_60ep.png`, `FIG_17_disentanglement_validation_60ep.png`, plus additional analysis figures
- Core evaluation package: `domcs_eeg/`
- Full research framework: `scripts/core_framework.py`, `scripts/experiment_groups.py`
- Real-time demo/service: `realtime_auth/`
- Local experience/demo app: `experience_app/`

### 3.2 Security artefacts present

The attack notebooks are not placeholders. They contain code for:

- clean B2T verification setup
- prototype map construction
- FGSM attack sweeps
- PGD attack sweeps
- targeted false-accept attack logic
- 50 Hz line-noise robustness sweeps
- merged summary creation and plotting

This means the security work exists in the repository snapshot at the implementation level.

### 3.3 What is not preserved in this snapshot

- no `.git` directory
- no commit hash
- no remote URL
- no branch information
- no canonical security results CSV in `experiments/results/`
- no standalone attack summary manuscript table

## 4. Implemented Model and Mathematical Formulation

The research framework in `scripts/core_framework.py` is the most publication-grade implementation in the repository and should be treated as the primary scientific reference.

### 4.1 Architecture

The full research model can be written as:

- shared EEG backbone: `f_theta(x)`
- identity embedding head: `z_id = normalize(g_id(f_theta(x)))`
- state embedding head: `z_state = normalize(g_state(f_theta(x)))`
- state classifier from identity branch: `s_hat = h(z_id)`

The intended learning signal is:

`L = L_arc + lambda_supcon * L_supcon + lambda_state * L_state + lambda_orth * L_orth`

with:

- `L_arc`: ArcFace angular-margin identity loss
- `L_supcon`: supervised contrastive loss
- `L_state`: cross-entropy loss for cognitive state supervision
- `L_orth`: orthogonality penalty between `z_id` and `z_state`

The orthogonality penalty is implemented as the mean squared cosine similarity between normalized identity and state embeddings:

`L_orth = mean( <z_id, z_state>^2 )`

This is a valid disentanglement regularizer for a paper, although it should be described as representation decorrelation rather than full causal disentanglement.

### 4.2 Locked training configuration

From `experiments/configs/main_60ep.yaml`, the main configuration is:

- subjects: 109
- channels: 64
- sampling rate: 128 Hz
- window length: 2.0 s
- step size: 1.0 s
- embedding dimension: 128
- epochs: 60
- batch size: 256
- learning rate: 3e-4
- weight decay: 1e-4
- seeds: 1, 2, 3, 4, 5
- ArcFace scale: 30.0
- ArcFace margin: 0.5
- SupCon temperature: 0.07
- `lambda_supcon = 0.5`
- `lambda_state = 0.5`
- `lambda_orth = 0.1`
- enrollment prototypes per subject: `K = 3`

### 4.3 Protocol

The primary protocol is B2T:

- train runs: `R01, R02`
- test runs: `R03-R14`
- train/validation separation: validation is created only from the train side
- model selection: based on validation loss, not test performance

This is methodologically strong and should be emphasized early in the paper.

## 5. Experimental Protocol Validity

The repository is unusually explicit about leakage prevention. The scientific logic is:

- training windows come only from baseline/resting runs
- validation windows come only from training-distribution baseline runs
- test windows come only from task-state runs
- test data is never used for model selection

This is exactly what a reviewer will want to see for a realistic biometric generalization study.

The framework also includes ablation-ready experiment groups:

- Group A: main reference experiment
- Group B: epoch study
- Group C: embedding dimension ablation
- Group D: prototype count ablation
- Group E: loss ablation
- Group F: enrollment size study
- Group G: protocol comparison
- Group H: enrollment strategy comparison
- Group I: per-run analysis
- Group J: disentanglement validation
- Group K: embedding visualization
- Group L: training dynamics
- Group M: computational practicality

This is excellent paper infrastructure.

## 6. Verified Main Results

### 6.1 Multi-seed B2T result

From `experiments/results/TABLE_IV_main_results.csv`:

| Seed | EER (%) | AUC | CRR (%) |
|---|---:|---:|---:|
| 1 | 3.73 | 0.9926 | 87.03 |
| 2 | 3.98 | 0.9925 | 86.77 |
| 3 | 3.89 | 0.9918 | 86.38 |
| 4 | 3.45 | 0.9938 | 87.15 |
| 5 | 3.70 | 0.9932 | 87.42 |
| Mean | 3.75 | 0.9928 | 86.95 |
| Std | 0.20 | 0.0008 | 0.40 |

Interpretation:

- The variance across seeds is low.
- The EER standard deviation of 0.20% is publication-friendly.
- The AUC remains consistently above 0.991 across all seeds.
- The model is stable enough to support a reproducibility claim.

### 6.2 Protocol comparison

From the README and figure inventory:

| Protocol | EER (%) | AUC | Relative difficulty vs random split |
|---|---:|---:|---:|
| Random split | 1.12 | 0.9991 | baseline |
| Same-task | 2.28 | 0.9969 | +104% EER |
| B2T | 3.75 | 0.9928 | +296% EER |

This protocol inflation result is one of the strongest parts of the work. It provides a clear and defensible narrative:

- conventional random split dramatically overestimates deployable performance
- same-task testing is still optimistic
- baseline-to-task testing is more realistic and substantially harder

### 6.3 Additional evidence present in the repository

The figure set indicates the following additional analyses were prepared:

- ROC/DET analysis
- score distribution analysis
- enrollment budget study
- protocol comparison visualization
- per-run EER variation
- task-type analysis
- session drift analysis
- identity heatmap
- subject heterogeneity
- t-SNE embeddings
- disentanglement validation
- training dynamics
- loss ablation
- interpretability via Grad-CAM, Grad-CAM++, integrated gradients, occlusion

This gives the paper a complete evidence stack beyond headline EER.

## 7. Security Experiment Audit

### 7.1 Answer to your direct question

Does the GitHub repository have the security experiments you did?

For this repository snapshot, yes. The security experiments are present in the delivered repo contents.

What is present:

- notebook-based attack evaluation code
- FGSM sweeps
- PGD sweeps
- targeted false-accept attack logic
- line-noise robustness analysis
- merged-summary generation code in notebooks

What is not preserved cleanly:

- no canonical tracked attack result CSV under `experiments/results/`
- no dedicated attack figure directory with final paper-labelled files
- no plain-text summary of final security numbers in `README.md`

### 7.2 Security scope implemented

The notebooks indicate three broad security dimensions:

1. Adversarial perturbation attacks
   - FGSM
   - PGD

2. Impersonation / false-accept oriented attacks
   - targeted attacks against incorrect claimed identity

3. Signal corruption robustness
   - 50 Hz line-noise robustness sweeps

This is a meaningful security package for a biometric paper, especially for TIFS-style positioning.

### 7.3 Current reporting weakness

The security experiments look real, but their reporting is weaker than the main biometric reporting because:

- they are notebook-centric
- final outputs are not surfaced in the main repository result tables
- the repository snapshot does not provide a single paper-ready attack summary table

For paper writing, you should treat the security section as implemented but not yet fully normalized into archival result artefacts.

## 8. Realtime and Productization Readiness

The repository already includes a usable deployment path.

### 8.1 FastAPI service

The `realtime_auth/` module exposes:

- health endpoint
- user listing
- enrollment endpoint
- verification endpoint
- identification endpoint
- master NPZ inspection and enrollment/verification
- EDF-based enrollment and verification
- threshold update endpoint

This is strong productization evidence for a research paper.

### 8.2 Runtime logic

The runtime service:

- loads the checkpoint
- extracts identity embeddings
- stores enrolled templates
- supports verification and identification
- accepts raw arrays, NPZ archives, and EDF inputs
- includes preprocessing for EDF:
  - channel selection
  - bandpass filtering
  - notch filtering
  - resampling to 128 Hz
  - z-normalization
  - 2 s windows with 1 s stride

### 8.3 Operational verification rule

The real-time authenticator uses:

- threshold = 0.58
- minimum enrollment windows = 3
- verification score = average of `0.6 * centroid similarity + 0.4 * max-template similarity`

This is deployable, but in the paper you should clarify that the deployed runtime decision rule is a product-oriented approximation layered on top of the research embedding model, not the full KMeans prototype evaluation used in the main experimental protocol.

## 9. Important Repository Inconsistencies You Should Know Before Writing

This section matters. These are the kinds of details that can cause confusion during paper drafting.

### 9.1 Two different model implementations exist

There are two distinct model paths:

1. `scripts/core_framework.py`
   - publication-oriented
   - parameter count: 305,666
   - richer architecture and experiment runner

2. `domcs_eeg/model.py`
   - compact runtime/inference package
   - parameter count: 243,970
   - simpler heads

The README reports `305,666 parameters`, which matches `scripts/core_framework.py`, not `domcs_eeg/model.py`.

Implication:

- for the paper, treat `scripts/core_framework.py` as the authoritative research implementation
- treat `domcs_eeg/` as the packaged inference/runtime implementation
- do not mix these two descriptions without explaining the distinction

### 9.2 README quickstart path mismatch

The README says:

`python scripts/evaluate.py --checkpoint checkpoints/seed_1/model_best.pt`

But the repository snapshot contains:

- `domcs_eeg/evaluate.py`
- not `scripts/evaluate.py`

This is a documentation inconsistency.

### 9.3 Portable reproducibility is incomplete

`scripts/train_60ep.py` contains hardcoded Linux-style paths and a hardcoded GPU selection:

- root path under `/home/nvidia/...`
- explicit `CUDA_VISIBLE_DEVICES = "1"`

That script appears to be the original experiment launcher rather than a portable final reproduction script.

### 9.4 Security experiments are not normalized into the same result structure

The main biometrics are archived cleanly.
The security experiments are archived primarily as notebooks.

That is acceptable for internal work, but weaker for release-quality reproducibility.

## 10. Publication-Grade Interpretation

### 10.1 What the results support strongly

The repository strongly supports the following claims:

- realistic cross-task EEG biometric verification is much harder than common optimistic protocols
- the DOMCS-style identity/state separation remains effective under cross-task shift
- the method is reproducible across random seeds
- the learned representation has good genuine/impostor separation
- the model is light enough for real-time inference
- the framework can be productized via API-backed verification

### 10.2 What must be stated carefully

Be careful with these claims:

- "disentanglement"
  - use as representation disentanglement/decorrelation, not full causal disentanglement

- "security robustness"
  - supported by implemented attack notebooks
  - but only fully claim final numeric robustness levels if you have the preserved attack outputs

- "fully reproducible from repo"
  - mostly true for the main protocol
  - weaker for security experiments because final outputs are not normalized into tracked result artefacts

## 11. Suggested Paper Result Narrative

A strong paper narrative from this repository is:

1. Existing EEG biometric evaluations are often protocol-inflated.
2. We define a strict baseline-to-task protocol using resting-state enrollment and task-state verification.
3. We propose a dual-objective cognitive-state disentanglement framework with identity metric learning.
4. Under this realistic protocol, performance remains strong at `3.75 ± 0.20%` EER and `0.9928 ± 0.0008` AUC over 109 subjects.
5. Simpler optimistic protocols understate the challenge by as much as `296%` in EER.
6. The learned representation is stable across seeds, interpretable, lightweight, and deployable.
7. Security-oriented analyses further probe adversarial impersonation and signal corruption robustness.

## 12. What You Can Reliably Write in the Paper Right Now

You can safely write the following based on the repository snapshot:

- dataset: EEGMMIDB, 109 subjects, 64 channels, 128 Hz, 2 s windows, 1 s stride
- protocol: train on R01-R02, test on R03-R14
- losses: ArcFace + SupCon + state supervision + orthogonality regularization
- evaluation: EER, AUC, CRR, protocol comparison
- multi-seed reproducibility across 5 seeds
- presence of interpretability analysis
- presence of computational practicality analysis
- existence of a real-time authentication system built on the learned embeddings
- existence of security/attack notebooks covering FGSM, PGD, targeted attack logic, and line-noise robustness

## 13. What Should Be Cleaned Before Final Submission

For a stronger final paper package, I recommend:

1. Create one canonical `security_results.csv` and `security_results.md`.
2. Export final attack figures from the notebook into `figures/` with stable names.
3. Add a short `SECURITY_EXPERIMENTS.md` explaining attack protocol, threat model, and preserved numbers.
4. Resolve the model duality by documenting:
   - research model path
   - deployment model path
5. Fix the README quickstart path mismatch.
6. Replace hardcoded Linux paths in training scripts with config-based paths.
7. Add one top-level reproduction script that runs the final paper pipeline end-to-end.

## 14. Final Audit Verdict

This repository snapshot is strong enough to support paper writing.

My research-scientist verdict is:

- main biometric evidence: strong
- protocol realism contribution: very strong
- reproducibility of primary result: strong
- interpretability and supporting analyses: strong
- deployment/product potential: strong
- security experiment presence: confirmed
- security experiment archival quality: moderate, not yet as clean as the main result tables

If you write the paper from this report, the key thing to remember is this:

The repository already contains the science, the main numbers, the protocol logic, the ablation structure, the deployment angle, and the security notebook work. What is still missing is not the research itself, but the final normalization of the security evidence into the same clean archival format as the main biometric experiments.
