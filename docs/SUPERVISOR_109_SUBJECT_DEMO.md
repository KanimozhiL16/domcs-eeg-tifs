# DOMCS-EEG 109-Subject Supervisor Demo

This demo uses the preprocessed EEGMMIDB NPZ file as a local database. The neural network is already trained from the paper workflow. The live app demonstrates biometric enrollment and verification for any of the 109 subjects.

## Database

Default local path:

```powershell
C:\Users\L.KANIMOZHI\Downloads\DAILY TASKS\APR 2026\21 APR 26\EEGMMIDB_win2s_step1s_fs128.npz
```

For another machine, set:

```powershell
$env:DOMCS_EEG_NPZ_PATH="C:\path\to\EEGMMIDB_win2s_step1s_fs128.npz"
```

The NPZ is not committed to GitHub because it is large. The code validates the expected keys: `X`, `y`, `subject_id`, `session`, `fs`, and `ch_names`.

## Run

```powershell
python run_realtime_auth.py
```

Open:

```text
http://127.0.0.1:8787
```

## Demonstration Steps

1. Confirm the Database Status panel shows 109 subjects.
2. Select any subject from the Subject Browser.
3. Click **Enroll Selected Subject**.
4. Click **Verify Same Subject** for a genuine verification attempt.
5. Select a different probe subject.
6. Click **Verify Impostor Subject**.
7. Read the Evidence Log: it records subject ID, windows used, similarity score, threshold, and accept/reject decision.

## Protocol

| Stage | Paper-aligned data |
|---|---|
| Enrollment | Baseline runs R01-R02 |
| Verification | Task runs R03-R14 |
| Representation | DOMCS-EEG identity embedding |
| Matching | Cosine similarity to enrolled template |
| Decision | Threshold-based accept/reject |

Enrollment in this app means creating a subject template from baseline EEG windows. It does not retrain the DOMCS-EEG neural network.

## Smoke Test

```powershell
python scripts\demo_smoke_test.py --subject 1 --impostor 2
```

Optional full subject audit:

```powershell
python scripts\demo_smoke_test.py --all-subjects
```

The full audit writes:

```text
demo_outputs\all_subject_demo_audit.csv
```
