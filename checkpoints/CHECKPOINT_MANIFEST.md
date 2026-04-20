# Model Checkpoints — DOMCS-EEG (60 Epochs, Brev A100-SXM4-80GB)

## Location on disk
All 5 seed checkpoints are in:
`10apr26/brev/DOMCS_EEG_Q1_PAPER_BACKUP_3/DOMCS_EEG_Q1_PAPER/checkpoints/seed_{1..5}/model_best.pt`

Also: `10apr26/brev/DOMCS_EEG_GITHUB_FINAL_20260406_2305/DOMCS-EEG/checkpoints/seed_1/model_best.pt`

## Checkpoint format
Each .pt is a dict: {"epoch": int, "model_state": OrderedDict, "arc_state": OrderedDict, "val_loss": float}
Load with: `torch.load("model_best.pt", map_location="cpu")`

## Best validation losses by seed
| Seed | Val Loss (best) | Notes |
|------|----------------|-------|
| 1 | See train_log_seed_1.csv | EER=3.69%, AUC=0.9925 |
| 2 | See train_log_seed_2.csv | EER=3.81%, AUC=0.9931 |
| 3 | See train_log_seed_3.csv | EER=4.01%, AUC=0.9926 |
| 4 | See train_log_seed_4.csv | EER=3.45%, AUC=0.9943 |
| 5 | See train_log_seed_5.csv | EER=3.87%, AUC=0.9931 |

## Loading model for inference
```python
from domcs_eeg.model import DOMCSEEG
model = DOMCSEEG.from_checkpoint("model_best.pt", device="cpu")
# z_id = model.get_identity_embedding(eeg_tensor)
```
