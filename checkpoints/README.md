# Pretrained Checkpoints

Seed 1 model checkpoint is included.

## Load
```python
import torch
from domcs_eeg.model import DOMCSEEG

model = DOMCSEEG()
ckpt  = torch.load('checkpoints/seed_1/model_best.pt',
                   map_location='cpu', weights_only=False)
state = {k.replace('backbone.', ''): v
         for k, v in ckpt['model_state'].items()}
model.load_state_dict(state, strict=False)
model.eval()
```

## Results (Seed 1, 60 epochs, B2T protocol)
- EER: 3.73% | AUC: 0.9926 | CRR: 87.03%
