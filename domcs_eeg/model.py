import torch
import torch.nn as nn
import torch.nn.functional as F

class DOMCSEEG(nn.Module):
    """
    DOMCS-EEG: Disentangled Multi-Component Supervised EEG Framework
    Architecture: 3-layer Conv1D backbone + identity/state heads
    Input: (B, 64, 256) - 64-channel EEG, 2s window at 128Hz
    Output: z_id (B,128), s_logit (B,2), h (B,256)
    """
    def __init__(self, n_channels=64, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64,  7, padding=3),
            nn.BatchNorm1d(64),  nn.ELU(),
            nn.Conv1d(64, 128,   5, padding=2),
            nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, 256,  3, padding=1),
            nn.BatchNorm1d(256), nn.ELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.id_head = nn.Sequential(
            nn.Linear(256, emb_dim), nn.BatchNorm1d(emb_dim))
        self.state_head = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.cs_head = nn.Sequential(
            nn.Linear(256, emb_dim), nn.BatchNorm1d(emb_dim))

    def forward(self, x):
        h    = self.net(x.float())
        z_id = F.normalize(self.id_head(h), dim=1)
        s_out= self.state_head(z_id)
        return z_id, s_out, h

    def get_identity_embedding(self, x):
        """Inference-only: returns L2-normalised identity embedding"""
        self.eval()
        with torch.no_grad():
            z_id, _, _ = self.forward(x)
        return z_id

