# DOMCSEEG is the public inference API equivalent of EEGDiSentangle
# defined in scripts/train_60ep.py. Architecturally identical;
# renamed for clarity in the public reproducibility release.
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGBackbone(nn.Module):
    """Three-layer Conv1D EEG backbone used by the 60-epoch paper run."""

    def __init__(self, n_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x.float()).squeeze(-1)


class DOMCSEEG(nn.Module):
    """
    DOMCS-EEG paper architecture.

    Input: (B, 64, 256), two-second EEG windows at 128 Hz.
    Output: identity embedding z_id and condition/state embedding z_cs.
    """

    def __init__(self, n_channels=64, emb_dim=128):
        super().__init__()
        self.backbone = EEGBackbone(n_channels)
        self.id_head = nn.Sequential(nn.Linear(256, emb_dim), nn.LayerNorm(emb_dim))
        self.state_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.cs_head = nn.Sequential(nn.Linear(256, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x):
        features = self.backbone(x)
        z_id = F.normalize(self.id_head(features), dim=1)
        z_cs = F.normalize(self.cs_head(features), dim=1)
        return z_id, z_cs

    def get_identity_embedding(self, x):
        """Inference-only identity embedding used by enrollment and verification."""
        self.eval()
        with torch.no_grad():
            z_id, _ = self.forward(x)
        return z_id

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        """Load a DOMCS-EEG inference model from a saved paper checkpoint."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint_path, map_location=device)
        if isinstance(payload, dict):
            state_dict = (
                payload.get("model_state_dict")
                or payload.get("model_state")
                or payload.get("state_dict")
                or payload.get("model")
                or payload
            )
        else:
            state_dict = payload

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value

        model = cls()
        load_result = model.load_state_dict(cleaned, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint does not match DOMCS-EEG architecture: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        model.to(device)
        model.eval()
        return model
