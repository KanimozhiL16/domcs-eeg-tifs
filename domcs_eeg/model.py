import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGBackbone(nn.Module):
    def __init__(self, n_channels: int = 64):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DOMCSEEG(nn.Module):
    def __init__(self, n_channels: int = 64, emb_dim: int = 128):
        super().__init__()
        self.backbone = EEGBackbone(n_channels=n_channels)
        self.id_head = nn.Sequential(nn.Linear(256, emb_dim), nn.LayerNorm(emb_dim))
        self.state_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.cs_head = nn.Sequential(nn.Linear(256, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor):
        features = self.backbone(x.float())
        z_id = F.normalize(self.id_head(features), dim=1)
        z_cs = F.normalize(self.cs_head(features), dim=1)
        s_out = self.state_head(z_id)
        return z_id, z_cs, s_out, features

    def get_identity_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z_id, _, _, _ = self.forward(x)
        return z_id

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str | torch.device = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model = cls()
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model


class ArcFaceLoss(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = torch.cos(theta + self.m * one_hot) * self.s
        return F.cross_entropy(logits, labels)
