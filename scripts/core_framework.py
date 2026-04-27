"""
DOMCS-EEG Core Framework - 01_core_framework.py
=================================================
Single source of truth for ALL experiments.
Every experiment script imports from this file.

Contains:
  1. Global config
  2. Dataset loading
  3. Split builders (B2T locked + protocol variants)
  4. Model definition (exact locked architecture)
  5. Loss definitions
  6. Training function with checkpoint/resume
  7. Embedding extraction
  8. Evaluation (EER, AUC, KMeans prototypes)
  9. Aggregation utilities
  10. Plotting utilities
  11. Manifest/logging utilities

LEAKAGE PREVENTION:
  - Training data: ONLY R01, R02 windows
  - Test data:     ONLY R03-R14 windows
  - Val split:     80/20 stratified split of R01/R02 ONLY
  - No test data ever touches model selection
  - Best checkpoint = lowest val_loss on R01/R02 split
  - This is scientifically safe because val split is from
    the same distribution as training (resting state)
"""

import os, sys, json, csv, time, math, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1: GLOBAL CONFIG - CHANGE ONLY HERE
# ============================================================

CFG = {
    # Paths
    "ROOT":     "/home/nvidia/24PHD1237",
    "NPZ_PATH": "/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz",
    "EXP_ROOT": "/home/nvidia/24PHD1237/EXPERIMENTS",
    "PAPER_OUT":"/home/nvidia/24PHD1237/PAPER_OUTPUTS",

    # LOCKED protocol - DO NOT CHANGE
    "TRAIN_RUNS": ["r01", "r02"],
    "TEST_RUNS":  ["r03","r04","r05","r06","r07",
                   "r08","r09","r10","r11","r12","r13","r14"],

    # Locked model config
    "EMB_DIM":    128,
    "BATCH_SIZE": 256,
    "LR":         3e-4,
    "WEIGHT_DECAY": 1e-4,
    "TEMPERATURE":  0.07,
    "LAMBDA_SUPCON": 0.5,
    "LAMBDA_STATE":  0.5,
    "LAMBDA_ORTH":   0.1,
    "KMEANS_K":      3,

    # Training settings
    "EPOCHS":    60,
    "PATIENCE":  15,
    "VAL_FRAC":  0.20,
    "SEEDS":     [1, 2, 3, 4, 5],

    # Device
    "GPU_ID": "1",   # change if GPU 1 is busy
}

DEVICE = None  # set in setup_device()

def setup_device(gpu_id=None):
    global DEVICE
    gid = gpu_id or CFG["GPU_ID"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    return DEVICE

# ============================================================
# SECTION 2: PATH UTILITIES
# ============================================================

def make_run_dir(group_path, run_name=None):
    """Create a unique timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"run_{ts}"
    path = os.path.join(group_path, name)
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def save_csv_from_list(rows, path):
    if not rows: return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def save_manifest(run_dir, config_dict):
    """Save exact config for reproducibility."""
    manifest = {
        "created": datetime.now().isoformat(),
        "config": config_dict,
        "framework_version": "v1.0"
    }
    save_json(manifest, os.path.join(run_dir, "manifest.json"))

# ============================================================
# SECTION 3: DATASET LOADING
# ============================================================

_DATA_CACHE = None

def load_dataset(npz_path=None):
    """Load NPZ once, cache in memory for session."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    path = npz_path or CFG["NPZ_PATH"]
    print(f"Loading dataset: {path}")
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    Y = data["y"] if "y" in data.files else data["Y"]
    runs_raw = data["session"] if "session" in data.files else data["runs"]
    runs = np.array([_canon_run(r) for r in runs_raw], dtype=object)

    print(f"  X: {X.shape} | Y: {Y.shape} | Runs: {runs.shape}")
    print(f"  Unique runs: {sorted(set(runs))}")
    print(f"  Subjects: {len(np.unique(Y))}")

    _DATA_CACHE = (X, Y, runs)
    return _DATA_CACHE

def _canon_run(x):
    """Canonicalize run labels to r01, r02 ... r14 format."""
    x = str(x).lower().strip()
    x = x.replace("session","").replace("_","").replace("-","")
    if x.startswith("r") and x[1:].isdigit():
        return "r" + x[1:].zfill(2)
    if x.startswith("run") and x[3:].isdigit():
        return "r" + str(int(x[3:])).zfill(2)
    return x

# ============================================================
# SECTION 4: SPLIT BUILDERS
# ============================================================

def build_b2t_split(X, Y, runs,
                    train_runs=None,
                    test_runs=None,
                    val_frac=None,
                    seed=1):
    """
    Build the LOCKED Baseline-to-Task split.
    
    LEAKAGE PREVENTION:
    - Train/val uses ONLY R01/R02 (resting state)
    - Test uses ONLY R03-R14 (task state)
    - val split is a stratified subset of train only
    - Test set is NEVER used for model selection
    """
    tr_runs  = train_runs or CFG["TRAIN_RUNS"]
    te_runs  = test_runs  or CFG["TEST_RUNS"]
    val_frac = val_frac   or CFG["VAL_FRAC"]

    train_idx = np.array([i for i,r in enumerate(runs) if r in tr_runs])
    test_idx  = np.array([i for i,r in enumerate(runs) if r in te_runs])

    X_tr_np = X[train_idx]; Y_tr_np = Y[train_idx]
    X_te_np = X[test_idx];  Y_te_np = Y[test_idx]

    # Val split - from training data ONLY (no leakage)
    tr_idx_sub, val_idx_sub = train_test_split(
        np.arange(len(X_tr_np)),
        test_size=val_frac,
        stratify=Y_tr_np,
        random_state=seed
    )

    state_all   = np.array([0 if r in tr_runs else 1 for r in runs], dtype=np.int64)
    state_train = state_all[train_idx]
    state_test  = state_all[test_idx]

    split = {
        "X_train_np": X_tr_np, "Y_train_np": Y_tr_np,
        "X_test_np":  X_te_np, "Y_test_np":  Y_te_np,
        "state_train": state_train, "state_test": state_test,
        "tr_idx_sub": tr_idx_sub, "val_idx_sub": val_idx_sub,
        "runs_test": runs[test_idx],
        "train_runs": tr_runs, "test_runs": te_runs
    }
    print(f"  Split: Train={len(X_tr_np)} | Val={len(val_idx_sub)} | Test={len(X_te_np)}")
    return split


def build_random_split(X, Y, runs, test_frac=0.20, seed=1):
    """
    GROUP G ONLY - Random split protocol.
    NOT the main protocol. Used only for protocol comparison.
    Marked clearly to prevent confusion.
    """
    N = len(X)
    all_idx = np.arange(N)
    tr_idx, te_idx = train_test_split(all_idx, test_size=test_frac,
                                       stratify=Y, random_state=seed)
    state_all = np.zeros(N, dtype=np.int64)
    return {
        "X_train_np": X[tr_idx], "Y_train_np": Y[tr_idx],
        "X_test_np":  X[te_idx], "Y_test_np":  Y[te_idx],
        "state_train": state_all[tr_idx],
        "state_test":  state_all[te_idx],
        "tr_idx_sub": np.arange(len(tr_idx)),
        "val_idx_sub": np.arange(len(tr_idx)),
        "runs_test": runs[te_idx],
        "train_runs": ["random"], "test_runs": ["random"]
    }


def build_same_task_split(X, Y, runs, seed=1):
    """
    GROUP G ONLY - Same-task split.
    Randomly splits task runs into train/test.
    NOT the main protocol.
    """
    task_runs = CFG["TEST_RUNS"]
    task_idx  = np.array([i for i,r in enumerate(runs) if r in task_runs])
    X_t = X[task_idx]; Y_t = Y[task_idx]
    tr_idx, te_idx = train_test_split(np.arange(len(X_t)),
                                       test_size=0.20,
                                       stratify=Y_t,
                                       random_state=seed)
    state_all = np.ones(len(X_t), dtype=np.int64)
    return {
        "X_train_np": X_t[tr_idx], "Y_train_np": Y_t[tr_idx],
        "X_test_np":  X_t[te_idx], "Y_test_np":  Y_t[te_idx],
        "state_train": state_all[tr_idx],
        "state_test":  state_all[te_idx],
        "tr_idx_sub": np.arange(len(tr_idx)),
        "val_idx_sub": np.arange(len(tr_idx)),
        "runs_test": runs[task_idx][te_idx],
        "train_runs": ["same_task"], "test_runs": ["same_task"]
    }


def build_enrollment_fraction_split(split, fraction, seed=1):
    """
    GROUP F - Vary enrollment fraction per subject.
    Subsamples training data subject-wise.
    Test set is UNCHANGED.
    """
    X_tr = split["X_train_np"]
    Y_tr = split["Y_train_np"]
    S_tr = split["state_train"]

    if fraction >= 1.0:
        return split  # use full enrollment

    subjects = np.unique(Y_tr)
    keep_idx = []
    rng = np.random.RandomState(seed)
    for sid in subjects:
        idx = np.where(Y_tr == sid)[0]
        n_keep = max(1, int(len(idx) * fraction))
        chosen = rng.choice(idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())

    keep_idx = np.array(keep_idx)
    new_split = dict(split)
    new_split["X_train_np"] = X_tr[keep_idx]
    new_split["Y_train_np"] = Y_tr[keep_idx]
    new_split["state_train"] = S_tr[keep_idx]

    # rebuild val split
    tr_idx_sub, val_idx_sub = train_test_split(
        np.arange(len(keep_idx)),
        test_size=CFG["VAL_FRAC"],
        stratify=Y_tr[keep_idx],
        random_state=seed
    )
    new_split["tr_idx_sub"]  = tr_idx_sub
    new_split["val_idx_sub"] = val_idx_sub
    return new_split

# ============================================================
# SECTION 5: DATASET CLASS
# ============================================================

class EEGDataset(Dataset):
    def __init__(self, X, Y, S):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.S = torch.tensor(S, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.S[i]


def make_loaders(split, seed=1, batch_size=None):
    """Build train, val, and test loaders from split dict."""
    bs = batch_size or CFG["BATCH_SIZE"]
    X_tr = split["X_train_np"]
    Y_tr = split["Y_train_np"]
    S_tr = split["state_train"]
    tr_sub = split["tr_idx_sub"]
    val_sub = split["val_idx_sub"]

    tr_loader = DataLoader(
        EEGDataset(X_tr[tr_sub], Y_tr[tr_sub], S_tr[tr_sub]),
        batch_size=bs, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=True
    )
    vl_loader = DataLoader(
        EEGDataset(X_tr[val_sub], Y_tr[val_sub], S_tr[val_sub]),
        batch_size=bs, shuffle=False, num_workers=0
    )
    te_loader = DataLoader(
        EEGDataset(split["X_test_np"], split["Y_test_np"], split["state_test"]),
        batch_size=512, shuffle=False, num_workers=0
    )
    return tr_loader, vl_loader, te_loader

# ============================================================
# SECTION 6: MODEL DEFINITION (LOCKED ARCHITECTURE)
# ============================================================

class DOMCSBackbone(nn.Module):
    """Shared CNN backbone - exact architecture from paper."""
    def __init__(self, Cin=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(Cin, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # -> (B, 256)


class DOMCSModel(nn.Module):
    """
    Full DOMCS-EEG model.
    At inference: only z_id is used.
    z_state and state_pred are discarded after training.
    """
    def __init__(self, emb_dim=128, Cin=64):
        super().__init__()
        self.emb_dim  = emb_dim
        self.backbone = DOMCSBackbone(Cin)

        # Identity head: 256 -> emb_dim -> emb_dim
        self.id_head = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # State head: 256 -> 64 -> emb_dim
        self.state_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # State classifier: emb_dim -> 2
        self.state_cls = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        f      = self.backbone(x)
        z_id   = F.normalize(self.id_head(f),    dim=1)
        z_state= F.normalize(self.state_head(f), dim=1)
        s_pred = self.state_cls(z_id)  # suppress state from identity embedding
        return z_id, z_state, s_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# SECTION 7: LOSS DEFINITIONS
# ============================================================

class ArcFaceLayer(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s; self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, labels):
        cos   = F.linear(F.normalize(x), F.normalize(self.W))
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        oh    = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1)
        return torch.cos(theta + self.m * oh) * self.s


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        """features: (B, 1, D) or (B, D)"""
        if features.dim() == 3:
            features = features.squeeze(1)
        f   = F.normalize(features, dim=1)
        sim = torch.matmul(f, f.T) / self.T
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(f.device)
        mask.fill_diagonal_(0)
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0])
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        mean_log = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        return -mean_log.mean()


def compute_orth_loss(z_id, z_state):
    """Element-wise orthogonality penalty."""
    dot = (F.normalize(z_id, dim=1) * F.normalize(z_state, dim=1)).sum(dim=1)
    return (dot ** 2).mean()


def compute_combined_loss(z_id, z_state, s_pred, yb, sb, arc_layer, supcon,
                           lambdas=None, use_supcon=True, use_state=True,
                           use_orth=True):
    """
    Central loss function. Supports ablation via use_* flags.
    lambdas: dict with keys supcon, state, orth
    """
    lam = lambdas or {
        "supcon": CFG["LAMBDA_SUPCON"],
        "state":  CFG["LAMBDA_STATE"],
        "orth":   CFG["LAMBDA_ORTH"]
    }

    arc_loss = nn.CrossEntropyLoss()(arc_layer(z_id, yb), yb)

    sup_loss   = supcon(z_id.unsqueeze(1), yb) if use_supcon else torch.tensor(0.)
    state_loss = nn.CrossEntropyLoss()(s_pred, sb) if use_state else torch.tensor(0.)
    orth_loss  = compute_orth_loss(z_id, z_state) if use_orth else torch.tensor(0.)

    total = (arc_loss
             + lam["supcon"] * sup_loss
             + lam["state"]  * state_loss
             + lam["orth"]   * orth_loss)

    return total, {
        "arc": arc_loss.item(),
        "sup": sup_loss.item() if use_supcon else 0.,
        "state": state_loss.item() if use_state else 0.,
        "orth": orth_loss.item() if use_orth else 0.,
        "total": total.item()
    }

# ============================================================
# SECTION 8: CHECKPOINT UTILITIES
# ============================================================

def save_checkpoint(run_dir, epoch, model, arc_layer, optimizer,
                    val_loss, is_best=False):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "arc_state":   arc_layer.state_dict(),
        "opt_state":   optimizer.state_dict(),
        "val_loss":    val_loss
    }
    torch.save(state, os.path.join(run_dir, "checkpoint_last.pt"))
    if is_best:
        torch.save(state, os.path.join(run_dir, "checkpoint_best.pt"))


def load_checkpoint(run_dir, model, arc_layer, optimizer=None, prefer_best=True):
    """Resume from checkpoint if it exists."""
    fname = "checkpoint_best.pt" if prefer_best else "checkpoint_last.pt"
    path  = os.path.join(run_dir, fname)
    if not os.path.exists(path):
        path = os.path.join(run_dir, "checkpoint_last.pt")
    if not os.path.exists(path):
        return 0, float("inf")

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    arc_layer.load_state_dict(ckpt["arc_state"])
    if optimizer and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    print(f"  Resumed from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return ckpt["epoch"], ckpt["val_loss"]

# ============================================================
# SECTION 9: TRAINING FUNCTION
# ============================================================

def train_one_seed(seed, split, run_dir, config=None,
                   use_supcon=True, use_state=True, use_orth=True,
                   resume=False, verbose=True):
    """
    Train DOMCS-EEG for one seed.
    
    Returns: DataFrame of per-epoch metrics

    Checkpoint strategy (no leakage):
    - best checkpoint = lowest validation loss
    - val split = 20% of R01/R02 only
    - test set never used for model selection
    """
    cfg = config or {}
    epochs    = cfg.get("epochs",     CFG["EPOCHS"])
    patience  = cfg.get("patience",   CFG["PATIENCE"])
    emb_dim   = cfg.get("emb_dim",    CFG["EMB_DIM"])
    bs        = cfg.get("batch_size", CFG["BATCH_SIZE"])
    lr        = cfg.get("lr",         CFG["LR"])
    wd        = cfg.get("weight_decay",CFG["WEIGHT_DECAY"])
    T         = cfg.get("temperature", CFG["TEMPERATURE"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    tr_loader, vl_loader, _ = make_loaders(split, seed=seed, batch_size=bs)
    num_classes = int(len(np.unique(split["Y_train_np"][split["tr_idx_sub"]])))

    model     = DOMCSModel(emb_dim=emb_dim).to(DEVICE)
    arc_layer = ArcFaceLayer(emb_dim, num_classes).to(DEVICE)
    supcon    = SupConLoss(temperature=T)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arc_layer.parameters()),
        lr=lr, weight_decay=wd
    )

    start_ep = 0
    best_val = float("inf")
    pat_count = 0

    if resume:
        start_ep, best_val = load_checkpoint(seed_dir, model, arc_layer, optimizer)

    log_rows = []
    log_path = os.path.join(seed_dir, "train_log.csv")

    for ep in range(start_ep + 1, epochs + 1):
        t0 = time.time()

        # --- TRAIN ---
        model.train(); arc_layer.train()
        tl = ta = tc = tt = 0.0
        comp_sums = {"arc":0,"sup":0,"state":0,"orth":0}

        for xb, yb, sb in tr_loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            optimizer.zero_grad()
            z_id, z_state, s_pred = model(xb)
            loss, comps = compute_combined_loss(
                z_id, z_state, s_pred, yb, sb,
                arc_layer, supcon,
                use_supcon=use_supcon,
                use_state=use_state,
                use_orth=use_orth
            )
            loss.backward()
            optimizer.step()

            tl += comps["total"]
            for k in comp_sums: comp_sums[k] += comps[k]
            preds = arc_layer(z_id, yb).argmax(1)
            tc += (preds == yb).sum().item()
            tt += len(yb)

        nb = len(tr_loader)
        tl /= nb; ta = tc / tt
        for k in comp_sums: comp_sums[k] /= nb

        # --- VALIDATE ---
        model.eval(); arc_layer.eval()
        vl = 0.0; vc = 0; vt = 0
        with torch.no_grad():
            for xb, yb, sb in vl_loader:
                xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                z_id, z_state, s_pred = model(xb)
                _, comps_v = compute_combined_loss(
                    z_id, z_state, s_pred, yb, sb,
                    arc_layer, supcon,
                    use_supcon=use_supcon,
                    use_state=use_state,
                    use_orth=use_orth
                )
                vl += comps_v["total"]
                preds = arc_layer(z_id, yb).argmax(1)
                vc += (preds == yb).sum().item()
                vt += len(yb)
        vl /= len(vl_loader)
        va = vc / vt

        elapsed = time.time() - t0
        is_best = vl < best_val

        if verbose:
            flag = "OK" if is_best else "  "
            print(f"[s={seed}] Ep {ep:03d}/{epochs} | "
                  f"tr={tl:.4f} vl={vl:.4f} | "
                  f"tr_acc={ta:.4f} vl_acc={va:.4f} | "
                  f"arc={comp_sums['arc']:.4f} | "
                  f"t={elapsed:.1f}s {flag}")

        row = {
            "epoch": ep,
            "tr_loss": tl, "val_loss": vl,
            "tr_acc": ta,  "val_acc": va,
            "arc": comp_sums["arc"],
            "sup": comp_sums["sup"],
            "state": comp_sums["state"],
            "orth": comp_sums["orth"]
        }
        log_rows.append(row)

        # Checkpoint
        if is_best:
            best_val = vl
            pat_count = 0
        else:
            pat_count += 1

        save_checkpoint(seed_dir, ep, model, arc_layer, optimizer, vl, is_best)

        # Save CSV incrementally
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        if pat_count >= patience:
            print(f"  Early stop at epoch {ep} (patience={patience})")
            break

    df_log = pd.DataFrame(log_rows)
    return df_log, model, arc_layer

# ============================================================
# SECTION 10: EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(model, X_np, Y_np, S_np, batch_size=512):
    """Extract L2-normalized identity embeddings."""
    model.eval()
    ds = EEGDataset(X_np, Y_np, S_np)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs = []
    with torch.no_grad():
        for xb, _, _ in loader:
            z_id, _, _ = model(xb.to(DEVICE))
            embs.append(z_id.cpu().numpy())
    E = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E / norms

# ============================================================
# SECTION 11: EVALUATION
# ============================================================

def compute_auc_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    A = auc(fpr, tpr)
    fnr = 1.0 - tpr
    k   = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[k] + fnr[k]) / 2.0
    return float(A), float(eer)


def build_prototypes(E_train, Y_train_np, K=3):
    """Build KMeans prototypes per subject."""
    subjects = sorted(np.unique(Y_train_np).tolist())
    pvecs = []; powner = []
    for sid in subjects:
        idx = np.where(Y_train_np == sid)[0]
        Xs  = E_train[idx]
        if len(Xs) < K:
            c = Xs.mean(0); c /= (np.linalg.norm(c) + 1e-12)
            pvecs.append(c.astype(np.float32))
            powner.append(int(sid))
            continue
        km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xs)
        for c in km.cluster_centers_:
            c /= (np.linalg.norm(c) + 1e-12)
            pvecs.append(c.astype(np.float32))
            powner.append(int(sid))
    return np.stack(pvecs), np.array(powner, dtype=np.int64)


def build_mean_prototype(E_train, Y_train_np):
    """GROUP H - Mean prototype (centroid) per subject."""
    subjects = sorted(np.unique(Y_train_np).tolist())
    pvecs = []; powner = []
    for sid in subjects:
        idx = np.where(Y_train_np == sid)[0]
        c = E_train[idx].mean(0)
        c /= (np.linalg.norm(c) + 1e-12)
        pvecs.append(c.astype(np.float32))
        powner.append(int(sid))
    return np.stack(pvecs), np.array(powner, dtype=np.int64)


def evaluate_on_test(E_train, Y_train_np, E_test, Y_test_np,
                     runs_test, test_runs, K=3):
    """
    Full B2T evaluation.
    Returns per-run rows and aggregate AUC/EER.
    """
    pvecs, powner = build_prototypes(E_train, Y_train_np, K=K)
    rows = []
    for run_name in test_runs:
        pidx   = np.where(runs_test == run_name)[0]
        if len(pidx) == 0:
            continue
        P      = E_test[pidx]
        y_run  = Y_test_np[pidx]
        sm     = P @ pvecs.T
        scores = []; labels = []
        for i in range(len(P)):
            yt = y_run[i]
            mg = (powner == yt); mi = (powner != yt)
            scores.append(float(sm[i, mg].max())); labels.append(1)
            scores.extend(sm[i, mi].tolist());     labels.extend([0] * mi.sum())
        A, E = compute_auc_eer(np.array(scores), np.array(labels, np.int32))
        rows.append({"run": run_name, "auc": A, "eer": E, "n_windows": len(pidx)})

    avg_auc = float(np.mean([r["auc"] for r in rows]))
    avg_eer = float(np.mean([r["eer"] for r in rows]))
    return avg_auc, avg_eer, rows, pvecs, powner


def run_full_evaluation(model, split, seed_dir, K=3,
                        save_embeddings=False):
    """Extract embeddings, build prototypes, evaluate on all test runs."""
    sp = split
    E_train = extract_embeddings(model,
                                  sp["X_train_np"], sp["Y_train_np"],
                                  sp["state_train"])
    E_test  = extract_embeddings(model,
                                  sp["X_test_np"], sp["Y_test_np"],
                                  sp["state_test"])

    avg_auc, avg_eer, per_run, pvecs, powner = evaluate_on_test(
        E_train, sp["Y_train_np"],
        E_test,  sp["Y_test_np"],
        sp["runs_test"], sp["test_runs"], K=K
    )

    if save_embeddings:
        np.save(os.path.join(seed_dir, "E_train.npy"), E_train)
        np.save(os.path.join(seed_dir, "E_test.npy"),  E_test)
        np.save(os.path.join(seed_dir, "pvecs.npy"),   pvecs)
        np.save(os.path.join(seed_dir, "powner.npy"),  powner)

    save_csv_from_list(per_run, os.path.join(seed_dir, "per_run_results.csv"))
    save_json({"avg_auc": avg_auc, "avg_eer": avg_eer, "per_run": per_run},
              os.path.join(seed_dir, "eval_summary.json"))

    return avg_auc, avg_eer, E_train, E_test

# ============================================================
# SECTION 12: MULTI-SEED RUNNER
# ============================================================

def run_multi_seed(group_dir, split_fn, train_cfg, exp_name,
                   seeds=None, use_supcon=True, use_state=True,
                   use_orth=True, K=3, resume=False,
                   save_embeddings=False):
    """
    Run training + evaluation across multiple seeds.
    Returns aggregate summary dict.
    TRAINING-REQUIRED experiment.
    """
    seeds = seeds or CFG["SEEDS"]
    run_dir = make_run_dir(group_dir, exp_name)
    save_manifest(run_dir, {**train_cfg, "seeds": seeds,
                             "use_supcon": use_supcon,
                             "use_state": use_state,
                             "use_orth": use_orth, "K": K})

    X, Y, runs = load_dataset()
    all_rows = []

    for seed in seeds:
        print(f"\n{'='*65}")
        print(f"  SEED {seed} | {exp_name}")
        print(f"{'='*65}")
        split = split_fn(X, Y, runs, seed=seed)

        df_log, model, arc_layer = train_one_seed(
            seed, split, run_dir, config=train_cfg,
            use_supcon=use_supcon, use_state=use_state,
            use_orth=use_orth, resume=resume
        )

        # Load best checkpoint for evaluation
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        ckpt = torch.load(os.path.join(seed_dir, "checkpoint_best.pt"),
                          map_location=DEVICE)
        nc = int(len(np.unique(split["Y_train_np"][split["tr_idx_sub"]])))
        model_best = DOMCSModel(emb_dim=train_cfg.get("emb_dim", CFG["EMB_DIM"])).to(DEVICE)
        arc_best   = ArcFaceLayer(train_cfg.get("emb_dim", CFG["EMB_DIM"]), nc).to(DEVICE)
        model_best.load_state_dict(ckpt["model_state"])
        arc_best.load_state_dict(ckpt["arc_state"])

        avg_auc, avg_eer, _, _ = run_full_evaluation(
            model_best, split, seed_dir, K=K,
            save_embeddings=save_embeddings
        )

        print(f"  [seed={seed}] AUC={avg_auc:.4f} | EER={avg_eer*100:.2f}%")
        all_rows.append({
            "seed": seed, "avg_auc": avg_auc, "avg_eer": avg_eer,
            "best_val_loss": float(df_log["val_loss"].min()),
            "best_val_acc":  float(df_log["val_acc"].max()),
            "n_epochs": len(df_log)
        })

    # Aggregate
    df_agg = pd.DataFrame(all_rows)
    df_agg.to_csv(os.path.join(run_dir, "multiseed_summary.csv"), index=False)

    summary = {
        "exp_name": exp_name,
        "mean_auc": float(df_agg["avg_auc"].mean()),
        "std_auc":  float(df_agg["avg_auc"].std()),
        "mean_eer": float(df_agg["avg_eer"].mean()),
        "std_eer":  float(df_agg["avg_eer"].std()),
        "min_eer":  float(df_agg["avg_eer"].min()),
        "max_eer":  float(df_agg["avg_eer"].max()),
        "rows":     all_rows
    }
    save_json(summary, os.path.join(run_dir, "aggregate_summary.json"))

    print(f"\n{'='*65}")
    print(f"  AGGREGATE | {exp_name}")
    print(f"  AUC:  {summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f}")
    print(f"  EER:  {summary['mean_eer']*100:.2f}% +/- {summary['std_eer']*100:.2f}%")
    print(f"{'='*65}")

    return summary, run_dir

# ============================================================
# SECTION 13: PLOTTING UTILITIES
# ============================================================

def plot_learning_curves(log_csv_path, save_dir, title="", seed=1):
    """Plot train/val loss and accuracy curves."""
    df = pd.read_csv(log_csv_path)
    ep = df["epoch"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    if title: fig.suptitle(title, fontsize=12, fontweight="bold")

    # Loss
    axes[0].plot(ep, df["tr_loss"],  "b-",  lw=2, label="Train")
    axes[0].plot(ep, df["val_loss"], "r--", lw=2, label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss - Seed {seed}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(ep, df["tr_acc"]*100,  "b-",  lw=2, label="Train")
    axes[1].plot(ep, df["val_acc"]*100, "r--", lw=2, label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"Accuracy - Seed {seed}")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Component losses
    for col, label, ls in [
        ("arc",   "ArcFace",     "-"),
        ("sup",   "SupCon",      "--"),
        ("state", "State",       "-."),
        ("orth",  "Orthogonality",":")
    ]:
        if col in df.columns:
            axes[2].plot(ep, df[col], ls, lw=1.5, label=label)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Component Loss")
    axes[2].set_title(f"Components - Seed {seed}")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"learning_curves_seed{seed}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path


def plot_multi_seed_curves(run_dir, seeds, save_dir):
    """Overlay all seed curves on one figure."""
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, seed in enumerate(seeds):
        log_path = os.path.join(run_dir, f"seed_{seed}", "train_log.csv")
        if not os.path.exists(log_path): continue
        df = pd.read_csv(log_path)
        ep = df["epoch"].values
        axes[0].plot(ep, df["tr_loss"],  color=colors[i], lw=1.5,
                     label=f"S{seed} train")
        axes[0].plot(ep, df["val_loss"], color=colors[i], lw=1.5,
                     linestyle="--", alpha=0.5, label=f"S{seed} val")
        axes[1].plot(ep, df["val_acc"]*100, color=colors[i], lw=1.5,
                     label=f"Seed {seed}")

    for ax, ylabel, title in [
        (axes[0], "Loss", "Loss - All Seeds"),
        (axes[1], "Val Accuracy (%)", "Val Accuracy - All Seeds")
    ]:
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "multiseed_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path


def plot_ablation_bar(results_list, metric_key, ylabel, title, save_path):
    """Generic bar chart for ablation tables."""
    labels  = [r["label"] for r in results_list]
    values  = [r[metric_key] for r in results_list]
    errors  = [r.get(metric_key+"_std", 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.2), 5))
    bars = ax.bar(labels, values, yerr=errors, capsize=4,
                  color="#1f77b4", alpha=0.85)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val*100:.2f}%" if metric_key == "mean_eer" else f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


def plot_per_run_bar(per_run_rows, save_path, metric="eer"):
    """Bar chart for per-run evaluation (Group I)."""
    runs   = [r["run"] for r in per_run_rows]
    values = [r[metric]*100 for r in per_run_rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(runs, values, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Task Run"); ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_title(f"Per-Run {metric.upper()} - B2T Protocol", fontsize=13, fontweight="bold")
    ax.axhline(np.mean(values), color="red", linestyle="--", label=f"Mean={np.mean(values):.2f}%")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()

# ============================================================
# SECTION 14: PAPER TABLE EXPORTER
# ============================================================

def export_paper_table(rows, columns, title, save_path):
    """Export a clean CSV table ready for LaTeX or Word."""
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(save_path, index=False)
    print(f"OK Paper table saved: {save_path}")
    print(df.to_string(index=False))
    return df


def build_aggregate_paper_tables(experiments_root, output_dir):
    """
    Walk all experiment subdirectories and collect
    aggregate_summary.json files into one master table.
    """
    rows = []
    for dirpath, _, files in os.walk(experiments_root):
        if "aggregate_summary.json" in files:
            with open(os.path.join(dirpath, "aggregate_summary.json")) as f:
                s = json.load(f)
            rows.append({
                "experiment":  s.get("exp_name", os.path.basename(dirpath)),
                "mean_auc":    s.get("mean_auc", 0),
                "std_auc":     s.get("std_auc", 0),
                "mean_eer_%":  round(s.get("mean_eer", 0)*100, 2),
                "std_eer_%":   round(s.get("std_eer", 0)*100, 2),
            })
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "MASTER_RESULTS_TABLE.csv")
        df.to_csv(path, index=False)
        print(f"\nOK Master results table: {path}")
        print(df.to_string(index=False))
        return df
    return None

# ============================================================
# READY MESSAGE
# ============================================================

if __name__ == "__main__":
    print("DOMCS-EEG Core Framework loaded.")
    print("Import this file in experiment scripts.")
    print(f"ROOT:     {CFG['ROOT']}")
    print(f"NPZ:      {CFG['NPZ_PATH']}")
    print(f"EPOCHS:   {CFG['EPOCHS']}")
    print(f"SEEDS:    {CFG['SEEDS']}")

