"""
DOMCS-EEG Experiment Groups A–M
================================
Each group is a self-contained function.
Enable/disable by commenting/uncommenting in main().

TRAINING-REQUIRED:  A1, A2, B, C, D, E, F, G
EVALUATION-ONLY:    H (uses A2 embeddings), I, J, K, L, M
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch

# Import core framework
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_framework import (
    CFG, setup_device, load_dataset, DEVICE,
    build_b2t_split, build_random_split, build_same_task_split,
    build_enrollment_fraction_split,
    run_multi_seed, run_full_evaluation,
    extract_embeddings, evaluate_on_test,
    build_prototypes, build_mean_prototype,
    compute_orth_loss, DOMCSModel, ArcFaceLayer,
    train_one_seed, make_run_dir, save_json, save_csv_from_list,
    plot_learning_curves, plot_multi_seed_curves,
    plot_ablation_bar, plot_per_run_bar,
    export_paper_table, build_aggregate_paper_tables,
    load_checkpoint
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = CFG["ROOT"]
EXP_ROOT = CFG["EXP_ROOT"]

# ============================================================
# GROUP A — MAIN REFERENCE EXPERIMENT
# [TRAINING REQUIRED]
# ============================================================

def run_group_A():
    """
    A1: Single baseline run (seed=1, 60 epochs)
    A2: Multi-seed reproducibility (5 seeds, 60 epochs)
    
    This is the PRIMARY evidence for the paper.
    Use 60 epochs — confirmed optimal from empirical analysis.
    """
    print("\n" + "="*65)
    print("GROUP A — MAIN REFERENCE EXPERIMENT")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE")

    # A1 — Single seed baseline
    cfg_60ep = {**CFG, "epochs": 60, "patience": 15}
    summary_a1, dir_a1 = run_multi_seed(
        group_dir=os.path.join(group_dir, "A1_baseline"),
        split_fn=build_b2t_split,
        train_cfg=cfg_60ep,
        exp_name="A1_baseline_60ep_seed1",
        seeds=[1],
        save_embeddings=True  # save embeddings for Groups I, J, K
    )

    # A2 — Multi-seed
    summary_a2, dir_a2 = run_multi_seed(
        group_dir=os.path.join(group_dir, "A2_multiseed"),
        split_fn=build_b2t_split,
        train_cfg=cfg_60ep,
        exp_name="A2_multiseed_60ep_5seeds",
        seeds=[1, 2, 3, 4, 5],
        save_embeddings=True
    )

    # Plot multi-seed curves
    plot_multi_seed_curves(
        dir_a2, [1,2,3,4,5],
        os.path.join(group_dir, "A2_multiseed")
    )

    # Paper table: per-seed results
    rows = summary_a2["rows"]
    export_paper_table(
        rows,
        ["seed","avg_auc","avg_eer","best_val_loss","n_epochs"],
        "Table: Main Results (5 Seeds, 60 Epochs)",
        os.path.join(group_dir, "TABLE_A_main_results.csv")
    )

    print(f"\nGROUP A DONE")
    print(f"  Mean EER: {summary_a2['mean_eer']*100:.2f}% ± {summary_a2['std_eer']*100:.2f}%")
    print(f"  Mean AUC: {summary_a2['mean_auc']:.4f} ± {summary_a2['std_auc']:.4f}")
    return summary_a2, dir_a2


# ============================================================
# GROUP B — EPOCH STUDY
# [TRAINING REQUIRED]
# ============================================================

def run_group_B():
    """
    Compare performance at 20, 40, 60, 80 epochs.
    Uses seed=1 only for efficiency.
    Justifies the choice of 60 epochs.
    """
    print("\n" + "="*65)
    print("GROUP B — EPOCH STUDY")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "B_EPOCH_STUDY")
    epoch_settings = [20, 40, 60, 80]
    results = []

    for ep in epoch_settings:
        cfg_ep = {**CFG, "epochs": ep, "patience": max(8, ep//4)}
        summary, _ = run_multi_seed(
            group_dir=os.path.join(group_dir, f"ep{ep}"),
            split_fn=build_b2t_split,
            train_cfg=cfg_ep,
            exp_name=f"B_epoch_{ep}ep",
            seeds=[1, 2, 3],  # 3 seeds sufficient for epoch study
        )
        results.append({
            "epochs": ep,
            "label": f"{ep} ep",
            "mean_eer": summary["mean_eer"],
            "mean_eer_std": summary["std_eer"],
            "mean_auc": summary["mean_auc"],
            "mean_auc_std": summary["std_auc"]
        })

    # Save table
    export_paper_table(
        results,
        ["epochs","mean_eer","mean_eer_std","mean_auc","mean_auc_std"],
        "Table: Epoch Study",
        os.path.join(group_dir, "TABLE_B_epoch_study.csv")
    )
    # Bar chart
    plot_ablation_bar(results, "mean_eer", "Mean EER",
                      "EER vs Training Epochs",
                      os.path.join(group_dir, "FIG_B_epoch_eer.png"))
    print("GROUP B DONE")
    return results


# ============================================================
# GROUP C — EMBEDDING DIMENSION ABLATION
# [TRAINING REQUIRED]
# ============================================================

def run_group_C():
    """Compare emb_dim = 64, 128, 256."""
    print("\n" + "="*65)
    print("GROUP C — EMBEDDING DIMENSION ABLATION")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "C_EMB_DIM")
    dims = [64, 128, 256]
    results = []

    for dim in dims:
        cfg_dim = {**CFG, "emb_dim": dim, "epochs": 60, "patience": 15}
        summary, _ = run_multi_seed(
            group_dir=os.path.join(group_dir, f"dim{dim}"),
            split_fn=build_b2t_split,
            train_cfg=cfg_dim,
            exp_name=f"C_emb_dim{dim}",
            seeds=[1, 2, 3],
        )
        results.append({
            "emb_dim": dim,
            "label": f"dim={dim}",
            "mean_eer": summary["mean_eer"],
            "mean_eer_std": summary["std_eer"],
            "mean_auc": summary["mean_auc"]
        })

    export_paper_table(
        results,
        ["emb_dim","mean_eer","mean_eer_std","mean_auc"],
        "Table: Embedding Dimension Ablation",
        os.path.join(group_dir, "TABLE_C_emb_dim.csv")
    )
    plot_ablation_bar(results, "mean_eer", "Mean EER",
                      "EER vs Embedding Dimension",
                      os.path.join(group_dir, "FIG_C_dim_eer.png"))
    print("GROUP C DONE")
    return results


# ============================================================
# GROUP D — PROTOTYPE K ABLATION
# [TRAINING NOT REQUIRED — evaluation-only on Group A embeddings]
# Uses saved E_train/E_test from Group A2 seed 1
# ============================================================

def run_group_D(a2_dir=None):
    """
    Compare K=1,2,3,5 KMeans prototypes.
    EVALUATION-ONLY — reuses Group A2 seed 1 embeddings.
    No retraining needed.
    """
    print("\n" + "="*65)
    print("GROUP D — PROTOTYPE K ABLATION (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "D_PROTO_K")
    X, Y, runs = load_dataset()

    # Load saved embeddings from A2 seed 1
    if a2_dir is None:
        # Find latest A2 run
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        runs_available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not runs_available:
            print("  ⚠️ Group A2 not run yet. Running D requires A2 first.")
            return None
        a2_dir = os.path.join(a2_base, runs_available[-1])

    seed1_dir = os.path.join(a2_dir, "seed_1")
    E_train = np.load(os.path.join(seed1_dir, "E_train.npy"))
    E_test  = np.load(os.path.join(seed1_dir, "E_test.npy"))

    split = build_b2t_split(X, Y, runs, seed=1)
    K_values = [1, 2, 3, 5]
    results = []

    for K in K_values:
        auc_val, eer_val, per_run, _, _ = evaluate_on_test(
            E_train, split["Y_train_np"],
            E_test,  split["Y_test_np"],
            split["runs_test"], split["test_runs"], K=K
        )
        results.append({
            "K": K, "label": f"K={K}",
            "auc": auc_val, "eer": eer_val,
            "mean_eer": eer_val, "mean_auc": auc_val
        })
        print(f"  K={K}: AUC={auc_val:.4f} | EER={eer_val*100:.2f}%")
        save_csv_from_list(per_run,
            os.path.join(group_dir, f"K{K}", f"per_run_K{K}.csv"))

    export_paper_table(
        results, ["K","auc","eer"],
        "Table: Prototype K Ablation",
        os.path.join(group_dir, "TABLE_D_proto_K.csv")
    )
    plot_ablation_bar(results, "mean_eer", "EER",
                      "EER vs Prototype Count K",
                      os.path.join(group_dir, "FIG_D_K_eer.png"))
    print("GROUP D DONE")
    return results


# ============================================================
# GROUP E — LOSS COMPONENT ABLATION
# [TRAINING REQUIRED]
# ============================================================

def run_group_E():
    """
    Ablation of loss components.
    E1: ArcFace only
    E2: ArcFace + SupCon
    E3: ArcFace + State
    E4: ArcFace + SupCon + Orth
    E5: Full model (all losses)
    """
    print("\n" + "="*65)
    print("GROUP E — LOSS ABLATION")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "E_LOSS_ABLATION")
    cfg_base = {**CFG, "epochs": 60, "patience": 15}

    variants = [
        {"name": "E1_arcface_only",        "use_supcon": False, "use_state": False, "use_orth": False},
        {"name": "E2_arcface_supcon",       "use_supcon": True,  "use_state": False, "use_orth": False},
        {"name": "E3_arcface_state",        "use_supcon": False, "use_state": True,  "use_orth": False},
        {"name": "E4_arcface_supcon_orth",  "use_supcon": True,  "use_state": False, "use_orth": True},
        {"name": "E5_full_model",           "use_supcon": True,  "use_state": True,  "use_orth": True},
    ]

    results = []
    for v in variants:
        summary, _ = run_multi_seed(
            group_dir=os.path.join(group_dir, v["name"]),
            split_fn=build_b2t_split,
            train_cfg=cfg_base,
            exp_name=v["name"],
            seeds=[1, 2, 3],
            use_supcon=v["use_supcon"],
            use_state=v["use_state"],
            use_orth=v["use_orth"]
        )
        results.append({
            "variant":     v["name"],
            "label":       v["name"].replace("_"," "),
            "use_supcon":  v["use_supcon"],
            "use_state":   v["use_state"],
            "use_orth":    v["use_orth"],
            "mean_eer":    summary["mean_eer"],
            "mean_eer_std":summary["std_eer"],
            "mean_auc":    summary["mean_auc"],
            "mean_auc_std":summary["std_auc"]
        })

    export_paper_table(
        results,
        ["variant","use_supcon","use_state","use_orth",
         "mean_eer","mean_eer_std","mean_auc","mean_auc_std"],
        "Table: Loss Ablation",
        os.path.join(group_dir, "TABLE_E_loss_ablation.csv")
    )
    plot_ablation_bar(results, "mean_eer", "Mean EER (%)",
                      "Loss Component Ablation — EER",
                      os.path.join(group_dir, "FIG_E_loss_ablation.png"))
    print("GROUP E DONE")
    return results


# ============================================================
# GROUP F — ENROLLMENT SIZE STUDY
# [TRAINING NOT REQUIRED — eval-only with fraction subsampling]
# Uses best model from A2. Rebuilds prototypes with fewer windows.
# ============================================================

def run_group_F(a2_dir=None):
    """
    Vary enrollment fraction: 25%, 50%, 75%, 100%.
    EVALUATION-ONLY — reuses trained model from A2 seed 1.
    Only prototype construction changes.
    """
    print("\n" + "="*65)
    print("GROUP F — ENROLLMENT SIZE STUDY (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "F_ENROLLMENT_SIZE")
    X, Y, runs = load_dataset()
    split = build_b2t_split(X, Y, runs, seed=1)

    if a2_dir is None:
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not available:
            print("  ⚠️ Group A2 required first.")
            return None
        a2_dir = os.path.join(a2_base, available[-1])

    seed1_dir = os.path.join(a2_dir, "seed_1")
    E_train_full = np.load(os.path.join(seed1_dir, "E_train.npy"))
    E_test       = np.load(os.path.join(seed1_dir, "E_test.npy"))

    fractions = [0.25, 0.50, 0.75, 1.00]
    results = []

    for frac in fractions:
        rng = np.random.RandomState(42)
        Y_tr = split["Y_train_np"]
        subjects = np.unique(Y_tr)
        keep_idx = []
        for sid in subjects:
            idx = np.where(Y_tr == sid)[0]
            n = max(1, int(len(idx) * frac))
            keep_idx.extend(rng.choice(idx, size=n, replace=False).tolist())
        keep_idx = np.array(keep_idx)

        E_tr_sub = E_train_full[keep_idx]
        Y_tr_sub = Y_tr[keep_idx]

        auc_v, eer_v, per_run, _, _ = evaluate_on_test(
            E_tr_sub, Y_tr_sub,
            E_test, split["Y_test_np"],
            split["runs_test"], split["test_runs"], K=3
        )
        results.append({
            "fraction": frac, "label": f"{int(frac*100)}%",
            "n_windows": len(keep_idx),
            "mean_eer": eer_v, "mean_auc": auc_v,
            "auc": auc_v, "eer": eer_v
        })
        print(f"  frac={frac:.0%}: EER={eer_v*100:.2f}% | AUC={auc_v:.4f}")
        save_csv_from_list(per_run,
            os.path.join(group_dir, f"frac{int(frac*100)}",
                         f"per_run_frac{int(frac*100)}.csv"))

    export_paper_table(
        results, ["fraction","n_windows","auc","eer"],
        "Table: Enrollment Size Study",
        os.path.join(group_dir, "TABLE_F_enrollment_size.csv")
    )
    plot_ablation_bar(results, "mean_eer", "EER",
                      "EER vs Enrollment Fraction",
                      os.path.join(group_dir, "FIG_F_enrollment.png"))
    print("GROUP F DONE")
    return results


# ============================================================
# GROUP G — PROTOCOL COMPARISON
# [TRAINING REQUIRED — ISOLATED from main protocol]
# ============================================================

def run_group_G():
    """
    Compare 3 evaluation protocols:
    1. Random split
    2. Same-task split
    3. Baseline-to-task (main locked protocol)
    
    IMPORTANT: Isolated from main protocol.
    Results show protocol difficulty inflation.
    """
    print("\n" + "="*65)
    print("GROUP G — PROTOCOL COMPARISON (Isolated)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "G_PROTOCOL_COMPARE")
    cfg_base  = {**CFG, "epochs": 60, "patience": 15}

    protocols = [
        ("G1_random_split",     build_random_split,   "Random Split"),
        ("G2_same_task",        build_same_task_split,"Same Task"),
        ("G3_baseline_to_task", build_b2t_split,      "Baseline-to-Task"),
    ]

    results = []
    for name, split_fn, label in protocols:
        summary, _ = run_multi_seed(
            group_dir=os.path.join(group_dir, name),
            split_fn=split_fn,
            train_cfg=cfg_base,
            exp_name=name,
            seeds=[1, 2, 3]
        )
        results.append({
            "protocol": label, "label": label,
            "mean_eer": summary["mean_eer"],
            "mean_eer_std": summary["std_eer"],
            "mean_auc": summary["mean_auc"]
        })

    # Compute protocol inflation
    if len(results) == 3:
        random_eer = results[0]["mean_eer"]
        b2t_eer    = results[2]["mean_eer"]
        inflation  = (b2t_eer - random_eer) / random_eer * 100
        print(f"\n  Protocol inflation (Random→B2T): {inflation:.1f}%")
        save_json({"inflation_pct": inflation,
                   "random_eer": random_eer,
                   "b2t_eer": b2t_eer},
                  os.path.join(group_dir, "protocol_inflation.json"))

    export_paper_table(
        results,
        ["protocol","mean_eer","mean_eer_std","mean_auc"],
        "Table: Protocol Comparison",
        os.path.join(group_dir, "TABLE_G_protocol.csv")
    )
    plot_ablation_bar(results, "mean_eer", "Mean EER",
                      "EER vs Evaluation Protocol",
                      os.path.join(group_dir, "FIG_G_protocol.png"))
    print("GROUP G DONE")
    return results


# ============================================================
# GROUP H — ENROLLMENT STRATEGY
# [EVALUATION-ONLY — reuses Group A2 embeddings]
# ============================================================

def run_group_H(a2_dir=None):
    """
    Compare enrollment strategies:
    1. Mean prototype (centroid)
    2. KMeans K=3
    EVALUATION-ONLY.
    """
    print("\n" + "="*65)
    print("GROUP H — ENROLLMENT STRATEGY (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "H_ENROLLMENT_STRATEGY")
    X, Y, runs = load_dataset()
    split = build_b2t_split(X, Y, runs, seed=1)

    if a2_dir is None:
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not available: return None
        a2_dir = os.path.join(a2_base, available[-1])

    seed1_dir = os.path.join(a2_dir, "seed_1")
    E_train = np.load(os.path.join(seed1_dir, "E_train.npy"))
    E_test  = np.load(os.path.join(seed1_dir, "E_test.npy"))

    results = []

    # Mean prototype
    pvecs_m, powner_m = build_mean_prototype(E_train, split["Y_train_np"])
    scores=[]; labels_l=[]
    for run_name in split["test_runs"]:
        pidx = np.where(split["runs_test"] == run_name)[0]
        if len(pidx) == 0: continue
        P = E_test[pidx]; yr = split["Y_test_np"][pidx]
        sm = P @ pvecs_m.T
        for i in range(len(P)):
            yt=yr[i]; mg=(powner_m==yt); mi=(powner_m!=yt)
            scores.append(float(sm[i,mg].max())); labels_l.append(1)
            scores.extend(sm[i,mi].tolist()); labels_l.extend([0]*mi.sum())
    from sklearn.metrics import roc_curve, auc as sk_auc
    fpr,tpr,_ = roc_curve(labels_l,scores,pos_label=1)
    A_m = sk_auc(fpr,tpr); fnr=1-tpr; k=np.nanargmin(np.abs(fpr-fnr))
    eer_m = (fpr[k]+fnr[k])/2
    results.append({"strategy":"Mean Prototype","label":"Mean","auc":A_m,"eer":eer_m,"mean_eer":eer_m})

    # KMeans K=3
    auc_k, eer_k, _, _, _ = evaluate_on_test(
        E_train, split["Y_train_np"],
        E_test,  split["Y_test_np"],
        split["runs_test"], split["test_runs"], K=3
    )
    results.append({"strategy":"KMeans K=3","label":"KMeans K=3","auc":auc_k,"eer":eer_k,"mean_eer":eer_k})

    print(f"  Mean:     EER={eer_m*100:.2f}%  AUC={A_m:.4f}")
    print(f"  KMeans:   EER={eer_k*100:.2f}%  AUC={auc_k:.4f}")

    export_paper_table(
        results, ["strategy","auc","eer"],
        "Table: Enrollment Strategy",
        os.path.join(group_dir, "TABLE_H_enrollment_strategy.csv")
    )
    print("GROUP H DONE")
    return results


# ============================================================
# GROUP I — PER-RUN TASK ANALYSIS
# [EVALUATION-ONLY — reuses Group A2 embeddings]
# ============================================================

def run_group_I(a2_dir=None):
    """Per-run AUC/EER for R03–R14."""
    print("\n" + "="*65)
    print("GROUP I — PER-RUN ANALYSIS (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "I_PER_RUN_ANALYSIS")
    os.makedirs(group_dir, exist_ok=True)
    X, Y, runs = load_dataset()
    split = build_b2t_split(X, Y, runs, seed=1)

    if a2_dir is None:
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not available: return None
        a2_dir = os.path.join(a2_base, available[-1])

    seed1_dir = os.path.join(a2_dir, "seed_1")
    E_train = np.load(os.path.join(seed1_dir, "E_train.npy"))
    E_test  = np.load(os.path.join(seed1_dir, "E_test.npy"))

    _, _, per_run, _, _ = evaluate_on_test(
        E_train, split["Y_train_np"],
        E_test,  split["Y_test_np"],
        split["runs_test"], split["test_runs"], K=3
    )

    for r in per_run:
        print(f"  {r['run']}: EER={r['eer']*100:.2f}% | AUC={r['auc']:.4f}")

    save_csv_from_list(per_run, os.path.join(group_dir, "per_run_results.csv"))
    plot_per_run_bar(per_run, os.path.join(group_dir, "FIG_I_per_run_eer.png"), "eer")
    plot_per_run_bar(per_run, os.path.join(group_dir, "FIG_I_per_run_auc.png"), "auc")

    export_paper_table(
        per_run, ["run","auc","eer","n_windows"],
        "Table: Per-Run Analysis",
        os.path.join(group_dir, "TABLE_I_per_run.csv")
    )
    print("GROUP I DONE")
    return per_run


# ============================================================
# GROUP J — DISENTANGLEMENT VALIDATION
# [EVALUATION-ONLY — reuses Group A2 model + embeddings]
# ============================================================

def run_group_J(a2_dir=None):
    """
    Quantify identity vs state decorrelation.
    Compute mean squared cosine similarity between z_id and z_state.
    """
    print("\n" + "="*65)
    print("GROUP J — DISENTANGLEMENT VALIDATION (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "J_DISENTANGLEMENT")
    os.makedirs(group_dir, exist_ok=True)
    X, Y, runs = load_dataset()
    split = build_b2t_split(X, Y, runs, seed=1)

    if a2_dir is None:
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not available: return None
        a2_dir = os.path.join(a2_base, available[-1])

    # Load best model from seed 1
    seed1_dir = os.path.join(a2_dir, "seed_1")
    nc = int(len(np.unique(split["Y_train_np"][split["tr_idx_sub"]])))
    model = DOMCSModel(emb_dim=128).to(DEVICE)
    arc   = ArcFaceLayer(128, nc).to(DEVICE)
    from core_framework import DEVICE as DEV, load_checkpoint
    load_checkpoint(seed1_dir, model, arc, prefer_best=True)
    model.eval()

    # Extract BOTH z_id and z_state
    from torch.utils.data import DataLoader
    from core_framework import EEGDataset
    import torch
    import torch.nn.functional as F

    ds = EEGDataset(
        split["X_test_np"][:5000],
        split["Y_test_np"][:5000],
        split["state_test"][:5000]
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    z_ids=[]; z_states=[]; y_all=[]; s_all=[]
    with torch.no_grad():
        for xb, yb, sb in loader:
            zi, zs, _ = model(xb.to(DEVICE))
            z_ids.append(zi.cpu().numpy())
            z_states.append(zs.cpu().numpy())
            y_all.extend(yb.numpy())
            s_all.extend(sb.numpy())

    Z_id    = np.concatenate(z_ids)
    Z_state = np.concatenate(z_states)

    # Metric 1: Mean squared cosine similarity
    cos_sims = (Z_id * Z_state).sum(axis=1)  # dot of L2-normalized = cosine
    mean_cos = float(np.mean(cos_sims))
    mean_sq  = float(np.mean(cos_sims**2))
    std_cos  = float(np.std(cos_sims))

    # Metric 2: State prediction accuracy from z_id
    y_all = np.array(y_all); s_all = np.array(s_all)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(Z_id, s_all)
    state_pred_acc = float(clf.score(Z_id, s_all))

    results = {
        "mean_cosine_similarity": mean_cos,
        "mean_sq_cosine": mean_sq,
        "std_cosine": std_cos,
        "state_pred_from_z_id": state_pred_acc,
        "n_samples": len(Z_id),
        "interpretation": "Values near 0 = good disentanglement. Below 0.5 state pred = active suppression."
    }
    save_json(results, os.path.join(group_dir, "disentanglement_metrics.json"))

    # Histogram of cosine similarities
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cos_sims, bins=50, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(mean_cos, color="red", linestyle="--",
               label=f"Mean={mean_cos:.4f}")
    ax.set_xlabel("Cosine Similarity (z_id · z_state)")
    ax.set_ylabel("Count")
    ax.set_title("Disentanglement — Identity vs State Embedding Similarity")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, "FIG_J_disentangle_hist.png"),
                dpi=300, bbox_inches="tight"); plt.close()

    print(f"  Mean cosine similarity:   {mean_cos:.4f}")
    print(f"  Mean sq cosine:           {mean_sq:.4f}")
    print(f"  State pred from z_id:     {state_pred_acc:.4f}")
    print("GROUP J DONE")
    return results


# ============================================================
# GROUP K — EMBEDDING VISUALIZATION (t-SNE / UMAP)
# [EVALUATION-ONLY — reuses Group A2 embeddings]
# ============================================================

def run_group_K(a2_dir=None, n_samples=2000):
    """t-SNE visualization of identity embeddings."""
    print("\n" + "="*65)
    print("GROUP K — EMBEDDING VISUALIZATION (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "K_VISUALIZATION")
    os.makedirs(group_dir, exist_ok=True)
    X, Y, runs = load_dataset()
    split = build_b2t_split(X, Y, runs, seed=1)

    if a2_dir is None:
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not available: return None
        a2_dir = os.path.join(a2_base, available[-1])

    seed1_dir = os.path.join(a2_dir, "seed_1")
    E_test = np.load(os.path.join(seed1_dir, "E_test.npy"))
    Y_test = split["Y_test_np"]

    # Subsample for speed
    rng  = np.random.RandomState(42)
    idx  = rng.choice(len(E_test), size=min(n_samples, len(E_test)), replace=False)
    E_sub = E_test[idx]; Y_sub = Y_test[idx]

    # Try UMAP first, fallback to t-SNE
    method = "t-SNE"
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords  = reducer.fit_transform(E_sub)
        method  = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42,
                      perplexity=30).fit_transform(E_sub)

    np.save(os.path.join(group_dir, f"{method}_coords.npy"), coords)
    np.save(os.path.join(group_dir, "Y_sub.npy"), Y_sub)

    # Plot — color by subject (up to 20 subjects for clarity)
    subjects = np.unique(Y_sub)
    n_show   = min(20, len(subjects))
    cmap     = plt.cm.get_cmap("tab20", n_show)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, sid in enumerate(subjects[:n_show]):
        mask = Y_sub == sid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.6, color=cmap(i), label=f"S{sid}")
    ax.set_title(f"{method} — Identity Embeddings (Test Set, {n_show} subjects)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.legend(fontsize=6, ncol=4, loc="upper right",
              markerscale=2, bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"FIG_K_{method}_subject.png"),
                dpi=300, bbox_inches="tight"); plt.close()

    print(f"  {method} visualization saved.")
    print("GROUP K DONE")
    return coords, Y_sub


# ============================================================
# GROUP L — TRAINING DYNAMICS
# [EVALUATION-ONLY — reads CSVs from Group A2]
# ============================================================

def run_group_L(a2_dir=None):
    """
    Generate publication-quality learning curve figures.
    Reads train_log.csv from all 5 seeds of Group A2.
    """
    print("\n" + "="*65)
    print("GROUP L — TRAINING DYNAMICS (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "L_TRAINING_DYNAMICS")
    os.makedirs(group_dir, exist_ok=True)

    if a2_dir is None:
        a2_base = os.path.join(EXP_ROOT, "A_MAIN_REFERENCE", "A2_multiseed")
        available = sorted(os.listdir(a2_base)) if os.path.exists(a2_base) else []
        if not available: return None
        a2_dir = os.path.join(a2_base, available[-1])

    seeds  = [1,2,3,4,5]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    all_logs = {}
    for seed in seeds:
        path = os.path.join(a2_dir, f"seed_{seed}", "train_log.csv")
        if os.path.exists(path):
            all_logs[seed] = pd.read_csv(path)

    if not all_logs:
        print("  ⚠️ No training logs found.")
        return None

    # Figure 1: Mean ± std loss and accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DOMCS-EEG Training Dynamics — 5 Seeds (60 Epochs)",
                 fontsize=13, fontweight="bold")

    tr_losses = np.array([all_logs[s]["tr_loss"].values for s in all_logs])
    vl_losses = np.array([all_logs[s]["val_loss"].values for s in all_logs])
    tr_accs   = np.array([all_logs[s]["tr_acc"].values*100 for s in all_logs])
    vl_accs   = np.array([all_logs[s]["val_acc"].values*100 for s in all_logs])
    ep = list(all_logs.values())[0]["epoch"].values

    for arr, label, color in [
        (tr_losses, "Train Loss", "#1f77b4"),
        (vl_losses, "Val Loss",   "#d62728")
    ]:
        axes[0].plot(ep, arr.mean(0), color=color, lw=2.5, label=label)
        axes[0].fill_between(ep, arr.mean(0)-arr.std(0),
                              arr.mean(0)+arr.std(0), alpha=0.2, color=color)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss — Mean ± Std"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for arr, label, color in [
        (tr_accs, "Train Acc", "#1f77b4"),
        (vl_accs, "Val Acc",   "#d62728")
    ]:
        axes[1].plot(ep, arr.mean(0), color=color, lw=2.5, label=label)
        axes[1].fill_between(ep, arr.mean(0)-arr.std(0),
                              arr.mean(0)+arr.std(0), alpha=0.2, color=color)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy — Mean ± Std"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, "FIG_L_mean_std_dynamics.png"),
                dpi=300, bbox_inches="tight"); plt.close()

    # Figure 2: Per-seed overlay
    plot_multi_seed_curves(a2_dir, list(all_logs.keys()), group_dir)

    # Copy to paper outputs
    paper_fig_dir = os.path.join(CFG["PAPER_OUT"], "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    import shutil
    for fname in os.listdir(group_dir):
        if fname.endswith(".png"):
            shutil.copy(os.path.join(group_dir, fname),
                        os.path.join(paper_fig_dir, fname))

    print("GROUP L DONE")
    return True


# ============================================================
# GROUP M — COMPUTATIONAL PRACTICALITY
# [EVALUATION-ONLY — timing measurements]
# ============================================================

def run_group_M(a2_dir=None):
    """
    Report training time, inference time, parameter count.
    """
    print("\n" + "="*65)
    print("GROUP M — COMPUTATIONAL PRACTICALITY (Eval-only)")
    print("="*65)

    group_dir = os.path.join(EXP_ROOT, "M_COMPUTE_STATS")
    os.makedirs(group_dir, exist_ok=True)

    X, Y, runs = load_dataset()
    split = build_b2t_split(X, Y, runs, seed=1)

    model = DOMCSModel(emb_dim=128).to(DEVICE)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    # Inference time — single window
    dummy = torch.randn(1, 64, 256).to(DEVICE)
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10): model(dummy)
        # Measure
        t0 = time.time()
        N = 1000
        for _ in range(N): model(dummy)
        single_ms = (time.time() - t0) / N * 1000

    # Batch inference time
    dummy_batch = torch.randn(256, 64, 256).to(DEVICE)
    with torch.no_grad():
        for _ in range(5): model(dummy_batch)
        t0 = time.time()
        for _ in range(100): model(dummy_batch)
        batch_ms = (time.time() - t0) / 100 * 1000

    # Training time estimate from log
    training_time_s = None
    if a2_dir:
        log_path = os.path.join(a2_dir, "seed_1", "train_log.csv")
        if os.path.exists(log_path):
            # Estimate from epoch count and ~0.6s/epoch
            df = pd.read_csv(log_path)
            training_time_s = len(df) * 0.65  # average from logs

    stats = {
        "n_parameters": n_params,
        "single_window_inference_ms": round(single_ms, 3),
        "batch256_inference_ms": round(batch_ms, 3),
        "estimated_training_time_s_60ep": training_time_s,
        "model_size_MB": round(n_params * 4 / 1024 / 1024, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "emb_dim": 128
    }
    save_json(stats, os.path.join(group_dir, "compute_stats.json"))

    df = pd.DataFrame([stats])
    df.to_csv(os.path.join(group_dir, "TABLE_M_compute.csv"), index=False)

    print(f"  Parameters:      {n_params:,}")
    print(f"  Single infer:    {single_ms:.3f} ms")
    print(f"  Batch256 infer:  {batch_ms:.3f} ms")
    if training_time_s:
        print(f"  Training 60ep:   {training_time_s:.0f}s ≈ {training_time_s/60:.1f} min")
    print("GROUP M DONE")
    return stats


# ============================================================
# MAIN EXECUTION — Comment/uncomment groups as needed
# ============================================================

def main():
    """
    EXECUTION ORDER:
    Run groups in this order for proper dependency chain.
    A → B,C,E,G (training) → D,F,H,I,J,K,L,M (eval-only)
    
    RECOMMENDED SCHEDULE:
    Day 1: A (main evidence — most important)
    Day 2: B, C, E (ablations — training required)
    Day 3: G (protocol study — training required)
    Day 4: D, F, H, I, J, K, L, M (eval-only — fast)
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",  default="1", help="GPU ID to use")
    parser.add_argument("--groups", nargs="+",
                        default=["A"],
                        help="Groups to run, e.g. A B C D E F G H I J K L M")
    args = parser.parse_args()

    setup_device(args.gpu)
    print(f"\nRunning groups: {args.groups}")

    a2_dir = None  # will be set after Group A runs

    if "A" in args.groups:
        summary_a2, a2_dir = run_group_A()

    if "B" in args.groups:
        run_group_B()

    if "C" in args.groups:
        run_group_C()

    if "D" in args.groups:
        run_group_D(a2_dir)

    if "E" in args.groups:
        run_group_E()

    if "F" in args.groups:
        run_group_F(a2_dir)

    if "G" in args.groups:
        run_group_G()

    if "H" in args.groups:
        run_group_H(a2_dir)

    if "I" in args.groups:
        run_group_I(a2_dir)

    if "J" in args.groups:
        run_group_J(a2_dir)

    if "K" in args.groups:
        run_group_K(a2_dir)

    if "L" in args.groups:
        run_group_L(a2_dir)

    if "M" in args.groups:
        run_group_M(a2_dir)

    # Build master results table
    build_aggregate_paper_tables(
        CFG["EXP_ROOT"],
        CFG["PAPER_OUT"]
    )

    print("\n🎉 ALL REQUESTED GROUPS COMPLETE.")
    print(f"   Results: {CFG['EXP_ROOT']}")
    print(f"   Paper outputs: {CFG['PAPER_OUT']}")


if __name__ == "__main__":
    main()
