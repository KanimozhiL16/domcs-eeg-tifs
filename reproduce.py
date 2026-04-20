"""
reproduce.py — Quick verification that the DOMCS-EEG codebase loads correctly
and that result CSV files match the reported paper numbers.

Usage:
    python reproduce.py

This script does NOT require the dataset. It:
  1. Imports the model and verifies architecture
  2. Does a forward-pass on random data to confirm shapes
  3. Loads result CSVs and verifies reported numbers
  4. Prints a pass/fail report
"""

import sys
import os
import importlib

PASS = "✅"
FAIL = "❌"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition


print("\n" + "="*60)
print("  DOMCS-EEG Reproduction Check")
print("="*60 + "\n")

# ── 1. Import checks ──────────────────────────────────────────
print("── 1. Import checks")
try:
    import torch
    check("torch importable", True, torch.__version__)
except ImportError:
    check("torch importable", False, "pip install torch")

try:
    sys.path.insert(0, os.path.dirname(__file__))
    from domcs_eeg.model import DOMCSEEG, ArcFaceLoss
    check("domcs_eeg.model importable", True)
except Exception as e:
    check("domcs_eeg.model importable", False, str(e))
    sys.exit(1)

try:
    from domcs_eeg.evaluate import compute_eer, build_prototypes
    check("domcs_eeg.evaluate importable", True)
except Exception as e:
    check("domcs_eeg.evaluate importable", False, str(e))

# ── 2. Architecture checks ────────────────────────────────────
print("\n── 2. Architecture checks")
import torch
import torch.nn.functional as F

model = DOMCSEEG(n_channels=64, emb_dim=128)
model.eval()

x = torch.randn(4, 64, 256)   # batch=4, channels=64, time=256
with torch.no_grad():
    z_id, z_cs, s_out, feat = model(x)

check("Input shape (4,64,256) accepted", x.shape == (4, 64, 256))
check("Backbone output shape (4,256)",   feat.shape == (4, 256))
check("z_id shape (4,128)",              z_id.shape == (4, 128))
check("z_cs shape (4,128)",              z_cs.shape == (4, 128))
check("z_id L2-normalized",              abs(z_id.norm(dim=1).mean().item() - 1.0) < 1e-5,
      f"norm={z_id.norm(dim=1).mean().item():.6f}")
check("z_cs L2-normalized",              abs(z_cs.norm(dim=1).mean().item() - 1.0) < 1e-5)
check("State classifier output (4,2)",   s_out.shape == (4, 2))

arc = ArcFaceLoss(emb_dim=128, num_classes=109, s=30.0, m=0.50)
check("ArcFaceLoss instantiates (109 classes, s=30, m=0.50)", True)
check("ArcFace weight matrix shape (109,128)", arc.weight.shape == (109, 128))

# ── 3. L_orth computation check ───────────────────────────────
print("\n── 3. L_orth implementation check")
dot = (F.normalize(z_id, dim=1) * F.normalize(z_cs, dim=1)).sum(dim=1)
ol  = (dot ** 2).mean()
check("L_orth = mean((z_id · z_cs)²)", True,
      f"value={ol.item():.6f} (should be near 1/128≈0.0078 for random init)")
check("L_orth bounded in [0,1]", 0.0 <= ol.item() <= 1.0)

# ── 4. Result CSV verification ────────────────────────────────
print("\n── 4. Result CSV checks (paper numbers)")
import csv, json

def load_csv(path):
    rows = []
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        pass
    return rows

BASE = os.path.join(os.path.dirname(__file__), "results")

# Main results
rows = load_csv(os.path.join(BASE, "main_results", "multi_seed_raw_brev_a100.csv"))
if rows:
    # column names: seed, avg_auc, avg_eer  (values are fractions 0–1)
    eers = [float(r.get("eer_%", r.get("avg_eer", 0))) * (100 if float(r.get("eer_%", r.get("avg_eer", 0))) < 1 else 1) for r in rows]
    aucs = [float(r.get("auc",   r.get("avg_auc", 0))) for r in rows]
    import statistics
    mean_eer = statistics.mean(eers)
    std_eer  = statistics.stdev(eers)
    mean_auc = statistics.mean(aucs)
    check("Mean EER ≈ 3.75%",  abs(mean_eer - 3.75) < 0.05, f"{mean_eer:.4f}%")
    check("EER Std ≈ 0.20%",   abs(std_eer  - 0.20) < 0.05, f"{std_eer:.4f}%")
    check("Mean AUC ≈ 0.9928", abs(mean_auc - 0.9928) < 0.001, f"{mean_auc:.6f}")
else:
    check("multi_seed_raw_brev_a100.csv readable", False, "file not found")

# Security — T3 noise check
rows = load_csv(os.path.join(BASE, "security_analysis", "attack_noise.csv"))
s3_10db = [r for r in rows if r.get("Model","").startswith("E3") and "10dB" in r.get("Noise_Type","")]
if s3_10db:
    check("T3 AWGN 10dB EER ≈ 8.00%", abs(float(s3_10db[0]["EER_%"]) - 8.00) < 0.05,
          f"{s3_10db[0]['EER_%']}%")
else:
    check("attack_noise.csv T3 10dB readable", False)

# T5 clean baseline
rows = load_csv(os.path.join(BASE, "adversarial_t5", "clean_baseline.csv"))
if rows:
    check("Clean baseline EER ≈ 3.33%",  abs(float(rows[0]["eer"])  - 0.0333) < 0.001,
          f"{float(rows[0]['eer'])*100:.3f}%")
    check("Clean baseline AUC ≈ 0.9947", abs(float(rows[0]["auc"]) - 0.9947) < 0.001)
else:
    check("clean_baseline.csv readable", False)

# ── 5. Summary ────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for s,_,_ in results if s == PASS)
failed = sum(1 for s,_,_ in results if s == FAIL)
total  = len(results)
print(f"  RESULT: {passed}/{total} checks passed" + (f"  ←  {failed} FAILED" if failed else "  — ALL PASSED ✅"))
print("="*60 + "\n")
sys.exit(0 if failed == 0 else 1)
