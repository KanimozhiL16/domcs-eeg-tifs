from __future__ import annotations

import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "TABLE_IV_main_results.csv"
CONFIG = ROOT / "experiments" / "configs" / "main_60ep.yaml"

EXPECTED_MEAN = {
    "eer_pct": 3.75,
    "auc": 0.9928,
    "crr_pct": 86.95,
}

EXPECTED_STD = {
    "eer_pct": 0.20,
    "auc": 0.0008,
    "crr_pct": 0.40,
}

EXPECTED_CONFIG_PATTERNS = {
    "protocol": r"type:\s*b2t",
    "train_runs": r"train_runs:\s*\[R01,\s*R02\]",
    "test_runs": r"test_runs:\s*\[R03,\s*R04,\s*R05,\s*R06,\s*R07,\s*R08,\s*R09,\s*R10,\s*R11,\s*R12,\s*R13,\s*R14\]",
    "enrollment_k": r"enrollment_k:\s*3",
    "seeds": r"seeds:\s*\[1,\s*2,\s*3,\s*4,\s*5\]",
    "dataset": r"name:\s*EEGMMIDB",
    "subjects": r"n_subjects:\s*109",
    "epochs": r"epochs:\s*60",
    "batch_size": r"batch_size:\s*256",
    "learning_rate": r"lr:\s*0\.0003",
    "weight_decay": r"weight_decay:\s*0\.0001",
    "supcon_weight": r"supcon:[\s\S]*?weight:\s*0\.5",
    "state_weight": r"state:[\s\S]*?weight:\s*0\.5",
    "orth_weight": r"orth:[\s\S]*?weight:\s*0\.1",
}


def _close(actual: float, expected: float, tol: float = 1e-8) -> bool:
    return math.isclose(actual, expected, rel_tol=tol, abs_tol=tol)


def read_summary_rows() -> dict[str, dict[str, float]]:
    with RESULTS.open(newline="") as f:
        rows = list(csv.DictReader(f))

    summary = {}
    for row in rows:
        seed = row["seed"].strip().lower()
        if seed in {"mean", "std"}:
            summary[seed] = {
                "eer_pct": float(row["eer_pct"]),
                "auc": float(row["auc"]),
                "crr_pct": float(row["crr_pct"]),
            }
    return summary


def verify_results() -> None:
    summary = read_summary_rows()
    if "mean" not in summary or "std" not in summary:
        raise AssertionError("TABLE_IV_main_results.csv must contain mean and std rows.")

    for metric, expected in EXPECTED_MEAN.items():
        actual = summary["mean"][metric]
        if not _close(actual, expected):
            raise AssertionError(f"Mean {metric} drifted: got {actual}, expected {expected}")

    for metric, expected in EXPECTED_STD.items():
        actual = summary["std"][metric]
        if not _close(actual, expected):
            raise AssertionError(f"Std {metric} drifted: got {actual}, expected {expected}")


def verify_protocol_and_config() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    for key, pattern in EXPECTED_CONFIG_PATTERNS.items():
        if not re.search(pattern, text):
            raise AssertionError(f"Protocol/config field drifted or missing: {key}")


def main() -> None:
    verify_results()
    verify_protocol_and_config()
    print("Paper table alignment verified for v11: EER, AUC, CRR, B2T protocol, seeds, and training hyperparameters match.")
    print("For raw NVIDIA evidence and rerun tolerance checks, run: python scripts/verify_reproducibility.py")


if __name__ == "__main__":
    main()
