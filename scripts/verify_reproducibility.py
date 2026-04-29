from __future__ import annotations

import csv
import math
import statistics as stats
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLE = ROOT / "experiments" / "results" / "TABLE_III_main_results.csv"
ORIGINAL_RAW = ROOT / "results" / "main_results" / "original_20260406_multi_seed_summary.csv"
RERUN_RAW = ROOT / "results" / "main_results" / "rerun_20260428_multi_seed_summary.csv"
CONFIG = ROOT / "experiments" / "configs" / "main_60ep.yaml"

EXPECTED_PAPER = {
    "eer_pct_mean": 3.76,
    "eer_pct_std": 0.21,
    "auc_mean": 0.9931,
    "auc_std": 0.0007,
    "crr_pct_mean": 86.95,
    "crr_pct_std": 0.40,
}

EXPECTED_CONFIG_SNIPPETS = [
    "lr: 0.0003",
    "weight_decay: 0.0001",
    "weight: 0.5",
    "weight: 0.1",
    "train_runs: [R01, R02]",
    "enrollment_k: 3",
]


def _read_paper_table(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(newline="")))
    mean = next(r for r in rows if r["seed"].strip().lower() == "mean")
    std = next(r for r in rows if r["seed"].strip().lower() == "std")
    return {
        "eer_pct_mean": float(mean["eer_pct"]),
        "eer_pct_std": float(std["eer_pct"]),
        "auc_mean": float(mean["auc"]),
        "auc_std": float(std["auc"]),
        "crr_pct_mean": float(mean["crr_pct"]),
        "crr_pct_std": float(std["crr_pct"]),
    }


def _read_raw_summary(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(newline="")))
    aucs = [float(r["avg_auc"]) for r in rows]
    eers_pct = [float(r["avg_eer"]) * 100.0 for r in rows]
    return {
        "eer_pct_mean": stats.mean(eers_pct),
        "eer_pct_std": stats.stdev(eers_pct),
        "auc_mean": stats.mean(aucs),
        "auc_std": stats.stdev(aucs),
    }


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _print_row(name: str, d: dict[str, float]) -> None:
    print(
        f"{name:32s} "
        f"EER={d['eer_pct_mean']:.2f} +/- {d['eer_pct_std']:.2f}%  "
        f"AUC={d['auc_mean']:.4f} +/- {d['auc_std']:.4f}"
    )


def verify_paper_lock() -> None:
    paper = _read_paper_table(PAPER_TABLE)
    for key, expected in EXPECTED_PAPER.items():
        tolerance = 1e-8
        actual = paper[key]
        if not _close(actual, expected, tolerance):
            raise AssertionError(f"Submitted paper table drifted for {key}: got {actual}, expected {expected}")


def verify_config() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    missing = [s for s in EXPECTED_CONFIG_SNIPPETS if s not in text]
    if missing:
        raise AssertionError("Config is missing expected submitted-paper fields: " + ", ".join(missing))


def verify_raw_ranges() -> None:
    original = _read_raw_summary(ORIGINAL_RAW)
    rerun = _read_raw_summary(RERUN_RAW)

    # These are deliberately tolerance checks, not exact equality checks.
    checks = [
        ("original EER mean", original["eer_pct_mean"], 3.76, 0.05),
        ("original AUC mean", original["auc_mean"], 0.9931, 0.0002),
        ("rerun EER mean", rerun["eer_pct_mean"], 3.76, 0.35),
        ("rerun AUC mean", rerun["auc_mean"], 0.9931, 0.0015),
    ]
    for label, actual, expected, tolerance in checks:
        if not _close(actual, expected, tolerance):
            raise AssertionError(
                f"{label} outside reproducibility tolerance: got {actual}, "
                f"expected around {expected} +/- {tolerance}"
            )


def main() -> None:
    verify_paper_lock()
    verify_config()
    verify_raw_ranges()

    print("DOMCS-EEG submitted-paper reproducibility check")
    print("-" * 72)
    _print_row("Submitted paper Table III", _read_paper_table(PAPER_TABLE))
    _print_row("Original NVIDIA A100", _read_raw_summary(ORIGINAL_RAW))
    _print_row("Independent A100 rerun", _read_raw_summary(RERUN_RAW))
    print("-" * 72)
    print("OK: submitted paper table, config, original raw evidence, and rerun evidence are consistent within tolerance.")


if __name__ == "__main__":
    main()
