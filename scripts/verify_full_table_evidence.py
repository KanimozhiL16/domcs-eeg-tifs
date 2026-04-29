from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "experiments/results/TABLE_IV_main_results.csv",
    "experiments/results/PAPER_RESULTS_LOCK.csv",
    "paper_evidence/master/MASTER_RESULTS_TABLE.csv",
    "paper_evidence/baselines/comparison_table.csv",
    "paper_evidence/security/TIFS_security_summary.csv",
    "paper_evidence/security/forgery_results.csv",
    "paper_evidence/security/attack_noise.csv",
    "paper_evidence/security/attack_leakage.csv",
    "paper_evidence/security/attack_mimicry.csv",
    "paper_evidence/security/attack_gap.csv",
    "paper_evidence/security/attack_whitebox_corrected.csv",
    "paper_evidence/security/attack_whitebox_embedding.csv",
    "paper_evidence/security/attack_summary_merged.csv",
    "paper_evidence/security/targeted_false_accept_summary.csv",
    "paper_evidence/interpretability/TABLE_INTERPRETABILITY.csv",
    "paper_evidence/interpretability/TABLE_IV_representation_summary_paper_values.csv",
    "paper_evidence/protocol/TABLE_V_protocol_comparison_paper_values.csv",
    "paper_evidence/bed/TABLE_VIII_bed_validation_paper_values.csv",
    "paper_evidence/supplementary/SUPPLEMENTARY_VALUES_LOCK.csv",
    "paper_evidence/audit/FULL_TABLE_AUDIT.md",
]


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require_file(path: str) -> None:
    if not (ROOT / path).is_file():
        raise AssertionError(f"Missing evidence file: {path}")


def require_close(actual: float, expected: float, tol: float, label: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise AssertionError(f"Missing row where {key}={value}")


def main() -> None:
    for path in REQUIRED_FILES:
        require_file(path)

    main_rows = read_csv("experiments/results/TABLE_IV_main_results.csv")
    mean_row = row_by(main_rows, "seed", "mean")
    std_row = row_by(main_rows, "seed", "std")
    require_close(float(mean_row["eer_pct"]), 3.76, 1e-9, "Main mean EER")
    require_close(float(mean_row["auc"]), 0.9931, 1e-9, "Main mean AUC")
    require_close(float(mean_row["crr_pct"]), 86.95, 1e-9, "Main mean CRR")
    require_close(float(std_row["eer_pct"]), 0.21, 1e-9, "Main EER std")
    require_close(float(std_row["auc"]), 0.0007, 1e-9, "Main AUC std")
    require_close(float(std_row["crr_pct"]), 0.40, 1e-9, "Main CRR std")

    forgery = read_csv("paper_evidence/security/forgery_results.csv")
    domcs = row_by(forgery, "Method", "DOMCS-EEG (ArcFace + SupCon + L_orth)")
    require_close(float(domcs["Random_EER"]), 4.17, 1e-9, "Forgery DOMCS random EER")
    require_close(float(domcs["Random_AUC"]), 0.9918, 1e-9, "Forgery DOMCS random AUC")
    require_close(float(domcs["Skilled_EER"]), 3.41, 1e-9, "Forgery DOMCS skilled EER")
    require_close(float(domcs["Skilled_AUC"]), 0.9938, 1e-9, "Forgery DOMCS skilled AUC")

    attacks = read_csv("paper_evidence/security/attack_summary_merged.csv")
    clean = row_by(attacks, "attack", "CLEAN")
    require_close(float(clean["auc"]), 0.9946964208827945, 1e-12, "Clean attack AUC")
    require_close(float(clean["eer"]), 0.033318136682425364, 1e-12, "Clean attack EER")

    targeted = read_csv("paper_evidence/security/targeted_false_accept_summary.csv")
    targeted_row = row_by(targeted, "attack", "TARGETED_FALSE_ACCEPT")
    require_close(
        float(targeted_row["targeted_success_rate"]),
        0.12774305555555557,
        1e-12,
        "Targeted false accept success rate",
    )

    protocol = read_csv("paper_evidence/protocol/TABLE_V_protocol_comparison_paper_values.csv")
    b2t = row_by(protocol, "protocol", "Baseline-to-task")
    require_close(float(b2t["eer_pct"]), 3.76, 1e-9, "Protocol B2T EER")
    require_close(float(b2t["auc"]), 0.9931, 1e-9, "Protocol B2T AUC")

    bed = read_csv("paper_evidence/bed/TABLE_VIII_bed_validation_paper_values.csv")
    bed_domcs = row_by(bed, "model", "DOMCS-EEG")
    require_close(float(bed_domcs["eer_pct"]), 22.93, 1e-9, "BED DOMCS EER")
    require_close(float(bed_domcs["auc"]), 0.846, 1e-9, "BED DOMCS AUC")

    print("Full table evidence verified.")
    print("Main: EER=3.76+-0.21, AUC=0.9931+-0.0007, CRR=86.95+-0.40")
    print("Security, protocol, interpretability, BED, and supplementary evidence files are present.")


if __name__ == "__main__":
    main()
