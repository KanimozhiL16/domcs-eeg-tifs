import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_auth.runtime import AuthConfig, BiometricAuthenticator, DEFAULT_DATABASE_PATH


def build_authenticator(database_path: Path) -> BiometricAuthenticator:
    return BiometricAuthenticator(
        AuthConfig(
            checkpoint_path=ROOT / "checkpoints" / "seed_1" / "model_best.pt",
            registry_path=ROOT / "runtime_store" / "demo_smoke_registry.json",
            embeddings_dir=ROOT / "runtime_store" / "demo_smoke_embeddings",
            database_path=database_path,
        )
    )


def print_subject_summary(authenticator: BiometricAuthenticator) -> list[dict]:
    meta = authenticator.database.load_metadata()
    print("NPZ:", meta["path"])
    print("Keys:", ", ".join(meta["keys"]))
    print("Shape:", meta["shape"])
    print("Subjects:", meta["num_subjects"])
    print("Runs:", ", ".join(meta["runs"]))
    ready = [row for row in meta["subjects"] if row["ready"]]
    print("Ready subjects:", len(ready))
    if meta["num_subjects"] != 109:
        print("WARNING: expected 109 subjects, found", meta["num_subjects"])
    return ready


def run_pair(authenticator: BiometricAuthenticator, enroll_subject: str, probe_subject: str) -> dict:
    user_id = f"subject_{enroll_subject}"
    enroll = authenticator.enroll_subject(enroll_subject, user_id=user_id)
    verify = authenticator.verify_subject(user_id, probe_subject)
    row = {
        "enroll_subject": enroll_subject,
        "probe_subject": probe_subject,
        "attempt_type": verify["attempt_type"],
        "decision": verify["decision"],
        "score": verify["score"],
        "threshold": verify["threshold"],
        "enrollment_windows": enroll["evidence"]["num_windows"],
        "probe_windows": verify["evidence"]["num_windows"],
    }
    print(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the 109-subject DOMCS-EEG demo database.")
    parser.add_argument("--database", default=str(DEFAULT_DATABASE_PATH), help="Path to EEGMMIDB_win2s_step1s_fs128.npz")
    parser.add_argument("--subject", default="1", help="Subject to enroll for the quick smoke test")
    parser.add_argument("--impostor", default="2", help="Different subject used as impostor probe")
    parser.add_argument("--all-subjects", action="store_true", help="Write a one-genuine/one-impostor audit for all ready subjects")
    args = parser.parse_args()

    database_path = Path(args.database)
    authenticator = build_authenticator(database_path)
    authenticator.store.reset()
    ready = print_subject_summary(authenticator)

    print("\nQuick smoke test")
    run_pair(authenticator, args.subject, args.subject)
    run_pair(authenticator, args.subject, args.impostor)

    if args.all_subjects:
        out_dir = ROOT / "demo_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all_subject_demo_audit.csv"
        rows = []
        ready_ids = [row["subject_id"] for row in ready]
        for idx, subject_id in enumerate(ready_ids):
            authenticator.store.reset()
            impostor_id = ready_ids[(idx + 1) % len(ready_ids)]
            rows.append(run_pair(authenticator, subject_id, subject_id))
            rows.append(run_pair(authenticator, subject_id, impostor_id))
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote:", out_path)


if __name__ == "__main__":
    main()
