import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch

from domcs_eeg.model import DOMCSEEG

DEFAULT_DATABASE_PATH = Path(
    r"C:\Users\L.KANIMOZHI\Downloads\DAILY TASKS\APR 2026\21 APR 26\EEGMMIDB_win2s_step1s_fs128.npz"
)
ENROLL_RUNS = ("r01", "r02")
VERIFY_RUNS = tuple(f"r{idx:02d}" for idx in range(3, 15))


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-12)


def _load_np_array(payload: bytes) -> np.ndarray:
    buffer = io.BytesIO(payload)
    if payload[:2] == b"PK":
        archive = np.load(buffer, allow_pickle=True)
        for key in ("X", "windows", "data", "eeg"):
            if key in archive.files:
                return np.asarray(archive[key], dtype=np.float32)
        raise ValueError("NPZ file must contain one of: X, windows, data, eeg")
    return np.asarray(np.load(buffer, allow_pickle=True), dtype=np.float32)


def _load_npz_archive(payload: bytes):
    return np.load(io.BytesIO(payload), allow_pickle=True)


def _canon_run(value: Any) -> str:
    token = str(value).lower().strip().replace("session", "").replace("run", "r")
    token = token.replace("_", "").replace("-", "")
    if token.startswith("r") and token[1:].isdigit():
        return "r" + token[1:].zfill(2)
    if token.isdigit():
        return "r" + token.zfill(2)
    return token


def _canon_subject(value: Any) -> str:
    token = str(value).strip().lower()
    digits = "".join(ch for ch in token if ch.isdigit())
    return str(int(digits)) if digits else token


def _sort_tokens(values: set[str] | list[str]) -> list[str]:
    def key_fn(value: str):
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    return sorted(values, key=key_fn)


def _split_run_list(text: str) -> list[str]:
    return [_canon_run(part) for part in text.split(",") if part.strip()]


def ensure_window_shape(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape == (64, 256):
            arr = arr[None, ...]
        elif arr.shape == (256, 64):
            arr = np.transpose(arr, (1, 0))[None, ...]
        else:
            raise ValueError("Expected a single EEG window with shape (64, 256) or (256, 64)")
    elif arr.ndim == 3:
        if arr.shape[1:] == (64, 256):
            pass
        elif arr.shape[1:] == (256, 64):
            arr = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError("Expected batched EEG windows with shape (N, 64, 256) or (N, 256, 64)")
    else:
        raise ValueError("EEG input must be a 2D or 3D array")
    return arr.astype(np.float32, copy=False)


def parse_master_npz_windows(payload: bytes, subject_id: str, run_list: str) -> np.ndarray:
    archive = _load_npz_archive(payload)
    x_key = next((key for key in ("X", "windows", "data", "eeg") if key in archive.files), None)
    y_key = next((key for key in ("y", "Y", "subject", "subjects", "labels") if key in archive.files), None)
    run_key = next((key for key in ("session", "sessions", "run", "runs", "trial", "trials") if key in archive.files), None)
    if x_key is None or y_key is None or run_key is None:
        raise ValueError("Master NPZ must contain data, subject, and run fields")
    windows = ensure_window_shape(np.asarray(archive[x_key], dtype=np.float32))
    subjects = np.asarray(archive[y_key])
    runs = np.asarray(archive[run_key])
    subject_token = _canon_subject(subject_id)
    requested_runs = set(_split_run_list(run_list))
    if not requested_runs:
        raise ValueError("Provide at least one run, for example: r01,r02")
    subject_mask = np.array([_canon_subject(value) == subject_token for value in subjects], dtype=bool)
    run_mask = np.array([_canon_run(value) in requested_runs for value in runs], dtype=bool)
    mask = subject_mask & run_mask
    if mask.sum() == 0:
        raise ValueError("No windows matched the selected subject and runs")
    return ensure_window_shape(windows[mask])


def inspect_master_npz(payload: bytes) -> dict[str, Any]:
    archive = _load_npz_archive(payload)
    y_key = next((key for key in ("y", "Y", "subject", "subjects", "labels") if key in archive.files), None)
    run_key = next((key for key in ("session", "sessions", "run", "runs", "trial", "trials") if key in archive.files), None)
    x_key = next((key for key in ("X", "windows", "data", "eeg") if key in archive.files), None)
    if x_key is None or y_key is None or run_key is None:
        raise ValueError("Master NPZ must contain data, subject, and run fields")
    windows = ensure_window_shape(np.asarray(archive[x_key], dtype=np.float32))
    subjects = sorted({_canon_subject(value) for value in np.asarray(archive[y_key])})
    runs = sorted({_canon_run(value) for value in np.asarray(archive[run_key])})
    return {
        "num_windows": int(windows.shape[0]),
        "window_shape": [int(dim) for dim in windows.shape[1:]],
        "subjects": subjects[:200],
        "runs": runs,
        "keys": archive.files,
    }


def _npz_member_shape(path: Path, key: str) -> list[int] | None:
    member = f"{key}.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                else:
                    shape, _, _ = np.lib.format._read_array_header(stream, version)
                return [int(dim) for dim in shape]
    except Exception:
        return None


class NpzSubjectDatabase:
    """Server-side preprocessed EEGMMIDB database for the 109-subject demo."""

    def __init__(
        self,
        path: Path | None = None,
        enrollment_runs: tuple[str, ...] = ENROLL_RUNS,
        verification_runs: tuple[str, ...] = VERIFY_RUNS,
    ):
        env_path = os.environ.get("DOMCS_EEG_NPZ_PATH")
        self.path = Path(env_path) if env_path else (path or DEFAULT_DATABASE_PATH)
        self.enrollment_runs = tuple(_canon_run(run) for run in enrollment_runs)
        self.verification_runs = tuple(_canon_run(run) for run in verification_runs)
        self._meta: dict[str, Any] | None = None

    def exists(self) -> bool:
        return self.path.exists()

    def _archive(self):
        if not self.exists():
            raise FileNotFoundError(
                "Preprocessed NPZ database not found. Set DOMCS_EEG_NPZ_PATH or place the file at "
                f"{self.path}"
            )
        return np.load(self.path, allow_pickle=True)

    def load_metadata(self, force: bool = False) -> dict[str, Any]:
        if self._meta is not None and not force:
            return self._meta

        archive = self._archive()
        x_key = next((key for key in ("X", "windows", "data", "eeg") if key in archive.files), None)
        subject_key = next(
            (key for key in ("subject_id", "subject", "subjects", "y", "Y", "labels") if key in archive.files),
            None,
        )
        run_key = next((key for key in ("session", "sessions", "run", "runs", "trial", "trials") if key in archive.files), None)
        if x_key is None or subject_key is None or run_key is None:
            raise ValueError("NPZ database must contain data, subject, and session/run fields")

        subjects_raw = np.asarray(archive[subject_key])
        runs_raw = np.asarray(archive[run_key])
        subjects = np.array([_canon_subject(value) for value in subjects_raw])
        runs = np.array([_canon_run(value) for value in runs_raw])
        unique_subjects = _sort_tokens(set(subjects.tolist()))
        unique_runs = _sort_tokens(set(runs.tolist()))

        subject_rows = []
        enroll_set = set(self.enrollment_runs)
        verify_set = set(self.verification_runs)
        for subject_id in unique_subjects:
            subject_mask = subjects == subject_id
            subject_runs = _sort_tokens(set(runs[subject_mask].tolist()))
            enrollment_count = int((subject_mask & np.isin(runs, list(enroll_set))).sum())
            verification_count = int((subject_mask & np.isin(runs, list(verify_set))).sum())
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "runs": subject_runs,
                    "total_windows": int(subject_mask.sum()),
                    "enrollment_windows": enrollment_count,
                    "verification_windows": verification_count,
                    "ready": bool(enrollment_count > 0 and verification_count > 0),
                }
            )

        x_shape = _npz_member_shape(self.path, x_key)
        if x_shape is None:
            x_shape = [int(dim) for dim in np.asarray(archive[x_key]).shape]
        fs = archive["fs"].item() if "fs" in archive.files and np.asarray(archive["fs"]).shape == () else None
        ch_names = [str(value) for value in np.asarray(archive["ch_names"]).tolist()] if "ch_names" in archive.files else []

        self._meta = {
            "path": str(self.path),
            "exists": True,
            "keys": list(archive.files),
            "x_key": x_key,
            "subject_key": subject_key,
            "run_key": run_key,
            "shape": x_shape,
            "num_windows": int(x_shape[0]) if x_shape else None,
            "num_subjects": len(unique_subjects),
            "subjects": subject_rows,
            "runs": unique_runs,
            "enrollment_runs": list(self.enrollment_runs),
            "verification_runs": list(self.verification_runs),
            "fs": int(fs) if fs is not None else None,
            "channel_count": len(ch_names) or (int(x_shape[1]) if x_shape and len(x_shape) > 1 else None),
            "ch_names": ch_names,
        }
        return self._meta

    def status(self) -> dict[str, Any]:
        if not self.exists():
            return {
                "path": str(self.path),
                "exists": False,
                "error": "NPZ database file not found",
                "enrollment_runs": list(self.enrollment_runs),
                "verification_runs": list(self.verification_runs),
            }
        return self.load_metadata()

    def subject_windows(self, subject_id: str, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
        meta = self.load_metadata()
        subject_token = _canon_subject(subject_id)
        run_list = self.enrollment_runs if mode == "enroll" else self.verification_runs
        archive = self._archive()
        windows = ensure_window_shape(np.asarray(archive[meta["x_key"]], dtype=np.float32))
        subjects = np.array([_canon_subject(value) for value in np.asarray(archive[meta["subject_key"]])])
        runs = np.array([_canon_run(value) for value in np.asarray(archive[meta["run_key"]])])
        mask = (subjects == subject_token) & np.isin(runs, list(run_list))
        if mask.sum() == 0:
            raise ValueError(f"No {mode} windows found for subject {subject_id}")
        selected = ensure_window_shape(windows[mask])
        evidence = {
            "subject_id": subject_token,
            "mode": mode,
            "runs": list(run_list),
            "num_windows": int(selected.shape[0]),
            "window_shape": [int(dim) for dim in selected.shape[1:]],
        }
        return selected, evidence


def preprocess_edf_payloads(payloads: list[bytes], low_freq: float = 1.0, high_freq: float = 40.0) -> np.ndarray:
    all_windows = []
    for payload in payloads:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        try:
            raw = mne.io.read_raw_edf(str(tmp_path), preload=True, verbose="ERROR")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) < 64:
            raise ValueError("EDF file has fewer than 64 EEG channels")
        raw.pick(eeg_picks[:64])
        raw.filter(low_freq, high_freq, verbose="ERROR")
        raw.notch_filter(freqs=[50, 60], verbose="ERROR")
        raw.resample(128, verbose="ERROR")
        data = raw.get_data().astype(np.float32)
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
        win = 256
        step = 128
        for start in range(0, data.shape[1] - win + 1, step):
            all_windows.append(data[:, start:start + win])
    if not all_windows:
        raise ValueError("EDF preprocessing produced no windows")
    return ensure_window_shape(np.stack(all_windows, axis=0))


@dataclass
class AuthConfig:
    checkpoint_path: Path
    registry_path: Path
    embeddings_dir: Path
    database_path: Path = DEFAULT_DATABASE_PATH
    threshold: float = 0.58
    min_windows: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EnrollmentStore:
    def __init__(self, registry_path: Path, embeddings_dir: Path):
        self.registry_path = registry_path
        self.embeddings_dir = embeddings_dir
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"users": {}})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_users(self) -> list[dict[str, Any]]:
        data = self._read()["users"]
        return [{"user_id": user_id, **meta} for user_id, meta in sorted(data.items())]

    def has_user(self, user_id: str) -> bool:
        return user_id in self._read()["users"]

    def get_user(self, user_id: str) -> dict[str, Any]:
        users = self._read()["users"]
        if user_id not in users:
            raise KeyError(f"Unknown user_id: {user_id}")
        return users[user_id]

    def save_user(self, user_id: str, embeddings: np.ndarray, centroid: np.ndarray) -> dict[str, Any]:
        users = self._read()
        path = self.embeddings_dir / f"{user_id}.npz"
        np.savez_compressed(
            path,
            embeddings=embeddings.astype(np.float32),
            centroid=centroid.astype(np.float32),
        )
        users["users"][user_id] = {
            "num_windows": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "storage_path": str(path),
        }
        self._write(users)
        return users["users"][user_id]

    def load_templates(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        meta = self.get_user(user_id)
        payload = np.load(meta["storage_path"], allow_pickle=False)
        return payload["embeddings"].astype(np.float32), payload["centroid"].astype(np.float32)

    def reset(self) -> None:
        if self.embeddings_dir.exists():
            try:
                shutil.rmtree(self.embeddings_dir)
            except PermissionError:
                for child in self.embeddings_dir.glob("*.npz"):
                    try:
                        child.unlink()
                    except PermissionError:
                        pass
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._write({"users": {}})


class BiometricAuthenticator:
    def __init__(self, config: AuthConfig):
        self.config = config
        self.store = EnrollmentStore(config.registry_path, config.embeddings_dir)
        self.database = NpzSubjectDatabase(config.database_path)
        self.model = DOMCSEEG.from_checkpoint(str(config.checkpoint_path), device=config.device)

    def _extract_embeddings(self, windows: np.ndarray) -> np.ndarray:
        eeg = ensure_window_shape(windows)
        tensor = torch.from_numpy(eeg).to(self.config.device)
        with torch.no_grad():
            embeddings = self.model.get_identity_embedding(tensor).detach().cpu().numpy()
        return _l2_normalize(embeddings.astype(np.float32))

    def enroll(self, user_id: str, windows: np.ndarray) -> dict[str, Any]:
        embeddings = self._extract_embeddings(windows)
        if embeddings.shape[0] < self.config.min_windows:
            raise ValueError(f"Enrollment requires at least {self.config.min_windows} windows")
        centroid = _l2_normalize(embeddings.mean(axis=0, keepdims=True))[0]
        meta = self.store.save_user(user_id=user_id, embeddings=embeddings, centroid=centroid)
        return {
            "user_id": user_id,
            "status": "enrolled",
            "num_windows": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "meta": meta,
        }

    def enroll_subject(self, subject_id: str, user_id: str | None = None) -> dict[str, Any]:
        windows, evidence = self.database.subject_windows(subject_id, mode="enroll")
        app_user_id = user_id or f"subject_{_canon_subject(subject_id)}"
        result = self.enroll(user_id=app_user_id, windows=windows)
        result["protocol"] = "B2T enrollment: baseline runs R01-R02"
        result["evidence"] = evidence
        return result

    def verify(self, user_id: str, windows: np.ndarray) -> dict[str, Any]:
        probe_embeddings = self._extract_embeddings(windows)
        templates, centroid = self.store.load_templates(user_id)
        centroid_scores = probe_embeddings @ centroid
        template_scores = probe_embeddings @ templates.T
        per_window = 0.6 * centroid_scores + 0.4 * template_scores.max(axis=1)
        score = float(per_window.mean())
        decision = "accept" if score >= self.config.threshold else "reject"
        return {
            "user_id": user_id,
            "decision": decision,
            "verified": bool(decision == "accept"),
            "score": score,
            "threshold": float(self.config.threshold),
            "num_probe_windows": int(probe_embeddings.shape[0]),
            "per_window_scores": per_window.round(6).tolist(),
        }

    def verify_subject(self, claimed_user_id: str, probe_subject_id: str) -> dict[str, Any]:
        windows, evidence = self.database.subject_windows(probe_subject_id, mode="verify")
        result = self.verify(user_id=claimed_user_id, windows=windows)
        result["claimed_user_id"] = claimed_user_id
        result["probe_subject_id"] = _canon_subject(probe_subject_id)
        result["protocol"] = "B2T verification: task runs R03-R14"
        result["evidence"] = evidence
        result["attempt_type"] = "genuine" if claimed_user_id == f"subject_{_canon_subject(probe_subject_id)}" else "impostor"
        return result

    def identify(self, windows: np.ndarray) -> dict[str, Any]:
        users = self.store.list_users()
        if not users:
            raise ValueError("No enrolled users available")
        probe_embeddings = self._extract_embeddings(windows)
        best_user = None
        best_score = -1.0
        for user in users:
            templates, centroid = self.store.load_templates(user["user_id"])
            centroid_scores = probe_embeddings @ centroid
            template_scores = probe_embeddings @ templates.T
            score = float((0.6 * centroid_scores + 0.4 * template_scores.max(axis=1)).mean())
            if score > best_score:
                best_score = score
                best_user = user["user_id"]
        accepted = best_score >= self.config.threshold
        return {
            "identified_user": best_user,
            "score": float(best_score),
            "threshold": float(self.config.threshold),
            "decision": "accept" if accepted else "reject",
        }

    @staticmethod
    def parse_uploaded_array(payload: bytes) -> np.ndarray:
        return ensure_window_shape(_load_np_array(payload))

    @staticmethod
    def parse_master_npz(payload: bytes, subject_id: str, run_list: str) -> np.ndarray:
        return parse_master_npz_windows(payload, subject_id, run_list)

    @staticmethod
    def inspect_master_npz(payload: bytes) -> dict[str, Any]:
        return inspect_master_npz(payload)

    @staticmethod
    def parse_raw_edf(payloads: list[bytes]) -> np.ndarray:
        return preprocess_edf_payloads(payloads)
