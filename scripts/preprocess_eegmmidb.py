"""
DOMCS-EEG Preprocessing Script
================================
Converts raw PhysioNet EEGMMIDB (.edf) files to the canonical
EEGMMIDB_win2s_step1s_fs128.npz archive used in DOMCS-EEG training.

Paper parameters (IEEE TIFS submission T-IFS-26761-2026):
  - Dataset  : PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)
  - Subjects : 109
  - fs       : 128 Hz (resampled)
  - Window   : 2 s
  - Step     : 1 s
  - Bandpass : 0.5–45 Hz (Butterworth, order 4)
  - Notch    : 50 Hz (Q=30)
  - Norm     : per-window per-channel z-score
  - Output   : X (N, 64, 256) float32, y (N,) int32

Usage:
  python preprocess_eegmmidb.py \
      --edf_root /path/to/eeg-motor-movementimagery-dataset-1.0.0/files \
      --out_path /path/to/EEGMMIDB_win2s_step1s_fs128.npz

PhysioNet source:
  https://physionet.org/content/eegmmidb/1.0.0/
"""

import re
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mne
from scipy.signal import butter, filtfilt, iirnotch

# ── DSP helpers ──────────────────────────────────────────────────────────────

def bandpass_np(x, fs, lo=0.5, hi=45.0, order=4):
    """Zero-phase Butterworth bandpass filter. x: [C, T]"""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def notch_np(x, fs, freq=50.0, q=30.0):
    """IIR notch filter (50 Hz power line). x: [C, T]"""
    if freq >= fs / 2:
        return x
    b, a = iirnotch(freq / (fs / 2), Q=q)
    return filtfilt(b, a, x, axis=-1)


def windowize(x_ct, fs, win_sec=2.0, step_sec=1.0):
    """
    Sliding-window segmentation with per-window per-channel z-score.
    Input : x_ct [C, T]
    Output: X    [N, C, win]
    """
    win  = int(round(win_sec  * fs))
    step = int(round(step_sec * fs))
    T    = x_ct.shape[1]
    if T < win:
        return np.zeros((0, x_ct.shape[0], win), dtype=np.float32)

    starts = list(range(0, T - win + 1, step))
    out    = np.empty((len(starts), x_ct.shape[0], win), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = x_ct[:, s : s + win]
        mu  = seg.mean(axis=1, keepdims=True)
        sd  = seg.std(axis=1,  keepdims=True) + 1e-8
        out[i] = ((seg - mu) / sd).astype(np.float32)
    return out


def resample_mne(raw, target_fs):
    if int(raw.info["sfreq"]) == int(target_fs):
        return raw
    return raw.copy().resample(sfreq=target_fs)


# ── Subject / run parsing ────────────────────────────────────────────────────

def parse_eegmmidb_subject_run(path: Path):
    """Extract subject (S001) and run (R06) from EEGMMIDB filename."""
    m = re.search(r"(S\d+).*(R\d+)\.edf$", path.name)
    if m:
        return m.group(1), m.group(2)
    subj = path.parent.name
    run  = path.stem.replace(subj, "")
    return subj, run


# ── Main preprocessing ───────────────────────────────────────────────────────

def preprocess_eegmmidb(edf_root: Path, out_path: Path,
                         target_fs: int = 128,
                         win_sec: float = 2.0,
                         step_sec: float = 1.0):

    edf_files = sorted([
        p for p in edf_root.rglob("*.edf")
        if not str(p).endswith(".edf.event")
    ])
    print(f"Found {len(edf_files)} EDF files in {edf_root}")
    assert len(edf_files) > 0, "No EDF files found — check --edf_root"

    X_all, y_all, subj_all, sess_all = [], [], [], []
    ch_names_ref = None
    subj_to_idx  = {}

    for p in tqdm(edf_files, desc="Preprocessing EEGMMIDB"):
        subj, run = parse_eegmmidb_subject_run(p)
        if subj not in subj_to_idx:
            subj_to_idx[subj] = len(subj_to_idx)

        try:
            raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
            raw.pick_types(eeg=True, exclude=[])
            raw = resample_mne(raw, target_fs)

            data     = raw.get_data()           # [C, T]
            fs       = int(raw.info["sfreq"])
            ch_names = raw.ch_names

            # Align channels across files
            if ch_names_ref is None:
                ch_names_ref = ch_names
            else:
                common = [c for c in ch_names_ref if c in ch_names]
                raw.pick_channels(common)
                data         = raw.get_data()
                ch_names     = raw.ch_names
                ch_names_ref = common

            # Filter
            data = bandpass_np(data, fs, 0.5, 45.0)
            data = notch_np(data, fs, 50.0)

            # Windowing + z-score
            Xw = windowize(data, fs, win_sec, step_sec)
            if len(Xw) == 0:
                continue

            X_all.append(Xw)
            y_all.append(np.full((len(Xw),), subj_to_idx[subj], dtype=np.int32))
            subj_all.extend([subj] * len(Xw))
            sess_all.extend([run]  * len(Xw))

        except Exception as e:
            print(f"⚠️  Skipped {p.name}: {repr(e)}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X          = X.astype(np.float32),
        y          = y.astype(np.int32),
        subject_id = np.array(subj_all),
        session    = np.array(sess_all),
        fs         = np.int32(target_fs),
        ch_names   = np.array(ch_names_ref if ch_names_ref else []),
    )

    print(f"\n✅ Saved: {out_path}")
    print(f"   X shape        : {X.shape}   (expect ~(173198, 64, 256))")
    print(f"   Unique subjects: {len(subj_to_idx)}   (expect 109)")
    print(f"   Unique sessions: {sorted(set(sess_all))}")
    print(f"   Channels       : {len(ch_names_ref)}")
    print(f"   fs             : {target_fs} Hz")

    return X, y


# ── Verification helper ──────────────────────────────────────────────────────

def verify_npz(npz_path: Path):
    """
    Verify a saved .npz against DOMCS-EEG paper claims.
    Run after preprocessing to confirm alignment.
    """
    d = np.load(npz_path, allow_pickle=True)

    print("\n══ DOMCS-EEG .npz Verification ══")
    print(f"  Keys      : {list(d.files)}")

    X  = d["X"]
    y  = d["y"]
    fs = int(d["fs"])

    checks = {
        "X.dtype == float32"          : X.dtype == np.float32,
        "X.ndim == 3"                 : X.ndim == 3,
        "X.shape[1] == 64 (channels)" : X.shape[1] == 64,
        "X.shape[2] == 256 (2s×128Hz)": X.shape[2] == 256,
        "fs == 128"                   : fs == 128,
        "N windows > 100000"          : X.shape[0] > 100_000,
        "Unique subjects == 109"      : len(set(d["subject_id"])) == 109,
        "Sessions include R01"        : "R01" in set(d["session"]),
        "Sessions include R14"        : "R14" in set(d["session"]),
        "X z-scored (|mean| < 0.1)"   : abs(X[:100].mean()) < 0.1,
    }

    all_pass = True
    for label, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {label}")
        if not result:
            all_pass = False

    print(f"\n  X shape : {X.shape}")
    print(f"  X range : [{X.min():.2f}, {X.max():.2f}]  (z-scored, expect ~±10)")
    print(f"  Subjects: {len(set(d['subject_id']))}")
    print(f"  Sessions: {sorted(set(d['session']))}")
    print("\n" + ("✅ ALL CHECKS PASSED" if all_pass else "❌ SOME CHECKS FAILED"))
    return all_pass


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOMCS-EEG EEGMMIDB Preprocessor")
    parser.add_argument("--edf_root", type=Path, required=True,
                        help="Path to PhysioNet EEGMMIDB files/ directory")
    parser.add_argument("--out_path", type=Path,
                        default=Path("EEGMMIDB_win2s_step1s_fs128.npz"),
                        help="Output .npz path")
    parser.add_argument("--verify_only", type=Path, default=None,
                        help="Skip preprocessing; just verify an existing .npz")
    args = parser.parse_args()

    if args.verify_only:
        verify_npz(args.verify_only)
    else:
        preprocess_eegmmidb(args.edf_root, args.out_path)
        verify_npz(args.out_path)
