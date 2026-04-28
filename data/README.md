# Dataset Placement

The full preprocessed EEGMMIDB archive is not stored in GitHub because it is large. Reviewers should download EEGMMIDB from PhysioNet and use the same preprocessing described in the paper, or place the prepared archive at the expected path.

PhysioNet source:

```text
https://physionet.org/content/eegmmidb/1.0.0/
```

Expected preprocessed file name:

```text
EEGMMIDB_win2s_step1s_fs128.npz
```

Expected archive structure:

| Key | Shape / Type |
|---|---|
| `X` | `(173198, 64, 256)` float32 |
| `y` | `(173198,)` int32 |
| `subject_id` | `(173198,)` string/integer subject identifiers |
| `session` | `(173198,)` run labels such as R01-R14 |
| `fs` | scalar, 128 |
| `ch_names` | `(64,)` channel labels |

The NVIDIA A100 runs used this canonical location:

```text
/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz
```

If your dataset is stored elsewhere, either create a symlink to the canonical path or adjust the dataset path in `scripts/train_60ep.py` before running the full experiment.

Example symlink on Linux:

```bash
mkdir -p /home/nvidia/24PHD1237/EEGMMIDB
ln -s /path/to/EEGMMIDB_win2s_step1s_fs128.npz /home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz
```

Before training, confirm the archive opens correctly:

```bash
python - <<'PY'
import numpy as np
p = '/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz'
d = np.load(p, allow_pickle=True)
print(d.files)
for k in d.files:
    x = d[k]
    print(k, getattr(x, 'shape', None), getattr(x, 'dtype', None))
PY
```
