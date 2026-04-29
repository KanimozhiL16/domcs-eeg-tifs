# Paper Evidence

This directory contains table-level evidence for the submitted DOMCS-EEG TIFS manuscript.

Run:

```bash
python scripts/verify_full_table_evidence.py
```

The verifier checks that all required evidence files exist and that the locked paper values match the repository's canonical values.

Evidence classes:

- `raw exported CSV`: copied from experiment output folders.
- `repository lock`: values intentionally locked for reproducibility checks.
- `paper-extracted lock`: values reported in the submitted paper and preserved as structured CSV because no standalone raw result CSV was found.
