# Dataset registry / provenance

**Current relay paired tactile datasets: 0.** The relay single RTX3090 is busy with another workload and has not generated any new trial for this project yet. This is intentionally distinct from earlier LXC two-3090 engineering smoke checks stored outside this repo.

For each future dataset, add a small `datasets/<id>.json` card with source host role, original task and scene seed, RoboTwin code/URDF/config hash, physics dt, `completion.json`/HDF5/raw-video/sidecar SHA256, frame and physics-step counts, causal alignment parameters, train/validation/test original episode IDs, rights and artifact location. The *raw bytes* live in local `data/`; GitHub retains only approved cards and small sanitized aggregates. Do not invent timestamps for existing raw RoboTwin episodes or import LXC samples into relay splits.

## Vision-only reference (not paired tactile)

- Existing relay RoboTwin `raw/lift_pot/aloha-agilex_clean_50/data/episode0.hdf5`: 114 RGB/state frames, no physics timestamp/seed stored in HDF5; seed is in adjacent `seed.txt` and may differ from episode index.
- Existing LeRobot conversion makes 113 training rows and synthetic 50fps timestamps. It cannot be retroactively aligned with a new contact run.

Use `src/validate_expert_data.py`, then `src/build_paired_index.py` and `src/register_dataset.py` to produce a **candidate** card for a truly same-run paired capture. The registrar rejects files outside this repo's `data/paired/` and missing/incorrect relay hostname+source role; it does not approve the train/validation/test split automatically.
