# Dataset registry / provenance

**Updated 2026-11-08:** relay same-run tactile collection, adapter training and M6–M8 closed-loop studies have run. The earlier zero-dataset note is obsolete. Candidate capture, training-admitted data, policy self-rollout and confirmation outcomes remain distinct; a successful simulation alone does not approve training. Historical LXC two-3090 smoke checks remain outside relay splits.

## Seed use and admission

- `tactile_seed_bank_m1_m6_v3.json`: earlier registered collection and study history.
- `tactile_seed_bank_m7_consumed_v1.json`: consumed M7 confirmation seeds.
- `tactile_seed_bank_m8_recovery_consumed_v1.json`: 100103–100118 consumed M8 tests, 100119–100126 consumed recovery development, and 100127–100128 reserved for the launched feedback technical batch. Failed, unknown and prerequisite-failed seeds are not fresh tests.
- Frozen reservation files remain unchanged; consumption is recorded separately with source SHA256. All seeds in the new registry are disallowed for training supervision, validation selection and future independent confirmation.
- Recovery v1 and v2 did not pass their development gates. Do not promote their successful subsets into training. Technical feedback pilot outcomes are not confirmation evidence.

Local final reports are under `data/reports/cwmj_m8_confirmation_summary/` and `data/reports/recovery_development_v2_final/`. New feedback pilot bytes use the dedicated `data/eval/controlled_feedback_pilot_v1` link to storage on the same workstation; they are not imported external-machine data.

For each future dataset, add a small `datasets/<id>.json` card with source host role, original task and scene seed, RoboTwin code/URDF/config hash, physics dt, `completion.json`/HDF5/raw-video/sidecar SHA256, frame and physics-step counts, causal alignment parameters, train/validation/test original episode IDs, rights and artifact location. The *raw bytes* live in local `data/`; GitHub retains only approved cards and small sanitized aggregates. Do not invent timestamps for existing raw RoboTwin episodes or import LXC samples into relay splits.

## Vision-only reference (not paired tactile)

- Existing relay RoboTwin `raw/lift_pot/aloha-agilex_clean_50/data/episode0.hdf5`: 114 RGB/state frames, no physics timestamp/seed stored in HDF5; seed is in adjacent `seed.txt` and may differ from episode index.
- Existing LeRobot conversion makes 113 training rows and synthetic 50fps timestamps. It cannot be retroactively aligned with a new contact run.

Use `src/validate_expert_data.py`, then `src/build_paired_index.py` and `src/register_dataset.py` to produce a **candidate** card for a truly same-run paired capture. The registrar rejects files outside this repo's `data/paired/` and missing/incorrect relay hostname+source role; it does not approve the train/validation/test split automatically.
