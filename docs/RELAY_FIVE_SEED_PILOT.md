# Five independent relay expert seeds: pilot split, not a trained model

All five RoboTwin `lift_pot` expert trials below were generated on this workstation's **single RTX 3090**, each in its own ignored `data/paired/` directory. Each passed the rule expert's planner and goal check; its SHA-bound pre-conversion gate, completed same-run RGB HDF5, physics, observation and saved-frame markers passed the relay-only validator and candidate registry. The earlier failed/planner or conversion-interrupted trial IDs and the LXC samples were excluded.

| Actual seed | Saved source frames | Causal usable next-command pairs | Pilot fold |
|---:|---:|---:|---|
| 100000 | 110 | 102 | train |
| 100003 | 112 | 104 | train |
| 100005 | 112 | 104 | validation |
| 100014 | 112 | 103 | train |
| 100048 | 113 | 105 | test |

Total: **559 source frames; 518 usable causal pairs; 5 distinct actual seeds**. Candidate cards with per-file SHA256 are in `datasets/cards/`; private bytes remain gitignored. The fixed-seed pilot split in `datasets/relay_lift_pot_seed_split_candidate_v1.json` groups by actual seed and is deliberately `candidate_not_approved`: 3 train, 1 validation, 1 test seed, with no retry of the same seed allowed to cross folds. This passes the **technical** minimum-five-seed gate but is too small and selected toward expert successes for statistical generalization claims.

`src/plan_relay_training.py` revalidates all current relay expert cards and fits normalization using **only the 309 usable training-fold pairs**, never the 104 validation or 105 test pairs. Its output `configs/relay_lift_pot_train_norm_candidate_v1.json` is **not approved for training**. Important diagnostic: only ~22.5–23.0% of complete training finger bins have nonzero mean contact proxy; every per-finger median and 25th/75th-force quartile is zero, so the existing IQR floor `0.1` triggers on **all 4×3 features**. Naively dividing rare positive forces by 0.1 could amplify them tenfold. Review a train-only, contact-aware/positive-bin normalization and bounded clipping, record its ablation, then lock this preprocessing before evaluating either model.

Before any scientific improvement claim: acquire more independent seeds and contact/occlusion/failure counterexamples; train **matched-data** visual-only and tactile models, compare shuffled/invalid tactile controls at identical seeds; preserve the original vision weights and source. The model and server remain **not tactile-conditioned**, and `datasets/provenance_v1.json:approved_paired_episodes` remains empty. Simulation impulse/dt is only a contact proxy, not GelSight or an actual gripper force limit. GPU leases were acquired only when Strata had zero active/queued requests, and Strata was restored after each bounded batch.
