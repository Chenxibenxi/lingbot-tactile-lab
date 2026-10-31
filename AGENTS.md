# Working rules: lingbot-tactile-lab

- This is a **private research repository**. Baseline PhiZJU / RoboTwin / LingBot training and evaluation checkouts are read-only reference; evolve new data, training and evaluation in this repository's opt-in code and reproducible branches, then deliberately integrate when gates pass.
- The relay workstation at `/mnt/relaydata` with its **single RTX 3090** is the sole source of *new* tactile datasets, training/evaluation outputs and parameters. Historical LXC results are external validation only and must never enter relay-sourced train/validation/test manifests.
- Before any GPU job check `nvidia-smi` and the process owner. Do not stop/evict another workload or claim a busy GPU. CPU tests are permitted.
- All experiment outputs live under `data/`, model artifacts under `models/`, run scratch under `work/`. These folders are local, private and ignored by Git; version code, configs, manifests, checksums and small sanitized reports only. No credentials/private logs/huge weights in GitHub.
- Existing RoboTwin expert HDF5 and LeRobot's 50 FPS timestamp do **not** contain genuine physical capture timing. Training paired tactile examples must originate from the same run and pass `validate_expert_data.py` plus a completion manifest.
- Never bypass `valid_mask`, inspect future force in current inference inputs, or turn impulse/dt proxy into a calibrated physical gripper limit. Real hardware force control needs separate sensor calibration and safety interlocks.
- Keep original 185800 vision checkpoint and outputs as baseline; training and inference tactile capability is the objective, not permanent logging-only scope. Compare same-dataset visual-only, shuffled-tactile, tactile and controlled-feedback variants.
- GitHub writes: explicitly named paths and reviewed SHA256/size list on a non-main research branch; do not snapshot-push. Remote private repository creation/push must use valid credentials; never display tokens.
