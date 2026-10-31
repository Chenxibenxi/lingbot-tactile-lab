# Private relay 3090 data area

All **new** tactile recordings and model outputs belong here, generated on this relay RTX 3090. `paired/` contains one immutable directory per attempted same-run RoboTwin expert capture. A valid candidate must include a successful `completion.json`, sidecar `.jsonl` streams, demonstration `episodeN.hdf5` and `src/validate_expert_data.py` verification. Failed/partial attempts stay quarantined and never enter splits. `raw/` is for auxiliary on-relay assets; `reports/` stores measured evaluation and hardware timing.

This folder is deliberately excluded from GitHub Git history because raw video, datasets and weights can exceed GitHub's 100MB/file limit and may contain private research data. Their filenames, sizes, SHA256 and provenance can be listed in `datasets/` manifests without uploading raw bytes. Before choosing an authenticated artifact store, never say a GitHub clone contains the full datasets.

Historic data from LXC GPUs are **not** relay-sourced and cannot be copied into `data/paired/` or silently counted as relay evaluations. Existing `/mnt/relaydata/lingbot/raw/` is vision-only reference and lacks true physical timestamps; do not reconstruct tactile pairs with t/50.
