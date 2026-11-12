#!/bin/bash
set -Eeuo pipefail
cd /mnt/relaydata/lingbot-tactile-lab
unset GPU_LEASE_DIR
OWNER=lingbot-tactile-lab/real-v2-bf16-stream
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 1200 --priority batch --task '128 full trajectories three epochs three matched tactile controls'
trap 'python3 "$LEASE" release --owner "$OWNER"' EXIT
export MUJOCO_GL=egl PYTHONPATH=/mnt/relaydata/lingbot-tactile-lab/src:/mnt/relaydata/lingbot/lingbot-vla-v2 OMP_NUM_THREADS=4
/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -u work/cwmj_full_trajectory_training_v1/run.py
