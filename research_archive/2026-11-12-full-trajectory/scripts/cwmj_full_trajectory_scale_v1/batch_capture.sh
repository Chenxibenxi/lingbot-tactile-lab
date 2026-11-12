#!/bin/bash
set -Eeuo pipefail
cd /mnt/relaydata/lingbot-tactile-lab
unset GPU_LEASE_DIR
OWNER=lingbot-tactile-lab/full-trajectory-capture
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 600 --priority batch --task 'full trajectories train48001-48127 capture and audit'
trap 'python3 "$LEASE" release --owner "$OWNER"' EXIT
export MUJOCO_GL=egl PYTHONPATH=/mnt/relaydata/lingbot-tactile-lab/src OMP_NUM_THREADS=4
PY=/home/admin001/miniconda3/envs/sim_env/bin/python
for seed in $(seq 48001 48127); do
 [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
 python3 "$LEASE" status --json | python3 -c 'import json,sys; x=json.load(sys.stdin); assert x.get("state")=="held" and x.get("holder")=="lingbot-tactile-lab/full-trajectory-capture" and x.get("pid_alive") and not x.get("waiting")'
 python3 "$LEASE" renew --owner "$OWNER" --ttl 600
 "$PY" work/cwmj_full_trajectory_scale_v1/collect.py "$seed"
 "$PY" work/cwmj_full_trajectory_scale_v1/quality.py "$seed"
 "$PY" work/cwmj_full_trajectory_scale_v1/report_capture.py "$seed"
done
