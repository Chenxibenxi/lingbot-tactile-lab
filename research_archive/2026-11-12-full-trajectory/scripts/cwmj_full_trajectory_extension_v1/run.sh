#!/bin/bash
set -Eeuo pipefail
cd /mnt/relaydata/lingbot-tactile-lab

while true; do
 state=$(systemctl --user show cwmj-full-trajectory-training-v1.service -p ActiveState --value)
 if [[ "$state" != active && "$state" != activating && "$state" != deactivating ]]; then break; fi
 sleep 60
done
python3 - <<'CHECK'
import json,time
from pathlib import Path
r=Path('/mnt/relaydata/lingbot-tactile-lab');s=json.loads((r/'data/reports/cwmj_full_trajectory_training_v1/status.json').read_text());assert s['status']=='completed' and len(s['completed_arms'])==3
c=json.loads((r/'datasets/cwmj_full_trajectory_extension_v1.json').read_text());assert time.time()<c['deadline_unix']
CHECK

unset GPU_LEASE_DIR
OWNER=lingbot-tactile-lab/real-v2-bf16-stream
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 1200 --priority batch --task 'continue three matched arms epochs4and5 until morning10'
trap 'python3 "$LEASE" release --owner "$OWNER"' EXIT
export MUJOCO_GL=egl PYTHONPATH=/mnt/relaydata/lingbot-tactile-lab/src:/mnt/relaydata/lingbot/lingbot-vla-v2 OMP_NUM_THREADS=4
/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -u work/cwmj_full_trajectory_extension_v1/run.py
