#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/mnt/relaydata/lingbot-tactile-lab
PY=/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
OWNER=lingbot-tactile-lab/real-v2-bf16-stream
SEED=${1:?}; CONDITION=${2:?}
unset GPU_LEASE_DIR
acquired=0
cleanup(){ rc=$?;trap - EXIT;if ((acquired));then python3 -B "$LEASE" release --owner "$OWNER";fi;exit "$rc"; }
trap cleanup EXIT
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 -B "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 2100 --priority batch --task "fixed recovery development seed=$SEED condition=$CONDITION; no Strata lifecycle changes"
acquired=1
export PYTHONPATH="$BASE/src:/mnt/relaydata/lingbot/lingbot-vla-v2:$BASE/work/overlay_policy_equivalent_pilot_v1"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ROBOTWIN_SOURCE=/mnt/relaydata/lingbot/RoboTwin
cd "$BASE/work/overlay_policy_equivalent_pilot_v1"
timeout --signal=TERM --kill-after=20s 1500s "$PY" -B "$BASE/src/record_recovery_development.py" --seed "$SEED" --condition "$CONDITION"
