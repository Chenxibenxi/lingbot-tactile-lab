#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/mnt/relaydata/lingbot-tactile-lab
PY=/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
OWNER=lingbot-tactile-lab/real-v2-bf16-stream
REPEAT=${1:?}
ARM=${2:?}
[[ "$REPEAT" =~ ^[012]$ ]]
[[ "$ARM" =~ ^(zero|true|mismatch|contact_state)$ ]]
unset GPU_LEASE_DIR
acquired=0
cleanup(){ rc=$?;trap - EXIT; if ((acquired));then python3 -B "$LEASE" release --owner "$OWNER";fi;exit "$rc"; }
trap cleanup EXIT
while true; do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    if python3 -B "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 3900 --priority batch --task 'M8 fixed-data duration training, no Strata lifecycle actions'; then
      acquired=1
      break
    fi
  fi
  sleep 30
done
export PYTHONPATH="$BASE/src:/mnt/relaydata/lingbot/lingbot-vla-v2"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /mnt/relaydata/lingbot/lingbot-vla-v2
timeout --signal=TERM --kill-after=20s 3600s "$PY" -B "$BASE/src/run_cwmj_m8_training.py" --repeat "$REPEAT" --arm "$ARM"
