#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/mnt/relaydata/lingbot-tactile-lab
PY=/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
OWNER=lingbot-tactile-lab/real-v2-bf16-stream
MICRO=${1:?}
case "$MICRO" in 1|2|4) ;; *) exit 2 ;; esac
export PYTHONPATH="$BASE/src:/mnt/relaydata/lingbot/lingbot-vla-v2"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$BASE/src/verify_microbatch_benchmark_v2.py"
unset GPU_LEASE_DIR
acquired=0
cleanup(){ rc=$?;trap - EXIT;if ((acquired));then python3 -B "$LEASE" release --owner "$OWNER";fi;exit "$rc"; }
trap cleanup EXIT
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 -B "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 1200 --priority batch --task "fixed microbatch $MICRO benchmark; no Strata lifecycle"
acquired=1
export PYTHONPATH="$BASE/src:/mnt/relaydata/lingbot/lingbot-vla-v2"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$BASE"
timeout --signal=TERM --kill-after=20s 1000s "$PY" -B "$BASE/src/run_microbatch_benchmark_v2.py" --micro "$MICRO"
