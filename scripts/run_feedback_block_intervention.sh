#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/mnt/relaydata/lingbot-tactile-lab
PY=/mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
OWNER=lingbot-tactile-lab/real-v2-bf16-stream
SEED=${1:?}; ARM=${2:?}; MODE=${3:?}
case "$SEED" in 100145|100146) ;; *) exit 2 ;; esac
case "$ARM" in live|history_zero) ;; *) exit 2 ;; esac
case "$MODE" in fresh_first|cached_block) ;; *) exit 2 ;; esac
export PYTHONPATH="$BASE/src:/mnt/relaydata/lingbot/lingbot-vla-v2:$BASE/work/overlay_policy_equivalent_pilot_v1"
CUDA_VISIBLE_DEVICES='' "$PY" -B "$BASE/src/verify_feedback_block_intervention.py" --seed "$SEED" --arm "$ARM" --execution-mode "$MODE"
unset GPU_LEASE_DIR
acquired=0
cleanup(){ rc=$?;trap - EXIT;if ((acquired));then python3 -B "$LEASE" release --owner "$OWNER";fi;exit "$rc"; }
trap cleanup EXIT
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 -B "$LEASE" acquire --owner "$OWNER" --pid "$$" --ttl 2100 --priority batch --task "feedback confirmation seed=$SEED diagnostic arm=$ARM cadence=$MODE"
acquired=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ROBOTWIN_SOURCE=/mnt/relaydata/lingbot/RoboTwin
cd "$BASE/work/overlay_policy_equivalent_pilot_v1"
timeout --signal=TERM --kill-after=20s 1500s "$PY" -B "$BASE/src/guarded_gpu_entry.py" --owner "$OWNER" --evidence "$BASE/data/reports/feedback_block_intervention_v1/s${SEED}_${ARM}_${MODE}_admission.json" "$BASE/src/run_feedback_block_intervention.py" --seed "$SEED" --arm "$ARM" --execution-mode "$MODE"
