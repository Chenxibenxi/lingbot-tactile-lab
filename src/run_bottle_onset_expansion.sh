#!/usr/bin/env bash
# Exactly two preregistered relay expert attempts, never success-cherry-picked.
set -euo pipefail
MODE=${1:?usage: run_bottle_onset_expansion.sh --dry-run|--run}
[[ "$MODE" == --dry-run || "$MODE" == --run ]] || { echo 'invalid mode' >&2; exit 2; }
BASE=/mnt/relaydata/lingbot-tactile-lab
SOURCE=/mnt/relaydata/lingbot/RoboTwin
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
OWNER=lingbot-tactile-lab/bottle-onset-expansion
SEEDS=(100013 100015)
if [[ "$MODE" == --dry-run ]]; then
  for seed in "${SEEDS[@]}"; do
    trial=relay_pickdiverse_s${seed}_onsetpilot_expert0
    bash "$BASE/src/run_expert_relay.sh" --dry-run "$trial" pick_diverse_bottles "$seed"
  done
  echo 'No GPU lease, source edit or data output during dry-run.'
  exit 0
fi
[[ "$(hostname)" == admin001-MS-7D90 ]] || { echo REFUSE_WRONG_RELAY_HOST; exit 4; }
[[ "$(nvidia-smi --list-gpus | wc -l)" == 1 ]] || { echo REFUSE_NOT_SINGLE_GPU; exit 4; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == 'NVIDIA GeForce RTX 3090' ]] || { echo REFUSE_NOT_3090; exit 4; }
[[ "$(sha256sum "$SOURCE/envs/_base_task.py" | cut -d' ' -f1)" == 448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101 ]] || { echo REFUSE_CHANGED_BASE_TASK; exit 4; }
# Do not allow an unrelated inherited diagnostic opt-in to leak privileged world geometry.
unset TACTILE_CONTACT_GEOMETRY_DIR
for seed in "${SEEDS[@]}"; do
  trial=relay_pickdiverse_s${seed}_onsetpilot_expert0
  for target in "$BASE/data/paired/$trial" "$BASE/data/paired/$trial.lock" "$BASE/data/paired/$trial.status"; do
    [[ ! -e "$target" ]] || { echo "REFUSE_REUSED_TRIAL $target" >&2; exit 3; }
  done
  # Deadline is the user's separate Strata restore. A trial must not overrun it.
  deadline=$(TZ=Asia/Shanghai date -d '2026-11-02 12:00:00' +%s)
  now=$(date +%s)
  (( now + 900 < deadline )) || { echo 'REFUSE_TOO_NEAR_STRATA_RESTORE' >&2; exit 4; }
  python3 -B "$LEASE" acquire --owner "$OWNER" --ttl 900 --priority interactive --task "single bottle onset seed $seed"
  trap 'python3 -B "$LEASE" release --owner "$OWNER" || true' EXIT
  if curl --max-time 3 -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo 'REFUSE_STRATA_RUNNING' >&2; exit 4
  fi
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || { echo 'REFUSE_OTHER_COMPUTE_JOB' >&2; exit 4; }
  set +e
  timeout --signal=TERM --kill-after=20s 660s bash "$BASE/src/run_expert_relay.sh" --run "$trial" pick_diverse_bottles "$seed"
  outcome=$?
  set -e
  echo "ONSET_EXPANSION_TRIAL seed=$seed result=$outcome"
  python3 -B "$LEASE" release --owner "$OWNER"
  trap - EXIT
  if (( outcome == 124 || outcome == 137 )); then
    echo 'ABORT_BOUNDED_TRIAL_TIMED_OUT_NO_REPLACEMENT' >&2; exit "$outcome"
  fi
  if (( outcome != 0 )) && [[ ! -e "$BASE/data/paired/$trial.status" ]]; then
    echo 'ABORT_UNCLASSIFIED_INFRASTRUCTURE_FAILURE_NO_REPLACEMENT' >&2; exit "$outcome"
  fi
done
