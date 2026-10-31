#!/usr/bin/env bash
# Single-writer, relay 3090 ONLY. Never auto-start behind another GPU process.
set -uo pipefail
MODE=${1:?usage: run_expert_relay.sh --dry-run|--run trial_id task seed}
TRIAL=${2:?trial_id required}
TASK=${3:?task required}
SEED=${4:?seed required}
[[ "$TRIAL" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid trial id'; exit 2; }
[[ "$TASK" == lift_pot || "$TASK" == click_bell ]] || { echo 'unsupported pilot task'; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'seed must be nonnegative integer'; exit 2; }
BASE=/mnt/relaydata/lingbot-tactile-lab
SOURCE=/mnt/relaydata/lingbot/RoboTwin
ENV=/mnt/relaydata/lingbot/conda/envs/lingbot-eval310
OUT=$BASE/data/paired/$TRIAL
OVERLAY=$BASE/work/overlay_expert
STATUS=$BASE/data/paired/$TRIAL.status
LOCK=$BASE/data/paired/$TRIAL.lock
if [[ "$MODE" == --dry-run ]]; then
  echo "DRY-RUN host=$(hostname) GPU=0 task=$TASK seed=$SEED output=$OUT"
  echo "Original RoboTwin remains read-only; no GPU/job started."
  exit 0
fi
[[ "$MODE" == --run ]] || { echo 'mode must be --dry-run or --run'; exit 2; }
[[ "$(hostname)" == "admin001-MS-7D90" ]] || { echo 'BLOCKED: not the approved relay workstation'; exit 4; }
[[ "$(nvidia-smi --list-gpus | wc -l)" -eq 1 ]] || { echo 'BLOCKED: expected one relay GPU'; exit 4; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "NVIDIA GeForce RTX 3090" ]] || { echo 'BLOCKED: relay GPU identity differs'; exit 4; }
[[ -f "$SOURCE/envs/_base_task.py" && -x "$ENV/bin/python" ]] || { echo 'source/env unavailable'; exit 2; }
[[ ! -e "$OUT" && ! -e "$LOCK" && ! -e "$STATUS" ]] || { echo "REFUSE reuse existing trial $TRIAL"; exit 3; }
# Treat ANY active compute job as an ownership conflict, not just >16GB VRAM.
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo 'BLOCKED: relay 3090 has a compute process; do not evict it.'; exit 4
fi
free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 16000 ]] || { echo "BLOCKED: only ${free_mib:-unknown} MiB GPU memory free"; exit 4; }
mkdir -p "$BASE/data/paired" "$BASE/work"
if [[ ! -f "$OVERLAY/envs/_base_task.py" ]]; then
  "$ENV/bin/python" -B "$BASE/src/make_expert_overlay.py" --source "$SOURCE" --out "$OVERLAY" --create || exit 5
fi
"$ENV/bin/python" - "$OVERLAY/envs/_base_task.py" <<'PY' || exit 5
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
assert not p.is_symlink() and 'self._force_probe.record_saved_frame' in s
PY
(set -C; : > "$LOCK") || { echo "REFUSE concurrent writer $TRIAL"; exit 3; }
export PATH="$ENV/bin:$PATH" CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BASE/src:$OVERLAY"
export ROBOTWIN_SOURCE="$SOURCE"
export FORCE_PROBE_EXPERT_DIR="$OUT/force"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$OVERLAY" || exit 5
"$ENV/bin/python" -B "$BASE/src/record_expert_demo.py" --task "$TASK" --seed "$SEED" --episode 0 --out "$OUT"
code=$?
printf 'exit_code=%s\n' "$code" > "$STATUS"
echo "RELAY_EXPERT_FINAL trial=$TRIAL code=$code"
exit "$code"
