#!/usr/bin/env bash
# One simulation-only open-grasp intervention; never writes positive expert data.
set -euo pipefail
MODE=${1:?use --dry-run or --run}
TRIAL=${2:?unique negative trial id required}
[[ "$MODE" == --dry-run || "$MODE" == --run ]] || { echo 'bad mode' >&2; exit 2; }
[[ "$TRIAL" == relay_pickdiverse_s100000_open_grasp1 ]] || { echo 'only predeclared controlled negative trial allowed' >&2; exit 2; }
BASE=/mnt/relaydata/lingbot-tactile-lab
ENV=/mnt/relaydata/lingbot/conda/envs/lingbot-eval310
SOURCE=/mnt/relaydata/lingbot/RoboTwin
LEASE=/home/admin001/ai-video-studio/tools/gpu_lease.py
OWNER=lingbot-tactile-lab/open-grasp-negative
OUT=$BASE/data/negative/$TRIAL
OVERLAY=$BASE/work/overlay_expert
if [[ "$MODE" == --dry-run ]]; then
  echo "DRY-RUN SIMULATION_NEGATIVE task=pick_diverse_bottles seed=100000 output=$OUT"
  echo 'Strata state untouched; no source rewrite, GPU or output was used.'
  exit 0
fi
[[ "$(hostname)" == admin001-MS-7D90 ]] || { echo 'wrong relay host' >&2; exit 4; }
[[ "$(nvidia-smi --list-gpus | wc -l)" == 1 ]] || { echo 'expected one GPU' >&2; exit 4; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == 'NVIDIA GeForce RTX 3090' ]] || { echo 'not a relay 3090' >&2; exit 4; }
[[ -f "$SOURCE/envs/_base_task.py" && -f "$OVERLAY/envs/_base_task.py" && ! -L "$OVERLAY/envs/_base_task.py" ]] || { echo 'missing safe RoboTwin overlay' >&2; exit 4; }
for p in "$OUT" "$OUT.lock" "$OUT.status"; do [[ ! -e "$p" ]] || { echo "REFUSE reused path $p" >&2; exit 3; }; done
python3 -B "$LEASE" acquire --owner "$OWNER" --ttl 1200 --priority interactive --task 'one bounded controlled simulated open-grasp negative'
trap 'python3 -B "$LEASE" release --owner "$OWNER" || true' EXIT
if curl --max-time 3 -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then echo 'REFUSE Strata already running' >&2; exit 4; fi
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || { echo 'REFUSE another GPU process' >&2; exit 4; }
free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 16000 ]] || { echo 'REFUSE insufficient free VRAM' >&2; exit 4; }
ffmpeg_binary=$("$ENV/bin/python" -B -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
[[ -x "$ffmpeg_binary" ]] && "$ffmpeg_binary" -hide_banner -loglevel error -h encoder=libx264 >/dev/null
[[ -L "$BASE/work/ffmpeg-bin/ffmpeg" && "$(realpath "$BASE/work/ffmpeg-bin/ffmpeg")" == "$(realpath "$ffmpeg_binary")" ]] || { echo 'private ffmpeg shim missing/different' >&2; exit 5; }
mkdir -p "$BASE/data/negative"
(set -C; : > "$OUT.lock") || { echo 'negative writer lock already exists' >&2; exit 3; }
export PYTHONPATH="$BASE/src:$OVERLAY" ROBOTWIN_SOURCE="$SOURCE"
export FORCE_PROBE_EXPERT_DIR="$OUT/force" TACTILE_SOURCE_MACHINE_ROLE=relay_single_rtx3090
export PATH="$ENV/bin:$BASE/work/ffmpeg-bin:$PATH" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$OVERLAY"
set +e
"$ENV/bin/python" -B "$BASE/src/record_controlled_negative.py" --out "$OUT" --task pick_diverse_bottles --seed 100000 --episode 0
result=$?
set -e
printf 'exit_code=%s\n' "$result" > "$OUT.status"
echo "CONTROLLED_NEGATIVE_FINAL trial=$TRIAL exit=$result"
exit "$result"
