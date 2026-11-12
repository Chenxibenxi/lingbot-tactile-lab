#!/bin/bash
set -Eeuo pipefail
cd /mnt/relaydata/lingbot-tactile-lab
for pair in 'cwmj-full-trajectory-training-v1 cwmj_full_trajectory_training_v1' 'cwmj-full-trajectory-extension-v1 cwmj_full_trajectory_extension_v1'; do
 read -r unit name <<< "$pair"
 while true; do
  state=$(systemctl --user show "$unit.service" -p ActiveState --value)
  if [[ "$state" != active && "$state" != activating && "$state" != deactivating ]]; then break; fi
  sleep 60
 done
 /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python work/cwmj_full_trajectory_training_v1/report.py "$name"
done
