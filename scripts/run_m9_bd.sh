#!/usr/bin/env bash
# M9: retrain ladder rungs B and D on ladder-v3 data, both through train_temporal.
set -u
cd "$(dirname "$0")/.."
PY=/home/kabilash01/miniconda3/envs/dl/bin/python
LOG=/tmp/m9_bd.log
for m in b d; do
  echo "=== START model_$m $(date '+%a %H:%M') ===" >> $LOG
  $PY -u -m renderer.training.train_temporal --config configs/ladder_v3/model_$m.json --mode train > /tmp/m9_train_$m.log 2>&1
  echo "=== DONE model_$m exit=$? $(date '+%a %H:%M') ===" >> $LOG
done
echo M9_BD_DONE >> $LOG
