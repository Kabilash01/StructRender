#!/usr/bin/env bash
# M8 chain: wait for watercolor targets -> verify -> GPU smoke -> F -> F0.
set -u
cd "$(dirname "$0")/.."
PY=/home/kabilash01/miniconda3/envs/dl/bin/python
LOG=/tmp/m8_chain.log
say(){ echo "=== $* $(date '+%a %H:%M') ===" | tee -a $LOG; }
# Wait on the targets themselves, not on a log. /tmp is tmpfs here, so a reboot
# erases the generation log and a log-based wait would block forever.
until [ "$(ls data_pipeline/teacher_targets/watercolor/*/ALIGNED 2>/dev/null | wc -l)" -ge 29 ]; do sleep 120; done
n=$(ls data_pipeline/teacher_targets/watercolor/*/ALIGNED 2>/dev/null | wc -l)
say "watercolor aligned: $n/29"
if [ "$n" -lt 29 ]; then say "ABORT: not all sessions aligned — dataset refuses unvalidated targets"; echo M8_CHAIN_DONE >> $LOG; exit 1; fi
rm -rf /tmp/pytest-of-$USER
say "smoke"
$PY -m renderer.epam.train_epam --config configs/epam/model_f.json --max-steps 20 >> $LOG 2>&1 \
  || { say "ABORT: smoke failed"; echo M8_CHAIN_DONE >> $LOG; exit 1; }
for tag in f f0; do
  say "START model_$tag"
  $PY -u -m renderer.epam.train_epam --config configs/epam/model_$tag.json > /tmp/m8_train_$tag.log 2>&1
  say "DONE model_$tag exit=$?"
done
echo M8_CHAIN_DONE >> $LOG
