#!/usr/bin/env bash
# Live dashboard for whatever StructRender job is running. Ctrl-C to exit.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while true; do
  clear
  echo "===== StructRender  $(date '+%H:%M:%S') ====================================="
  # --- target generation ---
  if [ -f /tmp/generate_all.log ]; then
    done_n=$(grep -cE '=== .* exit=' /tmp/generate_all.log 2>/dev/null || echo 0)
    cur=$(grep -E '^\s*\[[0-9]+/990\]' /tmp/generate_all.log | tail -1)
    if grep -q ALL_GENERATION_DONE /tmp/generate_all.log 2>/dev/null; then
      echo "TARGET GENERATION : COMPLETE ($done_n/14 sessions)"
    else
      echo "TARGET GENERATION : session $((done_n+1))/14   $cur"
    fi
  fi
  # --- training ---
  for f in /tmp/m7_trainA.log /tmp/m7_trainB.log /tmp/m7_trainC.log /tmp/m7_trainD.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    last=$(grep -E '"epoch"' "$f" | tail -1)
    if [ -n "$last" ]; then
      ep=$(echo "$last" | grep -oE '"epoch": [0-9]+' | grep -oE '[0-9]+')
      vl=$(echo "$last" | grep -oE '"val_l1": [0-9.]+' | cut -d' ' -f2)
      tl=$(echo "$last" | grep -oE '"train_loss": [0-9.]+' | cut -d' ' -f2)
      printf "%-16s : epoch %-3s  train_loss %-10s val_l1 %s\n" "$name" "${ep:-?}" "${tl:0:8}" "${vl:-—}"
    else
      printf "%-16s : starting...\n" "$name"
    fi
  done
  echo "-------------------------------------------------------------------"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader | awk -F', ' '{printf "GPU  : %s / %s   util %s   %s\n",$1,$2,$3,$4}'
  free -g | awk '/^Mem:/{printf "RAM  : %s used / %s total   %s available\n",$3,$2,$7}'
  echo "==================================================================="
  sleep 10
done
