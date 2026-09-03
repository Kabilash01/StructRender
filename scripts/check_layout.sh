#!/usr/bin/env bash
# Assert the repository layout matches KB §8. Exits non-zero on the first
# missing path. Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

required_dirs=(
  configs/data configs/teacher configs/structformer configs/training configs/eval configs/deployment
  agent/prompts agent/specifications agent/workflows agent/validators
  game_project/scenes game_project/scripts game_project/assets game_project/instrumentation
  game_project/tests game_project/tests/structural
  data_pipeline/exporters
  data_pipeline/recorder
  data_pipeline/exporters/depth data_pipeline/exporters/normals data_pipeline/exporters/material
  data_pipeline/exporters/motion data_pipeline/exporters/entity_id
  data_pipeline/entity_lifecycle data_pipeline/validation data_pipeline/schemas data_pipeline/teacher_targets
  renderer/structformer renderer/baselines renderer/temporal renderer/epam renderer/style
  renderer/losses renderer/training renderer/optional/gradient_balancing renderer/optional/one_step_dit
  inference/export inference/tensorrt/builder inference/tensorrt/benchmark
  inference/cuda/temporal_warp inference/cpp_runtime/include inference/cpp_runtime/src
  eval/protocols eval/baselines eval/metrics eval/plots eval/reports
  tests/unit tests/integration tests/e2e tests/fixtures
  scripts checkpoints
)

required_files=(
  README.md pyproject.toml CMakeLists.txt .gitignore .godot-version
  docs/StructRender_KB.md CLAUDE.md
)

fail=0
for d in "${required_dirs[@]}"; do
  if [[ ! -d "$d" ]]; then echo "MISSING DIR:  $d"; fail=1; fi
done
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then echo "MISSING FILE: $f"; fail=1; fi
done

if [[ $fail -ne 0 ]]; then
  echo "Layout check FAILED."
  exit 1
fi
echo "Layout check OK."
