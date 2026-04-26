#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

OVERRIDES=()
if [[ -n "${RF_SCENE_ROOTS:-}" ]]; then
  OVERRIDES+=("rf_scene_roots=${RF_SCENE_ROOTS}")
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  training/launch.py --config rf_vggt_final_full \
  "${OVERRIDES[@]}" "$@"
