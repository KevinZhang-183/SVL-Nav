#!/usr/bin/env bash
# Bake offline top-down texture maps for R2R scenes (2048px, 8m camera).
# Default: one floor_y per scene from episode start_position y in the dataset split.
# Camera is placed over the R2R reference-path bbox (local bake, not scene center).
# Run once on a machine with scene_datasets + GPU, then eval reads cache/topdown_texture/.

set -euo pipefail

RESOLUTION="${RESOLUTION:-2048}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-8.0}"
TRAJECTORY_MARGIN="${TRAJECTORY_MARGIN:-5.0}"
CACHE_DIR="${CACHE_DIR:-cache/topdown_texture}"
SCENE="${SCENE:-}"
SPLIT="${SPLIT:-}"

ARGS=(
  --exp-config run_OpenNav.yaml
  --cache-dir "${CACHE_DIR}"
  --resolution "${RESOLUTION}"
  --camera-height "${CAMERA_HEIGHT}"
  --trajectory-margin "${TRAJECTORY_MARGIN}"
)

if [[ -n "${SPLIT}" ]]; then
  ARGS+=(--split "${SPLIT}")
fi

if [[ -n "${SCENE}" ]]; then
  ARGS+=(--scene "${SCENE}")
else
  ARGS+=(--all)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 scripts/bake_topdown_texture.py "${ARGS[@]}" "$@"
