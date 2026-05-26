#!/usr/bin/env bash
# Bake offline top-down texture maps for all MP3D scenes (2048px).
# Run once on a machine with scene_datasets + GPU, then eval reads cache/topdown_texture/.

set -euo pipefail

RESOLUTION="${RESOLUTION:-2048}"
CACHE_DIR="${CACHE_DIR:-cache/topdown_texture}"
SCENE="${SCENE:-}"

ARGS=(
  --exp-config run_OpenNav.yaml
  --cache-dir "${CACHE_DIR}"
  --resolution "${RESOLUTION}"
)

if [[ -n "${SCENE}" ]]; then
  ARGS+=(--scene "${SCENE}")
else
  ARGS+=(--all)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/bake_topdown_texture.py "${ARGS[@]}" "$@"
