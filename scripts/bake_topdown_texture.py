#!/usr/bin/env python3
"""Bake offline top-down RGB texture maps for MP3D scenes (Scheme A)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401

from habitat_baselines.common.environments import get_env_class
from habitat_baselines.utils.env_utils import make_env_fn

from habitat_extensions.topdown_texture import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_FLOOR_SNAP,
    DEFAULT_TEXTURE_RESOLUTION,
    bake_floor_texture,
    configure_bake_sensors,
    load_r2r_scene_floor_y_map,
    save_texture_cache,
    snap_floor_y,
)
from vlnce_baselines.config.default import get_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bake MP3D top-down RGB texture maps for offline eval visualization. "
            "Default: one floor_y per scene from R2R episode start y, 8m camera, 2048px."
        )
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="run_OpenNav.yaml",
        help="Experiment yaml (for Habitat task/sim paths).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Single MP3D scene id (e.g. 17DRP5sb8fy). Omit with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Bake all scenes appearing in the R2R dataset split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="R2R split for episode start y lookup (default: TASK_CONFIG.DATASET.SPLIT).",
    )
    parser.add_argument(
        "--floor-y",
        type=float,
        default=None,
        help="Manual floor_y override (meters). Skips episode start y lookup.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache/topdown_texture",
        help="Output directory for PNG + JSON.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_TEXTURE_RESOLUTION,
        help="Top-down map max resolution (default 2048).",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Overhead sensor height above agent body (default 8.0 meters).",
    )
    parser.add_argument(
        "--floor-snap",
        type=float,
        default=DEFAULT_FLOOR_SNAP,
        help="Snap episode start y to this step (default 0.25).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-bake even if cache files exist.",
    )
    parser.add_argument(
        "--no-black-fallback",
        action="store_true",
        help="Disable NavMesh gray fallback for dark pixels.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Extra config overrides (habitat-style).",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace):
    opts = list(args.opts or [])
    return get_config(args.exp_config, opts)


def _resolve_split(args: argparse.Namespace, config) -> str:
    if args.split:
        return args.split
    return str(config.TASK_CONFIG.DATASET.SPLIT)


def _scene_targets(args: argparse.Namespace, floor_y_map: Dict[str, float]) -> List[str]:
    if args.scene:
        return [args.scene]
    if args.all:
        return sorted(floor_y_map.keys())
    raise SystemExit("Specify --scene <id> or --all")


def _build_floor_y_map(
    args: argparse.Namespace, config, scene_ids: Optional[List[str]]
) -> Dict[str, float]:
    split = _resolve_split(args, config)
    dataset = config.TASK_CONFIG.DATASET

    if args.floor_y is not None:
        floor_y = snap_floor_y(args.floor_y, floor_snap=args.floor_snap)
        if scene_ids is not None:
            return {sid: floor_y for sid in scene_ids}
        dataset_scenes = load_r2r_scene_floor_y_map(
            dataset.DATA_PATH,
            split,
            floor_snap=args.floor_snap,
            scene_ids=None,
        )
        return {sid: floor_y for sid in dataset_scenes}

    return load_r2r_scene_floor_y_map(
        dataset.DATA_PATH,
        split,
        floor_snap=args.floor_snap,
        scene_ids=scene_ids,
    )


def _build_config(args: argparse.Namespace, scene_id: str, config):
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = _resolve_split(args, config)
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_id]
    # Load all split episodes, then filter by CONTENT_SCENES. EPISODES_TO_LOAD=1
    # would only read the json head and leave most scenes with an empty list.
    config.TASK_CONFIG.DATASET.EPISODES_TO_LOAD = 0
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    configure_bake_sensors(
        config.TASK_CONFIG,
        map_resolution=args.resolution,
        camera_height=args.camera_height,
    )
    config.SENSORS = list(config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS)
    config.freeze()
    return config


def _needs_bake(cache_dir: str, scene_id: str, floor_y: float, force: bool) -> bool:
    if force:
        return True
    from habitat_extensions.topdown_texture import cache_paths

    png_path, json_path = cache_paths(cache_dir, scene_id, floor_y)
    return not (os.path.isfile(png_path) and os.path.isfile(json_path))


def bake_scene(
    args: argparse.Namespace,
    config,
    scene_id: str,
    floor_y: float,
) -> None:
    config = _build_config(args, scene_id, config)
    env = make_env_fn(config, get_env_class(config.ENV_NAME))
    try:
        env.reset()
        sim = env.get_habitat_sim()
        print(
            f"[bake] scene={scene_id} floor_y={floor_y:.2f} "
            f"(episode_start_y) camera={args.camera_height}m res={args.resolution}"
        )

        if not _needs_bake(args.cache_dir, scene_id, floor_y, args.force):
            print("  skip existing cache")
            return

        texture, meta = bake_floor_texture(
            sim,
            env,
            floor_y=floor_y,
            map_resolution=args.resolution,
            camera_height=args.camera_height,
            apply_fallback=not args.no_black_fallback,
        )
        meta["scene_id"] = scene_id
        meta["floor_y_source"] = "manual" if args.floor_y is not None else "episode_start"
        png_path, json_path = save_texture_cache(
            args.cache_dir, scene_id, floor_y, texture, meta
        )
        print(f"  saved -> {png_path}")
        print(f"         meta -> {json_path} shape={meta['map_shape']}")
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    base_config = _load_config(args)

    requested_scenes = [args.scene] if args.scene else None
    floor_y_map = _build_floor_y_map(args, base_config, requested_scenes)
    scenes = _scene_targets(args, floor_y_map)

    missing = [sid for sid in scenes if sid not in floor_y_map]
    if missing:
        split = _resolve_split(args, base_config)
        raise SystemExit(
            "No episode start y found for scene(s) "
            f"{missing} in split={split}. Use --floor-y to override."
        )

    os.makedirs(args.cache_dir, exist_ok=True)
    split = _resolve_split(args, base_config)
    print(
        f"Baking {len(scenes)} scene(s) from split={split} -> "
        f"{args.cache_dir} @ {args.resolution}px, camera={args.camera_height}m"
    )

    failed = []
    for scene_id in scenes:
        try:
            scene_config = _load_config(args)
            bake_scene(args, scene_config, scene_id, floor_y_map[scene_id])
        except Exception as exc:
            print(f"[ERROR] scene={scene_id}: {exc}", file=sys.stderr)
            failed.append(scene_id)
    if failed:
        print(f"Failed scenes ({len(failed)}): {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
