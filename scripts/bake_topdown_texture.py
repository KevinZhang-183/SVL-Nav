#!/usr/bin/env python3
"""Bake offline top-down RGB texture maps for MP3D scenes (Scheme A)."""

from __future__ import annotations

import argparse
import os
import sys

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401

from habitat_baselines.common.environments import get_env_class
from habitat_baselines.utils.env_utils import make_env_fn

from habitat_extensions.topdown_texture import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_TEXTURE_RESOLUTION,
    bake_floor_texture,
    configure_bake_sensors,
    discover_floor_heights,
    list_mp3d_scene_ids,
    save_texture_cache,
)
from vlnce_baselines.config.default import get_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake MP3D top-down RGB texture maps for offline eval visualization."
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
        help="Bake all scenes under data/scene_datasets/mp3d/.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default="data/scene_datasets/mp3d",
        help="MP3D root directory.",
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
        help="Overhead sensor height above agent body (meters).",
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


def _scene_targets(args: argparse.Namespace) -> list:
    if args.scene:
        return [args.scene]
    if args.all:
        return list_mp3d_scene_ids(args.scenes_dir)
    raise SystemExit("Specify --scene <id> or --all")


def _build_config(args: argparse.Namespace, scene_id: str):
    opts = list(args.opts or [])
    config = get_config(args.exp_config, opts)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_id]
    config.TASK_CONFIG.DATASET.EPISODES_TO_LOAD = 1
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


def bake_scene(args: argparse.Namespace, scene_id: str) -> None:
    config = _build_config(args, scene_id)
    env = make_env_fn(config, get_env_class(config.ENV_NAME))
    try:
        env.reset()
        sim = env.get_habitat_sim()
        floor_heights = discover_floor_heights(sim)
        print(f"[bake] scene={scene_id} floors={floor_heights}")

        for floor_y in floor_heights:
            if not _needs_bake(args.cache_dir, scene_id, floor_y, args.force):
                print(f"  skip existing floor_y={floor_y:.2f}")
                continue
            texture, meta = bake_floor_texture(
                sim,
                env,
                floor_y=floor_y,
                map_resolution=args.resolution,
                camera_height=args.camera_height,
                apply_fallback=not args.no_black_fallback,
            )
            meta["scene_id"] = scene_id
            png_path, json_path = save_texture_cache(
                args.cache_dir, scene_id, floor_y, texture, meta
            )
            print(f"  saved floor_y={floor_y:.2f} -> {png_path}")
            print(f"             meta -> {json_path} shape={meta['map_shape']}")
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    scenes = _scene_targets(args)
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Baking {len(scenes)} scene(s) -> {args.cache_dir} @ {args.resolution}px")
    failed = []
    for scene_id in scenes:
        try:
            bake_scene(args, scene_id)
        except Exception as exc:
            print(f"[ERROR] scene={scene_id}: {exc}", file=sys.stderr)
            failed.append(scene_id)
    if failed:
        print(f"Failed scenes ({len(failed)}): {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
