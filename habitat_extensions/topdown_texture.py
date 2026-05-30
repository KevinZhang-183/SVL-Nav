"""Offline baked top-down RGB texture maps aligned with NavMesh top-down grids."""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps

from habitat_extensions import maps as ext_maps

cv2 = try_cv2_import()

DEFAULT_TEXTURE_RESOLUTION = 2048
DEFAULT_CAMERA_HEIGHT = 8.0
DEFAULT_FLOOR_SNAP = 0.25
DEFAULT_BLACK_THRESHOLD = 15
OVERHEAD_OBS_KEYS = ("overhead_rgb", "overhead_rgb_sensor")


def scene_id_from_path(scene_path: str) -> str:
    normalized = scene_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 2 and parts[-1].endswith((".glb", ".ply")):
        return parts[-2]
    if parts:
        return os.path.splitext(parts[-1])[0]
    return normalized


def cache_basename(scene_id: str, floor_y: float) -> str:
    return f"{scene_id}_y{floor_y:.2f}"


def cache_paths(
    cache_dir: str, scene_id: str, floor_y: float
) -> Tuple[str, str]:
    base = cache_basename(scene_id, floor_y)
    return (
        os.path.join(cache_dir, f"{base}.png"),
        os.path.join(cache_dir, f"{base}.json"),
    )


def snap_floor_y(y: float, floor_snap: float = DEFAULT_FLOOR_SNAP) -> float:
    return round(float(y) / floor_snap) * floor_snap


def load_r2r_scene_floor_y_map(
    data_path: str,
    split: str,
    floor_snap: float = DEFAULT_FLOOR_SNAP,
    scene_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Map scene_id -> one floor_y from the first episode start_position in R2R json."""
    dataset_path = data_path.format(split=split)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"R2R dataset not found: {dataset_path}")

    out: Dict[str, float] = {}
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    for episode in payload.get("episodes", []):
        scene_path = str(episode.get("scene_id", ""))
        sid = scene_id_from_path(scene_path)
        if scene_ids is not None and sid not in scene_ids:
            continue
        if sid in out:
            continue
        start_position = episode.get("start_position")
        if not start_position or len(start_position) < 2:
            continue
        out[sid] = snap_floor_y(start_position[1], floor_snap=floor_snap)
    return out


def list_mp3d_scene_ids(scenes_dir: str) -> List[str]:
    if not os.path.isdir(scenes_dir):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(scenes_dir)):
        scene_path = os.path.join(scenes_dir, name)
        if not os.path.isdir(scene_path):
            continue
        glb = os.path.join(scene_path, f"{name}.glb")
        if os.path.isfile(glb):
            out.append(name)
    return out


def discover_floor_heights(
    sim: Any,
    floor_snap: float = DEFAULT_FLOOR_SNAP,
    num_samples: int = 300,
) -> List[float]:
    """Legacy NavMesh sampling; default baking uses episode start y instead."""
    pf = sim.pathfinder
    heights: List[float] = []
    for _ in range(num_samples):
        try:
            pt = pf.get_random_navigable_point()
        except Exception:
            break
        if pt is None:
            continue
        y = float(pt[1])
        snapped = round(y / floor_snap) * floor_snap
        heights.append(snapped)
    if not heights:
        state = sim.get_agent_state()
        y = round(float(state.position[1]) / floor_snap) * floor_snap
        heights.append(y)
    return sorted(set(heights))


def _navigable_center(sim: Any, floor_y: float) -> np.ndarray:
    pf = sim.pathfinder
    lower, upper = pf.get_bounds()
    cx = (float(lower[0]) + float(upper[0])) / 2.0
    cz = (float(lower[2]) + float(upper[2])) / 2.0
    for dy in (0.0, 0.25, 0.5, 1.0, -0.25, -0.5):
        candidate = np.array([cx, floor_y + dy, cz], dtype=np.float32)
        if pf.is_navigable(candidate):
            return candidate
    try:
        pt = pf.get_random_navigable_point()
        if pt is not None:
            return np.asarray(pt, dtype=np.float32)
    except Exception:
        pass
    return np.array([cx, floor_y + 0.05, cz], dtype=np.float32)


def _compute_hfov_deg(
    sim: Any, floor_y: float, camera_height: float, margin: float = 1.08
) -> float:
    lower, upper = sim.pathfinder.get_bounds()
    dx = float(upper[0]) - float(lower[0])
    dz = float(upper[2]) - float(lower[2])
    max_extent = max(dx, dz, 1.0)
    height = max(float(camera_height), 1.0)
    half_fov = math.atan((max_extent / 2.0) / height)
    hfov = math.degrees(2.0 * half_fov) * margin
    return float(min(max(hfov, 30.0), 150.0))


def _extract_overhead_rgb(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in OVERHEAD_OBS_KEYS:
        if key in obs:
            return np.asarray(obs[key][:, :, :3], dtype=np.uint8)
    for key, val in obs.items():
        kl = key.lower()
        if "overhead" in kl and "rgb" in kl:
            return np.asarray(val[:, :, :3], dtype=np.uint8)
    return None


def apply_black_fallback(
    texture_rgb: np.ndarray,
    label_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    threshold: int = DEFAULT_BLACK_THRESHOLD,
) -> np.ndarray:
    gray_bgr = ext_maps.colorize_topdown_map(
        label_map, fog_of_war_mask, fog_of_war_desat_amount=0.75
    )
    gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)
    out = texture_rgb.copy()
    if out.shape[:2] != gray_rgb.shape[:2]:
        gray_rgb = cv2.resize(
            gray_rgb,
            (out.shape[1], out.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    dark = texture_rgb.max(axis=2) < threshold
    out[dark] = gray_rgb[dark]
    return out


def bake_floor_texture(
    sim: Any,
    env: Any,
    floor_y: float,
    map_resolution: int = DEFAULT_TEXTURE_RESOLUTION,
    camera_height: float = DEFAULT_CAMERA_HEIGHT,
    black_threshold: int = DEFAULT_BLACK_THRESHOLD,
    apply_fallback: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    center = _navigable_center(sim, floor_y)
    center[1] = float(floor_y) + 0.05
    rotation = np.quaternion(1, 0, 0, 0)

    sim.set_agent_state(center, rotation)
    meters_per_px = habitat_maps.calculate_meters_per_pixel(map_resolution, sim)
    label_map = ext_maps.get_top_down_map(sim, map_resolution, meters_per_px)

    obs = env.get_observation_at(
        center.tolist(),
        rotation,
        keep_agent_at_new_pose=True,
    )
    texture = _extract_overhead_rgb(obs)
    if texture is None:
        raise RuntimeError(
            "Overhead RGB observation missing. Ensure OVERHEAD_RGB_SENSOR is enabled."
        )

    if texture.shape[0] != label_map.shape[0] or texture.shape[1] != label_map.shape[1]:
        texture = cv2.resize(
            texture,
            (label_map.shape[1], label_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if apply_fallback:
        texture = apply_black_fallback(
            texture, label_map, fog_of_war_mask=None, threshold=black_threshold
        )

    lower, upper = sim.pathfinder.get_bounds()
    metadata = {
        "scene_id": None,
        "floor_y": float(floor_y),
        "map_shape": [int(label_map.shape[0]), int(label_map.shape[1])],
        "map_resolution": int(map_resolution),
        "meters_per_px": float(meters_per_px),
        "camera_height": float(camera_height),
        "hfov_deg": float(_compute_hfov_deg(sim, floor_y, camera_height)),
        "bounds": {
            "lower": [float(x) for x in lower],
            "upper": [float(x) for x in upper],
        },
        "camera_position": [float(x) for x in center],
        "black_fallback": bool(apply_fallback),
        "black_threshold": int(black_threshold),
        "floor_y_source": "episode_start",
    }
    return texture, metadata


def save_texture_cache(
    cache_dir: str,
    scene_id: str,
    floor_y: float,
    texture_rgb: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    png_path, json_path = cache_paths(cache_dir, scene_id, floor_y)
    metadata = dict(metadata)
    metadata["scene_id"] = scene_id
    cv2.imwrite(png_path, cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return png_path, json_path


def load_texture_metadata(
    cache_dir: str, scene_id: str, floor_y: float
) -> Optional[Dict[str, Any]]:
    _, json_path = cache_paths(cache_dir, scene_id, floor_y)
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_floor_y(
    cache_dir: str, scene_id: str, agent_y: float, floor_snap: float = DEFAULT_FLOOR_SNAP
) -> Optional[float]:
    pattern = os.path.join(cache_dir, f"{scene_id}_y*.json")
    candidates: List[float] = []
    for json_path in glob.glob(pattern):
        base = os.path.basename(json_path)
        try:
            y_str = base[len(scene_id) + 2 : -5]
            candidates.append(float(y_str))
        except ValueError:
            continue
    if not candidates:
        return None
    target = round(float(agent_y) / floor_snap) * floor_snap
    return min(candidates, key=lambda y: abs(y - target))


def load_texture_for_agent(
    cache_dir: str,
    scene_id: str,
    agent_y: float,
    floor_snap: float = DEFAULT_FLOOR_SNAP,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    floor_y = find_best_floor_y(cache_dir, scene_id, agent_y, floor_snap=floor_snap)
    if floor_y is None:
        return None, None
    png_path, json_path = cache_paths(cache_dir, scene_id, floor_y)
    if not os.path.isfile(png_path) or not os.path.isfile(json_path):
        return None, None
    bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return rgb, meta


def overlay_labels_on_texture(
    texture_rgb: np.ndarray,
    label_map: np.ndarray,
) -> np.ndarray:
    bgr = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR)
    h, w = label_map.shape
    if bgr.shape[0] != h or bgr.shape[1] != w:
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay_values = set(
        range(15, 246),
    ).union(
        {
            ext_maps.MAP_SOURCE_POINT_INDICATOR,
            ext_maps.MAP_TARGET_POINT_INDICATOR,
            ext_maps.MAP_BORDER_INDICATOR,
        }
    )
    for val in overlay_values:
        mask = label_map == val
        if not np.any(mask):
            continue
        bgr[mask] = ext_maps.TOP_DOWN_MAP_COLORS[val]
    return bgr


def compose_texture_topdown_panel(
    info_td: Dict[str, Any],
    cache_dir: str,
    scene_id: str,
    agent_floor_y: float,
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    sim: Any = None,
    floor_snap: float = DEFAULT_FLOOR_SNAP,
) -> Optional[np.ndarray]:
    texture_rgb, meta = load_texture_for_agent(
        cache_dir, scene_id, agent_floor_y, floor_snap=floor_snap
    )
    if texture_rgb is None:
        return None

    label_map = info_td["map"]
    fog = info_td.get("fog_of_war_mask")
    bgr = overlay_labels_on_texture(texture_rgb, label_map)

    if fog is not None:
        desat = ext_maps.colorize_topdown_map(label_map, fog, fog_of_war_desat_amount=0.75)
        fog_mask = label_map != ext_maps.MAP_INVALID_POINT
        bgr[fog_mask] = (
            0.35 * bgr[fog_mask].astype(np.float32)
            + 0.65 * desat[fog_mask].astype(np.float32)
        ).astype(np.uint8)

    bgr = habitat_maps.draw_agent(
        image=bgr,
        agent_center_coord=info_td["agent_map_coord"],
        agent_rotation=info_td["agent_angle"],
        agent_radius_px=min(bgr.shape[0:2]) // 24,
    )

    if history_positions:
        from habitat_extensions import vis_overlay

        bgr = vis_overlay.draw_history_markers(
            bgr,
            sim,
            history_positions,
            bounds=info_td.get("bounds") or meta.get("bounds"),
            min_dist_m=0.35,
            color_bgr=(0, 140, 255),
            half_size_px=3,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def configure_bake_sensors(
    task_config: Any,
    map_resolution: int,
    camera_height: float,
) -> None:
    task_config.defrost()
    sensors = list(task_config.SIMULATOR.AGENT_0.SENSORS)
    if "OVERHEAD_RGB_SENSOR" not in sensors:
        sensors.append("OVERHEAD_RGB_SENSOR")
    task_config.SIMULATOR.AGENT_0.SENSORS = sensors
    task_config.SIMULATOR.OVERHEAD_RGB_SENSOR.WIDTH = int(map_resolution)
    task_config.SIMULATOR.OVERHEAD_RGB_SENSOR.HEIGHT = int(map_resolution)
    task_config.SIMULATOR.OVERHEAD_RGB_SENSOR.POSITION = [
        0.0,
        float(camera_height),
        0.0,
    ]
    task_config.SIMULATOR.OVERHEAD_RGB_SENSOR.ORIENTATION = [
        -1.5707963267948966,
        0.0,
        0.0,
    ]
    task_config.SIMULATOR.OVERHEAD_RGB_SENSOR.HFOV = 120.0
    task_config.freeze()
