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
DEFAULT_WHITE_THRESHOLD = 235
DEFAULT_TRAJECTORY_MARGIN_M = 5.0
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


def _append_xz_from_position(
    points: List[Tuple[float, float]], position: Any
) -> None:
    if position is None:
        return
    if isinstance(position, dict):
        _append_xz_from_position(points, position.get("position"))
        return
    if isinstance(position, (list, tuple)):
        if len(position) >= 3 and isinstance(position[0], (int, float)):
            points.append((float(position[0]), float(position[2])))
            return
        for item in position:
            _append_xz_from_position(points, item)


def load_r2r_scene_trajectory_xz_map(
    data_path: str,
    split: str,
    scene_ids: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    """Collect xz samples from reference paths (and start/goals) per scene."""
    dataset_path = data_path.format(split=split)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"R2R dataset not found: {dataset_path}")

    out: Dict[str, List[Tuple[float, float]]] = {}
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    for episode in payload.get("episodes", []):
        scene_path = str(episode.get("scene_id", ""))
        sid = scene_id_from_path(scene_path)
        if scene_ids is not None and sid not in scene_ids:
            continue
        pts = out.setdefault(sid, [])
        _append_xz_from_position(pts, episode.get("start_position"))
        _append_xz_from_position(pts, episode.get("goals"))
        for waypoint in episode.get("reference_path") or []:
            _append_xz_from_position(pts, waypoint)
    return out


def compute_xz_bbox(
    points_xz: Sequence[Tuple[float, float]],
    margin_m: float = DEFAULT_TRAJECTORY_MARGIN_M,
) -> Optional[Dict[str, float]]:
    if not points_xz:
        return None
    xs = [float(p[0]) for p in points_xz]
    zs = [float(p[1]) for p in points_xz]
    return {
        "min_x": min(xs) - margin_m,
        "max_x": max(xs) + margin_m,
        "min_z": min(zs) - margin_m,
        "max_z": max(zs) + margin_m,
        "center_x": (min(xs) + max(xs)) / 2.0,
        "center_z": (min(zs) + max(zs)) / 2.0,
    }


def compute_hfov_for_bbox(
    bbox: Dict[str, float],
    camera_height: float,
    margin: float = 1.12,
) -> float:
    half_extent = max(
        (bbox["max_x"] - bbox["min_x"]) / 2.0,
        (bbox["max_z"] - bbox["min_z"]) / 2.0,
        1.0,
    )
    height = max(float(camera_height), 1.0)
    half_fov = math.atan((half_extent * margin) / height)
    hfov = math.degrees(2.0 * half_fov)
    return float(min(max(hfov, 30.0), 150.0))


def build_bbox_grid_mask(
    shape: Tuple[int, int],
    bounds: Dict[str, Sequence[float]],
    bbox: Dict[str, float],
) -> np.ndarray:
    lower = bounds["lower"]
    upper = bounds["upper"]
    h, w = shape
    gs_z = abs(float(upper[2]) - float(lower[2])) / max(h, 1)
    gs_x = abs(float(upper[0]) - float(lower[0])) / max(w, 1)
    gz = np.arange(h, dtype=np.float64)[:, None]
    gy = np.arange(w, dtype=np.float64)[None, :]
    zz = float(lower[2]) + (gz + 0.5) * gs_z
    xx = float(lower[0]) + (gy + 0.5) * gs_x
    return (
        (xx >= bbox["min_x"])
        & (xx <= bbox["max_x"])
        & (zz >= bbox["min_z"])
        & (zz <= bbox["max_z"])
    )


def collect_runtime_trajectory_xz(
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]],
    sim: Any = None,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if history_positions:
        for p in history_positions:
            if p is None:
                continue
            q = np.asarray(p, dtype=np.float64).reshape(-1)
            if q.size >= 3:
                points.append((float(q[0]), float(q[2])))
    if sim is not None:
        try:
            pos = sim.get_agent_state().position
            points.append((float(pos[0]), float(pos[2])))
        except Exception:
            pass
    return points


def valid_texture_pixel_mask(
    texture_rgb: np.ndarray,
    black_threshold: int = DEFAULT_BLACK_THRESHOLD,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
) -> np.ndarray:
    peak = texture_rgb.max(axis=2)
    return (peak > black_threshold) & (peak < white_threshold)


def apply_white_background(
    texture_rgb: np.ndarray,
    label_map: Optional[np.ndarray] = None,
    black_threshold: int = DEFAULT_BLACK_THRESHOLD,
) -> np.ndarray:
    """Replace Habitat void / unrendered black pixels with white."""
    out = texture_rgb.copy()
    dark = out.max(axis=2) < black_threshold
    if label_map is not None:
        dark |= label_map == ext_maps.MAP_INVALID_POINT
    out[dark] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def make_white_topdown_canvas(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.full((h, w, 3), 255, dtype=np.uint8)


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


def _navigable_near_xz(
    sim: Any, x: float, z: float, floor_y: float
) -> np.ndarray:
    pf = sim.pathfinder
    offsets = [(0.0, 0.0)]
    for radius in (0.5, 1.0, 2.0, 3.0, 5.0):
        for dx in (-radius, 0.0, radius):
            for dz in (-radius, 0.0, radius):
                if dx == 0.0 and dz == 0.0:
                    continue
                offsets.append((dx, dz))
    for dx, dz in offsets:
        for dy in (0.05, 0.25, 0.5, 0.0, -0.25):
            candidate = np.array(
                [x + dx, float(floor_y) + dy, z + dz], dtype=np.float32
            )
            if pf.is_navigable(candidate):
                return candidate
    return np.array([x, float(floor_y) + 0.05, z], dtype=np.float32)


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
    trajectory_xz_points: Optional[Sequence[Tuple[float, float]]] = None,
    trajectory_margin_m: float = DEFAULT_TRAJECTORY_MARGIN_M,
    hfov_deg: Optional[float] = None,
    black_threshold: int = DEFAULT_BLACK_THRESHOLD,
    apply_fallback: bool = False,
    apply_white_background: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    trajectory_bbox = None
    if trajectory_xz_points:
        trajectory_bbox = compute_xz_bbox(
            trajectory_xz_points, margin_m=trajectory_margin_m
        )

    if trajectory_bbox is not None:
        center = _navigable_near_xz(
            sim,
            trajectory_bbox["center_x"],
            trajectory_bbox["center_z"],
            floor_y,
        )
        if hfov_deg is None:
            hfov_deg = compute_hfov_for_bbox(trajectory_bbox, camera_height)
    else:
        center = _navigable_center(sim, floor_y)
        if hfov_deg is None:
            hfov_deg = _compute_hfov_deg(sim, floor_y, camera_height)

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
    elif apply_white_background:
        texture = apply_white_background(
            texture, label_map=label_map, black_threshold=black_threshold
        )

    lower, upper = sim.pathfinder.get_bounds()
    metadata = {
        "scene_id": None,
        "floor_y": float(floor_y),
        "map_shape": [int(label_map.shape[0]), int(label_map.shape[1])],
        "map_resolution": int(map_resolution),
        "meters_per_px": float(meters_per_px),
        "camera_height": float(camera_height),
        "hfov_deg": float(hfov_deg),
        "bounds": {
            "lower": [float(x) for x in lower],
            "upper": [float(x) for x in upper],
        },
        "camera_position": [float(x) for x in center],
        "black_fallback": bool(apply_fallback),
        "black_threshold": int(black_threshold),
        "white_background": bool(apply_white_background and not apply_fallback),
        "floor_y_source": "episode_start",
        "white_threshold": int(DEFAULT_WHITE_THRESHOLD),
        "local_bake": trajectory_bbox is not None,
        "trajectory_margin_m": float(trajectory_margin_m),
    }
    if trajectory_bbox is not None:
        metadata["trajectory_bbox"] = trajectory_bbox
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


def compose_geometry_topdown_panel(
    info_td: Dict[str, Any],
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    sim: Any = None,
    bounds: Optional[Dict[str, Sequence[float]]] = None,
) -> np.ndarray:
    label_map = info_td["map"]
    fog = info_td.get("fog_of_war_mask")
    bgr = ext_maps.colorize_topdown_map(
        label_map, fog, fog_of_war_desat_amount=0.75
    )
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
            bounds=bounds or info_td.get("bounds"),
            min_dist_m=0.35,
            color_bgr=(0, 140, 255),
            half_size_px=3,
        )
    return bgr


def compose_texture_topdown_panel(
    info_td: Dict[str, Any],
    cache_dir: str,
    scene_id: str,
    agent_floor_y: float,
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    sim: Any = None,
    floor_snap: float = DEFAULT_FLOOR_SNAP,
    trajectory_margin_m: float = DEFAULT_TRAJECTORY_MARGIN_M,
    black_threshold: int = DEFAULT_BLACK_THRESHOLD,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    use_geometry_base: bool = False,
    apply_white_background: bool = True,
) -> Optional[np.ndarray]:
    texture_rgb, meta = load_texture_for_agent(
        cache_dir, scene_id, agent_floor_y, floor_snap=floor_snap
    )
    if texture_rgb is None:
        return None

    label_map = info_td["map"]
    bounds = info_td.get("bounds") or meta.get("bounds")
    if bounds is None:
        return None

    if use_geometry_base:
        bgr = compose_geometry_topdown_panel(
            info_td,
            history_positions=None,
            sim=sim,
            bounds=bounds,
        )
    else:
        bgr = make_white_topdown_canvas(label_map.shape)

    baked_bbox = meta.get("trajectory_bbox")
    runtime_points = collect_runtime_trajectory_xz(history_positions, sim=sim)
    runtime_bbox = compute_xz_bbox(
        runtime_points, margin_m=trajectory_margin_m
    )
    if runtime_bbox is None and baked_bbox is not None:
        runtime_bbox = baked_bbox

    if texture_rgb.shape[:2] != label_map.shape:
        texture_rgb = cv2.resize(
            texture_rgb,
            (label_map.shape[1], label_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if apply_white_background and not use_geometry_base:
        texture_rgb = apply_white_background(
            texture_rgb, label_map=label_map, black_threshold=black_threshold
        )
    elif apply_white_background and use_geometry_base:
        texture_rgb = apply_white_background(
            texture_rgb, label_map=None, black_threshold=black_threshold
        )

    navigable = label_map != ext_maps.MAP_INVALID_POINT
    texture_bgr = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR)

    if runtime_bbox is not None:
        bbox_mask = build_bbox_grid_mask(label_map.shape, bounds, runtime_bbox)
        valid_tex = valid_texture_pixel_mask(
            texture_rgb,
            black_threshold=black_threshold,
            white_threshold=white_threshold,
        )
        if use_geometry_base:
            blend_mask = bbox_mask & navigable & valid_tex
        else:
            blend_mask = bbox_mask & navigable
        bgr[blend_mask] = texture_bgr[blend_mask]
    elif not use_geometry_base:
        bgr[navigable] = texture_bgr[navigable]

    overlay_values = set(range(15, 246)).union(
        {
            ext_maps.MAP_SOURCE_POINT_INDICATOR,
            ext_maps.MAP_TARGET_POINT_INDICATOR,
            ext_maps.MAP_BORDER_INDICATOR,
        }
    )
    for val in overlay_values:
        mask = label_map == val
        if np.any(mask):
            bgr[mask] = ext_maps.TOP_DOWN_MAP_COLORS[val]

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
            bounds=bounds,
            min_dist_m=0.35,
            color_bgr=(0, 140, 255),
            half_size_px=3,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def configure_bake_sensors(
    task_config: Any,
    map_resolution: int,
    camera_height: float,
    hfov_deg: float = 120.0,
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
    task_config.SIMULATOR.OVERHEAD_RGB_SENSOR.HFOV = float(hfov_deg)
    task_config.freeze()
