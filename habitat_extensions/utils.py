from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import draw_collision

from habitat_extensions import maps

cv2 = try_cv2_import()


def _crop_topdown_labels(
    label_map: np.ndarray,
    agent_coord: Tuple[int, int],
    margin_px: int = 48,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Crop label map to navigable content; return crop origin for world-to-grid projection."""
    valid = label_map != maps.MAP_INVALID_POINT
    if not np.any(valid):
        return label_map, agent_coord, (0, 0)

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    r0 = max(0, int(rows[0]) - margin_px)
    r1 = min(label_map.shape[0], int(rows[-1]) + margin_px + 1)
    c0 = max(0, int(cols[0]) - margin_px)
    c1 = min(label_map.shape[1], int(cols[-1]) + margin_px + 1)
    cropped = label_map[r0:r1, c0:c1].copy()
    ax, ay = int(agent_coord[0]), int(agent_coord[1])
    agent_coord = (ax - r0, ay - c0)
    return cropped, agent_coord, (r0, c0)


def _center_on_square_canvas(
    bgr: np.ndarray, fill: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Pad image to a square canvas so the map sits in the center."""
    h, w = bgr.shape[0:2]
    side = max(h, w)
    if h == side and w == side:
        return bgr
    canvas = np.full((side, side, 3), fill, dtype=bgr.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = bgr
    return canvas


def _topdown_map_key(info: Dict) -> Optional[str]:
    if "top_down_map_vlnce" in info:
        return "top_down_map_vlnce"
    if "top_down_map" in info:
        return "top_down_map"
    return None


def _render_topdown_panel(
    info: Dict,
    map_k: str,
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    display_height: Optional[int] = None,
) -> np.ndarray:
    """Colorize geometric top-down map with agent pose and optional history markers."""
    info_td = info[map_k]
    label_map = info_td["map"]
    full_grid_shape = label_map.shape[0:2]
    agent_coord = tuple(info_td["agent_map_coord"])
    bounds = info_td.get("bounds")

    label_map, agent_coord, crop_origin = _crop_topdown_labels(label_map, agent_coord)
    fog_mask = info_td.get("fog_of_war_mask")
    if fog_mask is not None:
        r0, c0 = crop_origin
        r1 = r0 + label_map.shape[0]
        c1 = c0 + label_map.shape[1]
        fog_mask = fog_mask[r0:r1, c0:c1]

    td_map = maps.colorize_topdown_map(
        label_map,
        fog_mask,
        fog_of_war_desat_amount=0.75,
    )
    td_map = habitat_maps.draw_agent(
        image=td_map,
        agent_center_coord=agent_coord,
        agent_rotation=info_td["agent_angle"],
        agent_radius_px=min(td_map.shape[0:2]) // 36,
    )

    if history_positions:
        from habitat_extensions import vis_overlay

        marker_radius = max(4, int(round(5)))
        td_map = vis_overlay.draw_history_markers(
            td_map,
            None,
            history_positions,
            bounds=bounds,
            full_grid_shape=full_grid_shape,
            crop_origin=crop_origin,
            min_dist_m=0.35,
            color_bgr=(0, 140, 255),
            radius_px=marker_radius,
        )

    if td_map.shape[1] < td_map.shape[0]:
        td_map = np.rot90(td_map, 1)
    if td_map.shape[0] > td_map.shape[1]:
        td_map = np.rot90(td_map, 1)

    old_h, old_w, _ = td_map.shape
    if display_height is not None and display_height > 0 and display_height != old_h:
        top_down_width = int(float(display_height) / old_h * old_w)
        td_map = cv2.resize(
            td_map,
            (top_down_width, display_height),
            interpolation=cv2.INTER_AREA,
        )
    elif display_height is None:
        td_map = _center_on_square_canvas(td_map)

    return td_map


def observations_to_image(
    observation: Dict,
    info: Dict,
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    include_topdown_map: bool = True,
    include_egocentric: bool = True,
) -> np.ndarray:
    r"""Generate visualization frame from observation and info.

    When ``include_egocentric=False``, returns only the geometric top-down map
    (no RGB/Depth panels).
    """
    map_k = _topdown_map_key(info) if include_topdown_map else None

    if not include_egocentric:
        if map_k is None:
            raise ValueError(
                "include_egocentric=False requires top_down_map in info"
            )
        return _render_topdown_panel(
            info,
            map_k,
            history_positions=history_positions,
            display_height=None,
        )

    egocentric_view = []
    observation_size = -1

    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if map_k is not None:
        td_map = _render_topdown_panel(
            info,
            map_k,
            history_positions=history_positions,
            display_height=observation_size,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame
