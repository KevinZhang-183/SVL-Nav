from typing import Dict, Optional, Sequence, Union

import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import draw_collision

from habitat_extensions import maps

cv2 = try_cv2_import()


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
    td_map = info_td["map"]
    td_map = maps.colorize_topdown_map(
        td_map,
        info_td["fog_of_war_mask"],
        fog_of_war_desat_amount=0.75,
    )
    td_map = habitat_maps.draw_agent(
        image=td_map,
        agent_center_coord=info_td["agent_map_coord"],
        agent_rotation=info_td["agent_angle"],
        agent_radius_px=min(td_map.shape[0:2]) // 36,
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
        scale = display_height / float(old_h)
    else:
        scale = 1.0

    if history_positions:
        from habitat_extensions import vis_overlay

        marker_radius = max(4, int(round(5 * scale)))
        td_map = vis_overlay.draw_history_markers(
            td_map,
            None,
            history_positions,
            bounds=info_td.get("bounds"),
            min_dist_m=0.35,
            color_bgr=(0, 140, 255),
            radius_px=marker_radius,
        )
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
