from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import draw_collision

from habitat_extensions import maps

cv2 = try_cv2_import()


def _overhead_rgb_observation_key(observation: Dict) -> Optional[str]:
    for key in observation:
        kl = key.lower()
        if "overhead" in kl and "rgb" in kl:
            return key
    return None


def observations_to_image(
    observation: Dict,
    info: Dict,
    sim: Any = None,
    history_positions: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    include_overhead_rgb: bool = True,
    include_topdown_map: bool = True,
    include_texture_topdown: bool = False,
    scene_id: Optional[str] = None,
    agent_floor_y: Optional[float] = None,
    texture_cache_dir: Optional[str] = None,
    texture_floor_snap: float = 0.25,
    texture_trajectory_margin_m: float = 5.0,
    texture_black_threshold: int = 15,
    texture_white_threshold: int = 235,
) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
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

    oh_key = _overhead_rgb_observation_key(observation)
    if include_overhead_rgb and oh_key is not None:
        if observation_size == -1:
            observation_size = observation[oh_key].shape[0]
        oh = observation[oh_key][:, :, :3].astype(np.uint8)
        oh = cv2.resize(
            oh,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(oh)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if include_topdown_map and map_k is not None:
        info_td = info[map_k]
        td_map = None
        if (
            include_texture_topdown
            and scene_id
            and agent_floor_y is not None
            and texture_cache_dir
        ):
            from habitat_extensions import topdown_texture

            td_map = topdown_texture.compose_texture_topdown_panel(
                info_td,
                texture_cache_dir,
                scene_id,
                agent_floor_y,
                history_positions=history_positions,
                sim=sim,
                floor_snap=texture_floor_snap,
                trajectory_margin_m=texture_trajectory_margin_m,
                black_threshold=texture_black_threshold,
                white_threshold=texture_white_threshold,
            )

        if td_map is None:
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
                agent_radius_px=min(td_map.shape[0:2]) // 24,
            )
            if sim is not None and history_positions:
                from habitat_extensions import vis_overlay

                td_map = vis_overlay.draw_history_markers(
                    td_map,
                    sim,
                    history_positions,
                    bounds=info_td.get("bounds"),
                    min_dist_m=0.35,
                    color_bgr=(0, 140, 255),
                    half_size_px=3,
                )
            elif history_positions:
                from habitat_extensions import vis_overlay

                td_map = vis_overlay.draw_history_markers(
                    td_map,
                    None,
                    history_positions,
                    bounds=info_td.get("bounds"),
                    min_dist_m=0.35,
                    color_bgr=(0, 140, 255),
                    half_size_px=3,
                )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame
