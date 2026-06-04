"""Runtime toggles for geometric top-down map visualization."""

from __future__ import annotations

import os
from typing import Any


def _graphs_path(config: Any) -> str:
    graphs_file = config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE
    if os.path.isabs(graphs_file):
        return graphs_file
    return os.path.join(os.getcwd(), graphs_file)


def apply_nav_vis_config(config: Any) -> None:
    """Apply ``NAV_VIS.ENABLE_TOPDOWN_MAP`` to task measurements.

    Expects ``config`` to be defrosted. Updates ``config.TASK_CONFIG`` in place.
    """
    nav_vis = config.NAV_VIS
    task = config.TASK_CONFIG
    task.defrost()

    measurements = list(task.TASK.MEASUREMENTS)
    graphs_path = _graphs_path(config)
    if nav_vis.ENABLE_TOPDOWN_MAP and os.path.exists(graphs_path):
        if "TOP_DOWN_MAP_VLNCE" not in measurements:
            measurements.append("TOP_DOWN_MAP_VLNCE")
        td_cfg = task.TASK.TOP_DOWN_MAP_VLNCE
        td_cfg.MAP_RESOLUTION = int(nav_vis.MAP_RESOLUTION)
        td_cfg.DRAW_FIXED_WAYPOINTS = False
        td_cfg.DRAW_MP3D_AGENT_PATH = False
    else:
        measurements = [m for m in measurements if m != "TOP_DOWN_MAP_VLNCE"]
        if nav_vis.ENABLE_TOPDOWN_MAP and not os.path.exists(graphs_path):
            print(
                f"[Open-Nav] NAV_VIS.ENABLE_TOPDOWN_MAP=True but missing: {graphs_path}\n"
                "  Skipping TOP_DOWN_MAP_VLNCE measurement.\n"
                "  Place connectivity_graphs.pkl or set "
                "TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE.",
                flush=True,
            )
    task.TASK.MEASUREMENTS = measurements
    task.freeze()
