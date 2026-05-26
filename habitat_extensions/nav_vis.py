"""Runtime toggles for navigation visualization (overhead RGB, top-down map)."""

from __future__ import annotations

import os
from typing import Any

OVERHEAD_SENSOR_NAME = "OVERHEAD_RGB_SENSOR"


def _graphs_path(config: Any) -> str:
    graphs_file = config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE
    if os.path.isabs(graphs_file):
        return graphs_file
    return os.path.join(os.getcwd(), graphs_file)


def apply_nav_vis_config(config: Any) -> None:
    """Apply ``NAV_VIS`` flags to task sensors and measurements.

    Expects ``config`` to be defrosted. Updates ``config.TASK_CONFIG`` and
    ``config.SENSORS`` in place.
    """
    nav_vis = config.NAV_VIS
    task = config.TASK_CONFIG
    task.defrost()

    sensors = list(task.SIMULATOR.AGENT_0.SENSORS)
    if nav_vis.ENABLE_OVERHEAD_RGB:
        if OVERHEAD_SENSOR_NAME not in sensors:
            sensors.append(OVERHEAD_SENSOR_NAME)
    else:
        sensors = [s for s in sensors if s != OVERHEAD_SENSOR_NAME]
    task.SIMULATOR.AGENT_0.SENSORS = sensors

    if nav_vis.ENABLE_TEXTURE_TOPDOWN:
        task.TASK.TOP_DOWN_MAP_VLNCE.MAP_RESOLUTION = int(
            nav_vis.TEXTURE_RESOLUTION
        )

    measurements = list(task.TASK.MEASUREMENTS)
    graphs_path = _graphs_path(config)
    if nav_vis.ENABLE_TOPDOWN_MAP and os.path.exists(graphs_path):
        if "TOP_DOWN_MAP_VLNCE" not in measurements:
            measurements.append("TOP_DOWN_MAP_VLNCE")
    else:
        measurements = [m for m in measurements if m != "TOP_DOWN_MAP_VLNCE"]
        if nav_vis.ENABLE_TOPDOWN_MAP and not os.path.exists(graphs_path):
            print(
                f"[Open-Nav] NAV_VIS.ENABLE_TOPDOWN_MAP=True but missing: {graphs_path}\n"
                "  Skipping TOP_DOWN_MAP_VLNCE (SWG will use default map_size).\n"
                "  Place connectivity_graphs.pkl or set "
                "TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE.",
                flush=True,
            )
    task.TASK.MEASUREMENTS = measurements
    task.freeze()

    config.SENSORS = list(task.SIMULATOR.AGENT_0.SENSORS)
