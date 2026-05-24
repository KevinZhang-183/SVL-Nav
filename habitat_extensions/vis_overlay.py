"""Visualization helpers for top-down maps (history markers, etc.)."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from habitat.core.simulator import Simulator

from habitat_extensions import maps as ext_maps


def draw_history_markers(
    bgr: np.ndarray,
    sim: Optional[Simulator],
    positions_xyz: Sequence[Union[np.ndarray, Sequence[float]]],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    min_dist_m: float = 0.35,
    color_bgr: Tuple[int, int, int] = (0, 140, 255),
    half_size_px: int = 3,
) -> np.ndarray:
    r"""Draw subsampled visited positions as small squares on an already colorized
    top-down BGR image (after ``colorize_topdown_map`` / ``draw_agent``).

    Uses Habitat ``to_grid`` when ``sim`` is provided. When the environment runs
    in subprocess workers (no direct sim handle), it falls back to
    ``static_to_grid`` with metric ``bounds``.
    """
    out = bgr.copy()
    if not positions_xyz:
        return out

    h, w = out.shape[0:2]
    kept: List[np.ndarray] = []
    last: Optional[np.ndarray] = None
    for p in positions_xyz:
        if p is None:
            continue
        q = np.asarray(p, dtype=np.float64).reshape(-1)
        if q.size < 3:
            continue
        xz = q[[0, 2]]
        if last is None or float(np.linalg.norm(xz - last[[0, 2]])) >= min_dist_m:
            kept.append(q[:3])
            last = q[:3]

    for q in kept:
        if sim is not None:
            from habitat.utils.visualizations import maps as habitat_maps

            gx, gy = habitat_maps.to_grid(
                float(q[2]),
                float(q[0]),
                (h, w),
                sim,
            )
        elif bounds is not None:
            gx, gy = ext_maps.static_to_grid(
                float(q[2]),
                float(q[0]),
                (h, w),
                bounds,
            )
        else:
            continue
        if not (0 <= gx < h and 0 <= gy < w):
            continue
        y0, y1 = max(0, gx - half_size_px), min(h, gx + half_size_px + 1)
        x0, x1 = max(0, gy - half_size_px), min(w, gy + half_size_px + 1)
        if y0 < y1 and x0 < x1:
            out[y0:y1, x0:x1] = color_bgr
    return out
