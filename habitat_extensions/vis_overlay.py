"""Visualization helpers for top-down maps (history markers, etc.)."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import

from habitat_extensions import maps as ext_maps

cv2 = try_cv2_import()


def draw_history_markers(
    bgr: np.ndarray,
    sim: Optional[Simulator],
    positions_xyz: Sequence[Union[np.ndarray, Sequence[float]]],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    min_dist_m: float = 0.35,
    color_bgr: Tuple[int, int, int] = (0, 140, 255),
    radius_px: int = 5,
    outline_bgr: Tuple[int, int, int] = (255, 255, 255),
    outline_thickness: int = 2,
) -> np.ndarray:
    r"""Draw subsampled visited positions on a colorized top-down BGR image.

    Uses filled circles with a light outline so markers stay visible over the
    agent trajectory line. Call after the map is resized to the final mosaic size.
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
        center = (int(gy), int(gx))
        cv2.circle(
            out,
            center,
            radius_px + outline_thickness,
            outline_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            out,
            center,
            radius_px,
            color_bgr,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return out
