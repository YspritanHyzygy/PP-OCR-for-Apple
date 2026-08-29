from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import DetectedTextRegion


def rectify_region(
    source: np.ndarray,
    region: DetectedTextRegion,
    output_height: int,
    maximum_width: int,
    minimum_width: int = 16,
) -> tuple[np.ndarray, int]:
    """Perspective-sample one detector quad along its normalized reading axis."""

    points = np.asarray([(point.x, point.y) for point in region.quad], dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"region {region.id} 没有四个角点")
    inline = float(np.linalg.norm(points[1] - points[0]))
    crossline = float(np.linalg.norm(points[3] - points[0]))
    if inline <= 1 or crossline <= 1:
        raise ValueError(f"region {region.id} 尺寸过小")

    used_width = max(
        minimum_width,
        min(maximum_width, round(output_height * inline / crossline)),
    )
    u = (np.arange(used_width, dtype=np.float64) + 0.5) / used_width
    v = (np.arange(output_height, dtype=np.float64) + 0.5) / output_height
    grid_u, grid_v = np.meshgrid(u, v)
    top = (
        points[0][None, None, :] * (1 - grid_u[:, :, None])
        + points[1][None, None, :] * grid_u[:, :, None]
    )
    bottom = (
        points[3][None, None, :] * (1 - grid_u[:, :, None])
        + points[2][None, None, :] * grid_u[:, :, None]
    )
    sample_points = top * (1 - grid_v[:, :, None]) + bottom * grid_v[:, :, None]
    x = np.clip(sample_points[:, :, 0], 0, source.shape[1] - 1)
    y = np.clip(sample_points[:, :, 1], 0, source.shape[0] - 1)
    crop = np.stack(
        [
            ndimage.map_coordinates(
                source[:, :, channel],
                [y, x],
                order=1,
                mode="nearest",
            )
            for channel in range(3)
        ],
        axis=2,
    )
    return crop.astype(np.float32), used_width
