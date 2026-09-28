"""Post-process pairwise satellite change masks into temporal evidence labels."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def _connected_components(mask: np.ndarray):
    """Yield 8-connected pixel coordinates without adding a scipy dependency."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = deque([(int(y), int(x))])
        ys, xs = [], []
        while queue:
            cy, cx = queue.popleft()
            ys.append(cy)
            xs.append(cx)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        yield np.asarray(ys), np.asarray(xs)


def enrich_change_result(row: dict[str, Any], path_a: str, path_b: str) -> dict[str, Any]:
    """Add cautious change-type evidence and the earliest supported observation.

    Road and water labels are spectral/shape heuristics over AnyChange pixels,
    so they are explicitly marked as heuristic evidence in each result.
    """
    result = dict(row)
    if row.get("status") != "ok" or not row.get("mask_path"):
        result.update({"change_types": [], "change_events": [], "earliest_supported_observation": None})
        return result

    with rasterio.open(row["mask_path"]) as mask_src:
        class_map = mask_src.read(1)
        pixel_area = abs(mask_src.transform.a * mask_src.transform.e)
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        a, b = src_a.read(range(1, 7)).astype(np.float32), src_b.read(range(1, 7)).astype(np.float32)
        nodata_a = np.any(src_a.read(range(1, 7)) == -9999, axis=0)
        nodata_b = np.any(src_b.read(range(1, 7)) == -9999, axis=0)
    valid = ~(nodata_a | nodata_b) & (class_map != 255)
    changed = (class_map > 0) & (class_map < 255) & valid
    changed_count = int(changed.sum())

    # Band order: B2,B3,B4,B8,B11,B12; reflectance is stored x10000.
    a /= 10000.0
    b /= 10000.0
    ndwi_a = (a[1] - a[3]) / (a[1] + a[3] + 1e-6)
    ndwi_b = (b[1] - b[3]) / (b[1] + b[3] + 1e-6)
    ndbi_a = (a[4] - a[3]) / (a[4] + a[3] + 1e-6)
    ndbi_b = (b[4] - b[3]) / (b[4] + b[3] + 1e-6)
    water_a, water_b = ndwi_a >= 0.2, ndwi_b >= 0.2
    water_delta_pixels = int(((water_a != water_b) & changed).sum())
    water_delta = float((water_b & valid).sum() - (water_a & valid).sum()) * pixel_area / 10000.0

    events: list[dict[str, Any]] = []
    built_mask = (class_map == 1) & valid
    clearance_mask = (class_map == 2) & valid
    growth_mask = (class_map == 3) & valid
    built_ha = float(built_mask.sum() * pixel_area / 10000.0)
    clearance_ha = float(clearance_mask.sum() * pixel_area / 10000.0)
    growth_ha = float(growth_mask.sum() * pixel_area / 10000.0)
    if built_ha > 0:
        events.append({"type": "construction_or_built_expansion", "direction": "appearance_or_expansion",
                       "area_ha": built_ha, "basis": "AnyChange mask + NDBI/WorldCover rule"})
    built_contraction = changed & (ndbi_a >= 0.1) & (ndbi_b <= ndbi_a - 0.15)
    built_contraction_ha = float(built_contraction.sum() * pixel_area / 10000.0)
    if built_contraction_ha > 0:
        events.append({"type": "built_area_contraction_or_disappearance", "direction": "contraction_or_disappearance",
                       "area_ha": built_contraction_ha, "basis": "AnyChange pixels + NDBI decrease heuristic"})
    if clearance_ha > 0:
        events.append({"type": "vegetation_clearance", "direction": "loss", "area_ha": clearance_ha,
                       "basis": "AnyChange mask + NDVI decrease rule"})
    if growth_ha > 0:
        events.append({"type": "vegetation_gain", "direction": "gain", "area_ha": growth_ha,
                       "basis": "AnyChange mask + NDVI increase rule"})
    if water_delta_pixels >= 4 and abs(water_delta) >= 0.01:
        events.append({"type": "water_extent_variation",
                       "direction": "increase" if water_delta > 0 else "decrease",
                       "area_ha": abs(water_delta), "changed_pixels_overlapping_model_mask": water_delta_pixels,
                       "basis": "NDWI threshold delta within AnyChange pixels"})

    # A connected, elongated built-change component is a road-development
    # candidate. It remains a heuristic: optical tiles cannot prove road use.
    road_candidate_ha = 0.0
    for ys, xs in _connected_components(built_mask):
        if len(xs) < 8:
            continue
        height = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        elongation = max(height, width) / max(1, min(height, width))
        fill = len(xs) / max(1, height * width)
        if elongation >= 4.0 and fill <= 0.45:
            road_candidate_ha += len(xs) * pixel_area / 10000.0
    if road_candidate_ha > 0:
        events.append({"type": "possible_road_development", "direction": "appearance_or_expansion",
                       "area_ha": float(road_candidate_ha), "basis": "elongated built-change component heuristic"})

    start_year = int(row["year_a"])
    observed_year = int(row["year_b"])
    fraction = float(row.get("changed_fraction") or 0.0)
    confidence = float(row.get("mean_confidence") or 0.0)
    supported = changed_count >= 4 and (fraction >= 0.0001 or confidence >= 0.5)
    result.update({
        "change_types": list(dict.fromkeys(event["type"] for event in events)),
        "change_events": events,
        "earliest_supported_observation": observed_year if supported else None,
        "baseline_observation": start_year,
        "observation_years_available_for_pair": [start_year, observed_year],
        "temporal_evidence_note": "Only available observations in this dataset are 2021 and 2025; no intermediate date is available.",
        "heuristic_change_types": [event["type"] for event in events
                                   if event["type"] in {"water_extent_variation", "possible_road_development"}],
    })
    return result
