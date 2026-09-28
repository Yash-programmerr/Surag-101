#!/usr/bin/env python3
"""Zero-shot AnyChange detection for paired Sentinel-2 tile GeoTIFFs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.warp import transform_geom
from torchange.models.segment_any_change import AnyChange
from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask


_MODEL: AnyChange | None = None
_DEVICE: str | None = None
EPS = 1e-6
CLASS_NAMES = {1: "new_built", 2: "vegetation_loss", 3: "vegetation_gain", 4: "other_change"}
CLASS_COLORS = {1: np.array([255, 0, 0]), 2: np.array([255, 140, 0]),
                3: np.array([0, 200, 0]), 4: np.array([255, 255, 0])}
SUMMARY_FIELDS = ["tile_id", "category", "year_a", "year_b", "status", "changed_fraction",
                  "new_built_ha", "vegetation_loss_ha", "vegetation_gain_ha", "other_change_ha",
                  "total_change_ha", "mean_confidence", "change_source", "mask_path",
                  "preview_path", "polygons_path"]


def load_anychange(sam_checkpoint: str = "models/sam_vit_b_01ec64.pth") -> AnyChange:
    """Load and cache AnyChange with its underlying SAM placed on MPS or CPU."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        checkpoint = Path(sam_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        # Verified API: AnyChange(model_type='vit_b', sam_checkpoint=...) is
        # a wrapper, not nn.Module. Move its underlying SAM module, then make
        # a new generator that holds that moved SAM instance.
        _MODEL = AnyChange(model_type="vit_b", sam_checkpoint=str(checkpoint))
        _MODEL.device = torch.device(_DEVICE)
        _MODEL.sam = _MODEL.sam.to(_MODEL.device).eval()
        layernorm = _MODEL.sam.image_encoder.neck[3]
        weight = layernorm.weight.data.reshape(-1, 1, 1)
        bias = layernorm.bias.data.reshape(-1, 1, 1)
        _MODEL.inv_transform = lambda embedding: (embedding - bias) / weight
        # A 2x2 prompt grid keeps CPU/MPS inference practical for 256px tiles
        # while retaining AnyChange's pretrained zero-shot proposal mechanism.
        _MODEL.make_mask_generator(points_per_side=2)
        # Torchange 0.0.4 builds its NumPy sampling grid as float64.  Its
        # generator passes that directly to MPS, which supports float32 only.
        _MODEL.maskgen.point_grids = _MODEL.maskgen.point_grids.astype(np.float32)
        original_apply_coords = _MODEL.maskgen.predictor.transform.apply_coords
        _MODEL.maskgen.predictor.transform.apply_coords = (
            lambda coords, original_size: original_apply_coords(coords, original_size).astype(np.float32)
        )
    return _MODEL


def _raw_tile(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        data = src.read(list(range(1, 8)))
        profile = src.profile.copy()
    nodata = np.any(data[:6] == -9999, axis=0)
    return data, nodata, profile


def tile_rgb_uint8(path: str) -> np.ndarray:
    """Convert B4/B3/B2 reflectance into the requested display RGB stretch."""
    data, nodata, _ = _raw_tile(path)
    rgb = data[[2, 1, 0]].astype(np.float32) / 10000.0
    rgb = np.clip(rgb / 0.3, 0.0, 1.0) ** 0.8
    result = np.moveaxis((rgb * 255.0).astype(np.uint8), 0, -1)
    result[nodata] = 0
    return result


def find_pairs(index_csv: str, year_a: int = 2021, year_b: int = 2025) -> list[dict[str, str]]:
    """Find tile IDs having paths in both requested years."""
    by_id: dict[str, dict[int, dict[str, str]]] = {}
    with Path(index_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tile_id = row.get("tile_id", "")
            try:
                year = int(row.get("year", ""))
            except ValueError:
                continue
            if tile_id and year in (year_a, year_b) and row.get("path"):
                by_id.setdefault(tile_id, {})[year] = row
    pairs = []
    for tile_id, rows in by_id.items():
        if year_a in rows and year_b in rows:
            pairs.append({"tile_id": tile_id, "category": rows[year_b].get("category", ""),
                          "path_a": rows[year_a]["path"], "path_b": rows[year_b]["path"]})
    return pairs


def _change_instances(rgb_a: np.ndarray, rgb_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Call verified AnyChange.forward(img1, img2) and rasterize its RLE masks."""
    model = load_anychange()
    with torch.inference_mode():
        changemasks, _, _ = model.forward(rgb_a, rgb_b)  # forward(img1, img2) -> (MaskData, MaskData, MaskData)
    model.clear_cached_embedding()
    cls = np.zeros(rgb_a.shape[:2], dtype=np.uint8)
    confidence = np.zeros(rgb_a.shape[:2], dtype=np.float32)
    scores = changemasks["change_confidence"]
    for rle, score in zip(changemasks["rles"], scores):
        mask = rle_to_mask(rle).astype(bool)
        score_value = float(score.detach().cpu()) if torch.is_tensor(score) else float(score)
        # forward() already applies model.change_confidence_threshold. If a
        # future model returns unfiltered scores, retain the documented 0.5 fallback.
        if hasattr(model, "change_confidence_threshold") or score_value >= 0.5:
            cls[mask] = 1
            confidence[mask] = np.maximum(confidence[mask], score_value)
    return cls, confidence


def _write_preview(rgb_a: np.ndarray, rgb_b: np.ndarray, class_map: np.ndarray, output: Path) -> None:
    overlay = rgb_b.astype(np.float32).copy()
    for code, color in CLASS_COLORS.items():
        mask = class_map == code
        overlay[mask] = 0.4 * overlay[mask] + 0.6 * color
    Image.fromarray(np.concatenate((rgb_a, rgb_b, overlay.astype(np.uint8)), axis=1)).save(output)


def _write_polygons(class_map: np.ndarray, valid: np.ndarray, profile: dict[str, Any], output: Path,
                    tile_id: str, category: str, year_a: int, year_b: int, pixel_area: float,
                    confidence: np.ndarray) -> None:
    items = []
    for geom, code in shapes(class_map, mask=(class_map > 0) & valid, transform=profile["transform"]):
        code = int(code)
        if code == 0:
            continue
        if profile.get("crs"):
            geom = transform_geom(profile["crs"], "EPSG:4326", geom)
        pixels = class_map == code
        items.append({"type": "Feature", "geometry": geom, "properties": {
            "tile_id": tile_id, "category": category, "class_code": code,
            "class_name": CLASS_NAMES[code], "area_m2": float(pixels.sum() * pixel_area),
            "year_a": year_a, "year_b": year_b, "confidence": float(confidence[pixels].mean())}})
    output.write_text(json.dumps({"type": "FeatureCollection", "features": items}), encoding="utf-8")


def _append_summary(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def detect_change_ml(path_a: str, path_b: str, tile_id: str, category: str, out_dir: str,
                     year_a: int, year_b: int) -> dict[str, Any]:
    """Detect, label, write, and summarize one AnyChange tile pair."""
    try:
        rgb_a, rgb_b = tile_rgb_uint8(path_a), tile_rgb_uint8(path_b)
        raw_a, nodata_a, _ = _raw_tile(path_a)
        raw_b, nodata_b, profile_b = _raw_tile(path_b)
        cls, confidence = _change_instances(rgb_a, rgb_b)
        valid = ~(nodata_a | nodata_b)
        cls[~valid] = 0
        reflectance_a, reflectance_b = raw_a[:6].astype(np.float32) / 10000.0, raw_b[:6].astype(np.float32) / 10000.0
        ndvi_a = (reflectance_a[3] - reflectance_a[2]) / (reflectance_a[3] + reflectance_a[2] + EPS)
        ndvi_b = (reflectance_b[3] - reflectance_b[2]) / (reflectance_b[3] + reflectance_b[2] + EPS)
        ndbi_b = (reflectance_b[4] - reflectance_b[3]) / (reflectance_b[4] + reflectance_b[3] + EPS)
        changed = (cls == 1) & valid
        class_map = np.zeros(cls.shape, dtype=np.uint8)
        class_map[changed] = 4
        class_map[changed & ((ndvi_b - ndvi_a) >= 0.15)] = 3
        class_map[changed & (ndvi_a >= 0.3) & (ndvi_b <= ndvi_a - 0.15)] = 2
        class_map[changed & (ndbi_b >= 0.1) & (raw_a[6] != 50)] = 1
        output_base = Path(out_dir) / category
        output_base.mkdir(parents=True, exist_ok=True)
        stem = f"{tile_id}_{year_a}_{year_b}_ml"
        mask_path, preview_path, polygons_path = (output_base / f"{stem}_mask.tif", output_base / f"{stem}_preview.png", output_base / f"{stem}_polygons.geojson")
        profile_b.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
        write_map = class_map.copy(); write_map[~valid] = 255
        with rasterio.open(mask_path, "w", **profile_b) as dst:
            dst.write(write_map, 1)
        _write_preview(rgb_a, rgb_b, class_map, preview_path)
        pixel_area = abs(profile_b["transform"].a * profile_b["transform"].e)
        _write_polygons(class_map, valid, profile_b, polygons_path, tile_id, category, year_a, year_b, pixel_area, confidence)
        hectares = {CLASS_NAMES[code]: float((class_map == code).sum() * pixel_area / 10000.0) for code in CLASS_NAMES}
        row = {"tile_id": tile_id, "category": category, "year_a": year_a, "year_b": year_b,
               "status": "ok", "changed_fraction": float(changed.sum() / max(1, valid.sum())),
               "new_built_ha": hectares["new_built"], "vegetation_loss_ha": hectares["vegetation_loss"],
               "vegetation_gain_ha": hectares["vegetation_gain"], "other_change_ha": hectares["other_change"],
               "total_change_ha": float(changed.sum() * pixel_area / 10000.0),
               "mean_confidence": float(confidence[changed].mean()) if changed.any() else 0.0,
               "change_source": "anychange_ml", "mask_path": str(mask_path), "preview_path": str(preview_path),
               "polygons_path": str(polygons_path)}
        _append_summary(Path(out_dir) / "change_summary_ml.csv", row)
        return row
    except Exception as exc:
        return {"tile_id": tile_id, "status": f"error: {exc}"}


def _selftest() -> int:
    pairs = find_pairs("india_tiles/tiles_index.csv")
    if not pairs:
        print("FAIL: no 2021/2025 tile pairs found")
        return 1
    pair = pairs[0]
    row = detect_change_ml(**pair, out_dir="india_tiles/change", year_a=2021, year_b=2025)
    if row.get("status") != "ok":
        print(f"FAIL: {row.get('status')}")
        return 1
    with rasterio.open(row["mask_path"]) as src:
        mask = src.read(1)
    print(f"changed pixels: {int(np.isin(mask, [1, 2, 3, 4]).sum())}")
    print("breakdown: " + ", ".join(f"{name}={int((mask == code).sum())}" for code, name in CLASS_NAMES.items()))
    print(f"preview: {row['preview_path']}")
    print("PASS")
    return 0


def _existing_ok_ids(summary_path: Path, year_a: int, year_b: int) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return {row["tile_id"] for row in csv.DictReader(handle) if row.get("status") == "ok" and
                row.get("year_a") == str(year_a) and row.get("year_b") == str(year_b)}


def _run(args: argparse.Namespace) -> int:
    root = Path(args.input_dir)
    pairs = find_pairs(str(root / "tiles_index.csv"), args.year_a, args.year_b)
    if args.tile_id:
        pairs = [pair for pair in pairs if pair["tile_id"] == args.tile_id]
    done = set() if args.force else _existing_ok_ids(root / "change" / "change_summary_ml.csv", args.year_a, args.year_b)
    pairs = [pair for pair in pairs if pair["tile_id"] not in done]
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        print("No tile pairs found to process.")
        return 1
    rows = []
    for index, pair in enumerate(pairs, 1):
        rows.append(detect_change_ml(**pair, out_dir=str(root / "change"), year_a=args.year_a, year_b=args.year_b))
        if index % 5 == 0 or index == len(pairs):
            print(f"{index}/{len(pairs)} processed")
    ok = [row for row in rows if row.get("status") == "ok"]
    print(f"Totals: ok={len(ok)} errors={len(rows) - len(ok)} mean_changed_fraction=" +
          (f"{np.mean([row['changed_fraction'] for row in ok]):.6f}" if ok else "n/a"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--input-dir", default="india_tiles")
    run.add_argument("--year-a", type=int, default=2021)
    run.add_argument("--year-b", type=int, default=2025)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--tile-id")
    run.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest")
    args = parser.parse_args()
    return _selftest() if args.command == "selftest" else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
