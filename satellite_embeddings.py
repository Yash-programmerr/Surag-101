#!/usr/bin/env python3
"""Create raw pooled embeddings with TorchGeo's Sentinel-2 MoCo ResNet-50.

README note: these tiles provide only B2, B3, B4, B8, B11, and B12.  The
remaining seven Sentinel-2 inputs expected by the 13-band checkpoint are
zero-filled, so this is an approximation of the checkpoint's native input.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torchgeo.models import ResNet50_Weights, resnet50


_MODEL: nn.Module | None = None
_DEVICE: str | None = None
_EXPECTED_BANDS: list[str] | None = None
_PREPROCESS: Callable[[torch.Tensor], torch.Tensor] | None = None


def load_encoder() -> tuple[nn.Module, str, list[str]]:
    """Load the real Sentinel-2-pretrained ResNet-50 encoder."""
    global _MODEL, _DEVICE, _EXPECTED_BANDS, _PREPROCESS
    if _MODEL is None:
        weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
        model = resnet50(weights=weights)
        # This TorchGeo/timm ResNet has global_pool followed directly by fc and
        # no forward_features method; replacing fc preserves global_pool's
        # flattened 2048-dimensional features as the model forward output.
        model.fc = nn.Identity()
        _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL = model.eval().to(_DEVICE)
        _EXPECTED_BANDS = list(weights.meta["bands"])
        _PREPROCESS = weights.transforms if hasattr(weights, "transforms") else None
    return _MODEL, _DEVICE, list(_EXPECTED_BANDS)


def read_tile_bands(path: str, expected_bands: list[str]) -> np.ndarray:
    """Read six available reflectance bands into Sentinel-2 checkpoint order."""
    global _PREPROCESS
    if _PREPROCESS is None:
        # Make the public function correct when called before load_encoder().
        load_encoder()

    with rasterio.open(path) as src:
        raw = src.read(list(range(1, 7)))
    if raw.shape[1:] != (256, 256):
        raise ValueError(f"Expected 256x256 tile, got {raw.shape[1:]} for {path}")

    result = np.zeros((len(expected_bands), 256, 256), dtype=np.float32)
    available = ("B2", "B3", "B4", "B8", "B11", "B12")
    for source_index, band_name in enumerate(available):
        values = raw[source_index]
        reflectance = values.astype(np.float32) / 10000.0
        reflectance[values == -9999] = 0.0
        result[expected_bands.index(band_name)] = reflectance

    # TorchGeo supplies this checkpoint's normalization transform. It operates
    # on a CHW tensor, and the function contract returns a float32 ndarray.
    if _PREPROCESS is not None:
        normalized = _PREPROCESS(torch.from_numpy(result))
        result = normalized.detach().cpu().numpy().astype(np.float32, copy=False)
    return result


def embed_tile_batch(paths: list[str]) -> np.ndarray:
    """Embed paths as unnormalised, pooled 2048-dimensional feature vectors."""
    if not paths:
        return np.empty((0, 2048), dtype=np.float32)
    model, device, expected_bands = load_encoder()
    batch = np.stack([read_tile_bands(path, expected_bands) for path in paths])
    with torch.inference_mode():
        features = model(torch.from_numpy(batch).to(device))
    features = features.detach().cpu().numpy().astype(np.float32, copy=False)
    if features.shape != (len(paths), 2048):
        raise RuntimeError(f"Expected pooled (n, 2048) features, got {features.shape}")
    return features


def _atomic_save(array: np.ndarray, tile_ids: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = out_dir / "sat_embeddings.npy"
    ids_path = out_dir / "sat_tile_ids.json"
    array_tmp = out_dir / "sat_embeddings.npy.tmp"
    ids_tmp = out_dir / "sat_tile_ids.json.tmp"
    with array_tmp.open("wb") as handle:
        np.save(handle, array)
    with ids_tmp.open("w", encoding="utf-8") as handle:
        json.dump(tile_ids, handle)
    os.replace(array_tmp, array_path)
    os.replace(ids_tmp, ids_path)


def embed_all(
    index_csv: str = "india_tiles/tiles_index.csv",
    out_dir: str = "india_tiles/embeddings",
    limit: int = 0,
    batch_size: int = 32,
) -> None:
    """Embed available, not-yet-embedded index rows and atomically checkpoint."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    out_path = Path(out_dir)
    array_path = out_path / "sat_embeddings.npy"
    ids_path = out_path / "sat_tile_ids.json"
    if array_path.exists() and ids_path.exists():
        embeddings = np.load(array_path)
        with ids_path.open(encoding="utf-8") as handle:
            tile_ids = json.load(handle)
        if embeddings.ndim != 2 or embeddings.shape[1] != 2048 or len(embeddings) != len(tile_ids):
            raise ValueError("Existing embeddings and tile IDs have incompatible shapes")
        embeddings = embeddings.astype(np.float32, copy=False)
    else:
        embeddings = np.empty((0, 2048), dtype=np.float32)
        tile_ids = []

    existing_ids = set(tile_ids)
    rows = pd.read_csv(index_csv)
    candidates = [
        (str(row.tile_id), str(row.path))
        for row in rows.itertuples(index=False)
        if str(row.tile_id) not in existing_ids and Path(str(row.path)).is_file()
    ]
    if limit > 0:
        candidates = candidates[:limit]
    total = len(candidates)
    done = 0
    batches_since_save = 0
    for start in range(0, total, batch_size):
        part = candidates[start : start + batch_size]
        part_ids, part_paths = zip(*part)
        vectors = embed_tile_batch(list(part_paths))
        embeddings = np.concatenate((embeddings, vectors), axis=0)
        tile_ids.extend(part_ids)
        done += len(part)
        batches_since_save += 1
        print(f"{done}/{total} embedded", flush=True)
        if batches_since_save == 20:
            _atomic_save(embeddings, tile_ids, out_path)
            batches_since_save = 0
    _atomic_save(embeddings, tile_ids, out_path)


def _selftest() -> int:
    try:
        model, device, expected_bands = load_encoder()
        del model
        rows = pd.read_csv("india_tiles/tiles_index.csv")
        path = next(str(row.path) for row in rows.itertuples(index=False) if Path(str(row.path)).is_file())
        vector = embed_tile_batch([path])[0]
        print(f"shape: {vector.shape}")
        print(f"first 5 values: {vector[:5]}")
        if vector.shape != (2048,) or not np.isfinite(vector).all():
            print("FAIL: expected 2048 finite feature values")
            return 1
        print(f"PASS (device: {device})")
        return 0
    except StopIteration:
        print("FAIL: no tile path in india_tiles/tiles_index.csv exists on disk")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    embed = subparsers.add_parser("embed")
    embed.add_argument("--limit", type=int, default=0)
    embed.add_argument("--batch-size", type=int, default=32)
    subparsers.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        return _selftest()
    embed_all(limit=args.limit, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
