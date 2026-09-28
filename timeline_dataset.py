#!/usr/bin/env python3
"""Download a consistent February–April Sentinel-2 timeline on existing UTM tiles.

Self-contained: band, CloudScore+, computePixels and RGB logic are copied from
bulk_tiles.py; that module is neither imported nor modified. April 30 is inclusive.
GAUL 2015 state names/boundaries are historical, not current administrative units.
Missing change scores are never interpreted as zero-change controls.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import ee
import numpy as np
import rasterio
from PIL import Image

YEARS = [2022, 2023, 2024, 2025, 2026]
WINDOW_START_MD = "02-01"
WINDOW_END_MD = "04-30"
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
CLOUD_SCORE_MIN = 0.60
MAX_SCENE_CLOUD = 70
NODATA = -9999
TILE_PX, SCALE = 256, 10
OUT = Path("india_tiles_timeline")
GEE_PROJECT = "geesatellite"
S2 = "COPERNICUS/S2_SR_HARMONIZED"
CSP = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
SOURCE_INDEX = Path("india_tiles/tiles_index.csv")
CHANGE_SUMMARY = Path("india_tiles/change/change_summary.csv")
INDEX_FIELDS = ["tile_id", "year", "path", "thumb", "state", "category", "source_category",
                "selection_group", "epsg",
                "x0", "y0", "x1", "y1", "lon", "lat", "valid_fraction", "ndvi_mean",
                "ndbi_mean", "n_scenes", "mean_scene_cloud_pct", "window", "downloaded_utc"]
SELECTION_FIELDS = ["tile_id", "state", "category", "source_category", "epsg", "x0", "y0", "x1", "y1",
                    "lon", "lat", "total_change_ha", "selection_group"]
_initialized = False
_label = None


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def ee_init():
    global _initialized, _label
    if not _initialized:
        # Use existing credentials; authentication needing browser input is explicit.
        ee.Initialize(project=GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
        _label = (ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
                  .rename("WorldCover_2021").toInt16().unmask(0))
        _initialized = True


def with_retries(fn, tries=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            message = str(exc).lower()
            if any(part in message for part in ("did not match", "no bands", "permission denied",
                                                 "invalid_grant", "not registered")):
                raise
            if attempt == tries - 1:
                raise
            delay = min(48, 3 * 2 ** attempt) + random.random()
            print(f"Retry {attempt + 1}/{tries - 1} in {delay:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(delay)


def dual_tiles():
    if not SOURCE_INDEX.exists():
        raise FileNotFoundError(str(SOURCE_INDEX))
    rows_by_id = defaultdict(dict)
    for row in read_csv(SOURCE_INDEX):
        if row.get("year") in {"2021", "2025"}:
            rows_by_id[row["tile_id"]][int(row["year"])] = row
    tiles = []
    for tile_id, observations in sorted(rows_by_id.items()):
        if not {2021, 2025} <= observations.keys():
            continue
        row = dict(observations[2025])
        for other in observations.values():
            for key in ("epsg", "x0", "y0", "x1", "y1"):
                if float(other[key]) != float(row[key]):
                    raise ValueError(f"Inconsistent grid for {tile_id}: {key}")
        if (float(row["x1"]) - float(row["x0"]) != TILE_PX * SCALE or
                float(row["y1"]) - float(row["y0"]) != TILE_PX * SCALE):
            raise ValueError(f"Unexpected grid dimensions for {tile_id}")
        tiles.append(row)
    if not tiles:
        raise ValueError("No tiles have both 2021 and 2025 rows")
    return tiles


def change_scores():
    scores = {}
    if not CHANGE_SUMMARY.exists():
        print(f"MISSING: {CHANGE_SUMMARY}; change statistics are unavailable.", flush=True)
        return scores
    for row in read_csv(CHANGE_SUMMARY):
        if row.get("status", "ok") != "ok":
            continue
        if row.get("year_a") and {row["year_a"], row.get("year_b")} != {"2021", "2025"}:
            continue
        try:
            score = float(row["total_change_ha"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(score) and score >= 0:
            scores[row["tile_id"]] = score
    return scores


def assign_states(tiles, refresh=False):
    cache_path = OUT / "tile_states.csv"
    cached = {row["tile_id"]: row for row in read_csv(cache_path)}
    if not refresh and all(row["tile_id"] in cached and
                           all(row[k] == cached[row["tile_id"]][k] for k in ("lon", "lat"))
                           for row in tiles):
        print(f"State assignment: using {len(tiles)} cached points from {cache_path}", flush=True)
        return [dict(row, state=cached[row["tile_id"]]["state"]) for row in tiles]
    ee_init()
    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(row["lon"]), float(row["lat"])]), {"tile_id": row["tile_id"]})
        for row in tiles])
    states = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM0_NAME", "India"))

    def tag(point):
        matches = states.filterBounds(point.geometry()).sort("ADM1_NAME")
        state = ee.Algorithms.If(matches.size().gt(0), matches.first().get("ADM1_NAME"), "UNASSIGNED")
        return ee.Feature(None, {"tile_id": point.get("tile_id"), "state": state})

    print(f"State assignment: one batched Earth Engine request for {len(tiles)} lon/lat points", flush=True)
    result = with_retries(lambda: points.map(tag).getInfo())
    names = {feature["properties"]["tile_id"]: feature["properties"]["state"]
             for feature in result["features"]}
    assigned = [dict(row, state=names.get(row["tile_id"], "UNASSIGNED")) for row in tiles]
    write_csv(cache_path, ["tile_id", "lon", "lat", "state"], assigned)
    return assigned


def region_for(row):
    return ee.Geometry.Rectangle([float(row[k]) for k in ("x0", "y0", "x1", "y1")],
                                 proj=f"EPSG:{row['epsg']}", geodesic=False)


def collection(row, year, cloud_filter=True):
    # EE end dates are exclusive, so advance to May 1 to include all of April 30.
    end = (date.fromisoformat(f"{year}-{WINDOW_END_MD}") + timedelta(days=1)).isoformat()
    col = ee.ImageCollection(S2).filterBounds(region_for(row)).filterDate(f"{year}-{WINDOW_START_MD}", end)
    return col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD)) if cloud_filter else col


def availability():
    row = dual_tiles()[0]
    ee_init()
    expressions = {str(year): ee.Dictionary({"all": collection(row, year, False).size(),
                                           "cloud_lt_70": collection(row, year).size()}) for year in YEARS}
    counts = with_retries(lambda: ee.Dictionary(expressions).getInfo())
    print(f"Sample tile: {row['tile_id']} ({row['lon']}, {row['lat']})")
    print(f"Window: {WINDOW_START_MD} through {WINDOW_END_MD}, inclusive, every year")
    print("year  all_scenes  scene_cloud_lt_70")
    for year in YEARS:
        info = counts[str(year)]
        print(f"{year}  {info['all']:10d}  {info['cloud_lt_70']:17d}")
        if not info["all"] or not info["cloud_lt_70"]:
            print(f"WARNING: {year} has ZERO {'total' if not info['all'] else 'cloud-filtered'} scenes")
    print("Counts precede the per-pixel CloudScore+ >= 0.60 mask; valid coverage can still be low.")


def states_report():
    tiles = assign_states(dual_tiles(), refresh=True)
    scores = change_scores()
    grouped = defaultdict(list)
    for row in tiles:
        grouped[row["state"]].append(row)
    reports = []
    print("state | dual_date_tiles | scored_tiles | median_total_change_ha | max_total_change_ha")
    for state, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        values = [scores[row["tile_id"]] for row in group if row["tile_id"] in scores]
        med, maximum = (median(values), max(values)) if values else ("", "")
        report = dict(state=state, dual_date_tiles=len(group), scored_tiles=len(values),
                      median_total_change_ha=med, max_total_change_ha=maximum)
        reports.append(report)
        print(f"{state} | {len(group)} | {len(values)} | {med if med != '' else 'N/A'} | {maximum if maximum != '' else 'N/A'}")
    path = OUT / "states_report.csv"
    write_csv(path, list(reports[0]), reports)
    print(f"Saved: {path}")


def score_baselines():
    """Create absent historical ranking scores using explicitly documented rules.

    The original change_detection.py is absent in this checkout. These are new
    proxy scores, not a reproduction of that missing detector: |delta NDVI| >=
    .15 for gain/loss (loss requires baseline NDVI >= .3), plus NDBI >= .1,
    delta NDBI >= .15 and baseline WorldCover != 50 for built-change candidates.
    Inputs are the existing 2021/2025 composites, not the new seasonal timeline.
    """
    if CHANGE_SUMMARY.exists():
        raise FileExistsError(f"Refusing to overwrite existing {CHANGE_SUMMARY}")
    tiles = dual_tiles()
    observations = {(row["tile_id"], row["year"]): row for row in read_csv(SOURCE_INDEX)}
    result = []
    print("Scoring existing 2021/2025 composites; source=timeline_baseline_rules_v1", flush=True)
    print("Rules: NDVI gain >= 0.15; loss >= 0.15 with baseline NDVI >= 0.3; "
          "built candidate: NDBI_b >= 0.1, delta NDBI >= 0.15, WorldCover_a != 50", flush=True)
    for i, row in enumerate(tiles, 1):
        with rasterio.open(observations[(row["tile_id"], "2021")]["path"]) as src_a, \
                rasterio.open(observations[(row["tile_id"], "2025")]["path"]) as src_b:
            if src_a.crs != src_b.crs or src_a.transform != src_b.transform or src_a.shape != src_b.shape:
                raise ValueError(f"Mismatched source raster grids: {row['tile_id']}")
            a, b = src_a.read(), src_b.read()
            valid = (a[:6] != NODATA).all(axis=0) & (b[:6] != NODATA).all(axis=0)
            area = abs(src_b.transform.a * src_b.transform.e - src_b.transform.b * src_b.transform.d)
        af, bf = a[:6].astype(np.float32) / 10000, b[:6].astype(np.float32) / 10000
        ndvi_a = (af[3] - af[2]) / (af[3] + af[2] + 1e-6)
        ndvi_b = (bf[3] - bf[2]) / (bf[3] + bf[2] + 1e-6)
        ndbi_a = (af[4] - af[3]) / (af[4] + af[3] + 1e-6)
        ndbi_b = (bf[4] - bf[3]) / (bf[4] + bf[3] + 1e-6)
        gain = valid & (ndvi_b - ndvi_a >= .15)
        loss = valid & (ndvi_a >= .3) & (ndvi_a - ndvi_b >= .15)
        built = valid & (ndbi_b >= .1) & (ndbi_b - ndbi_a >= .15) & (a[6] != 50)
        # Disjoint classes with built > loss > gain, matching the ML type priority.
        loss &= ~built
        gain &= ~(built | loss)
        changed = built | loss | gain
        result.append(dict(tile_id=row["tile_id"], category=row["category"], year_a=2021, year_b=2025,
                           status="ok" if valid.any() else "error: no jointly valid pixels",
                           changed_fraction=float(changed.sum() / max(1, valid.sum())),
                           new_built_ha=float(built.sum() * area / 10000),
                           vegetation_loss_ha=float(loss.sum() * area / 10000),
                           vegetation_gain_ha=float(gain.sum() * area / 10000),
                           total_change_ha=float(changed.sum() * area / 10000),
                           change_source="timeline_baseline_rules_v1"))
        if i % 100 == 0 or i == len(tiles):
            print(f"{i}/{len(tiles)} baseline pairs scored", flush=True)
    write_csv(CHANGE_SUMMARY, list(result[0]), result)
    print(f"Saved: {CHANGE_SUMMARY} ({len(result)} rows)")


def select_tiles(tiles, states, per_state):
    scores = change_scores()
    selected = []
    available_names = {row["state"] for row in tiles if row["state"] != "UNASSIGNED"}
    for state in states:
        if state == "UNASSIGNED":
            raise ValueError("UNASSIGNED is excluded from timeline selection")
        if state not in available_names:
            raise ValueError(f"State {state!r} is absent; use the exact GAUL names in states-report")
        candidates = [row for row in tiles if row["state"] == state]
        n = min(per_state, len(candidates))
        if n < per_state:
            print(f"SHORTFALL: {state}: requested={per_state}, available={n}, shortfall={per_state - n}; taking all")
        # Largest remainders give the closest integer 50/30/20 split when n
        # is not a multiple of ten. Ties go high_change, control, random.
        proportions = (0.50, 0.30, 0.20)
        counts = [math.floor(n * share) for share in proportions]
        for index in sorted(range(3), key=lambda i: (-(n * proportions[i] - counts[i]), i))[:n - sum(counts)]:
            counts[index] += 1
        n_high, n_low, n_random = counts
        scored = [row for row in candidates if row["tile_id"] in scores]
        high = sorted(scored, key=lambda row: (-scores[row["tile_id"]], row["tile_id"]))[:n_high]
        taken = {row["tile_id"] for row in high}
        low = sorted((row for row in scored if row["tile_id"] not in taken),
                     key=lambda row: (scores[row["tile_id"]], row["tile_id"]))[:n_low]
        taken.update(row["tile_id"] for row in low)
        remaining = sorted((row for row in candidates if row["tile_id"] not in taken),
                           key=lambda row: row["tile_id"])
        random_rows = random.Random(42).sample(remaining, min(n_random, len(remaining)))
        group = [dict(row, source_category=row["category"], category=label,
                      total_change_ha=scores.get(row["tile_id"], ""), selection_group=label)
                 for label, rows in (("high_change", high), ("control", low), ("random", random_rows))
                 for row in rows]
        print(f"{state}: available={len(candidates)}, selected={len(group)}, "
              f"high_change={len(high)}, control={len(low)}, random={len(random_rows)}, "
              f"unscored_candidates={len(candidates) - len(scored)}")
        if len(group) < n:
            print(f"SHORTFALL: {state}: {n - len(group)} additional tiles cannot fill the scored "
                  "high/control groups; missing/error scores are eligible only for random")
        selected.append(group)
    # Round-robin keeps a small --limit-tiles balanced across the requested states.
    return [group[i] for i in range(max(map(len, selected))) for group in selected if i < len(group)]


def selection_check(args):
    states = [state.strip() for state in args.states.split(",") if state.strip()]
    if len(states) != 4 or len(set(states)) != 4:
        raise ValueError("--states must contain exactly four distinct GAUL state names")
    selected = select_tiles(assign_states(dual_tiles()), states, args.tiles_per_state)
    selected_ids = {row["tile_id"] for row in selected}
    print(f"Total selected tiles: {len(selected)}")
    print(f"Total tile-years: {len(selected) * len(YEARS)}")
    print("Selected group totals:", dict(sorted(Counter(row["category"] for row in selected).items())))
    previous = {row["tile_id"]: row for row in read_csv(OUT / "selected_tiles.csv")}
    print(f"Existing test tiles in manifest: {len(previous)}")
    for tile_id, row in sorted(previous.items()):
        files = [OUT / "tiles" / f"{tile_id}_{year}.tif" for year in YEARS]
        complete = all(path.exists() for path in files)
        valid = complete
        if complete:
            for year, path in zip(YEARS, files):
                with rasterio.open(path) as src:
                    valid &= (src.count, src.width, src.height) == (7, TILE_PX, TILE_PX)
                    valid &= src.tags().get("timeline_window") == f"{year}-{WINDOW_START_MD}/{year}-{WINDOW_END_MD}"
        print(f"  {tile_id}: {'IN' if tile_id in selected_ids else 'OUTSIDE'} new selection; "
              f"files={sum(path.exists() for path in files)}/5; valid={valid}")
    print("Selection check performed no downloads.")


def fetch_tile(row, year, previous=None):
    dest = OUT / "tiles" / f"{row['tile_id']}_{year}.tif"
    thumb = OUT / "thumbs" / f"{row['tile_id']}_{year}.jpg"
    window = f"{year}-{WINDOW_START_MD}/{year}-{WINDOW_END_MD}"
    if previous and previous.get("window") != window:
        raise ValueError(f"Existing index window differs for {row['tile_id']}/{year}; use a separate output directory")
    if dest.exists():
        with rasterio.open(dest) as src:
            arr = src.read()
            tags = src.tags()
        if tags.get("timeline_window") and tags["timeline_window"] != window:
            raise ValueError(f"Existing raster window differs: {dest}")
        if not previous and not tags.get("timeline_window"):
            raise ValueError(f"Cannot verify date window for unindexed file {dest}; file kept unchanged")
        n_scenes = previous.get("n_scenes") if previous else tags.get("n_scenes")
        cloud = previous.get("mean_scene_cloud_pct") if previous else tags.get("mean_scene_cloud_pct")
        downloaded = previous.get("downloaded_utc") if previous else tags.get("downloaded_utc")
        action = "existing"
    else:
        col = collection(row, year)
        info = with_retries(lambda: ee.Dictionary({"n": col.size(), "c": col.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE")}).getInfo())
        n_scenes, cloud = info["n"], info.get("c")
        if not n_scenes:
            raise RuntimeError(f"no_scenes_for_tile: {row['tile_id']} year={year} window={window}")
        linked = col.linkCollection(ee.ImageCollection(CSP), ["cs_cdf"])
        masked = linked.map(lambda image: image.updateMask(image.select("cs_cdf").gte(CLOUD_SCORE_MIN)))
        optical = masked.select(BANDS).median().round().toInt16().unmask(NODATA).toInt16()
        request = {"expression": optical.addBands(_label), "fileFormat": "GEO_TIFF", "grid": {
            "dimensions": {"width": TILE_PX, "height": TILE_PX},
            "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": float(row["x0"]),
                                "shearY": 0, "scaleY": -SCALE, "translateY": float(row["y1"])},
            "crsCode": f"EPSG:{row['epsg']}"}}
        data = with_retries(lambda: ee.data.computePixels(request))
        with rasterio.MemoryFile(data) as memory:
            with memory.open() as src:
                arr, profile = src.read(), src.profile
        if arr.shape != (7, TILE_PX, TILE_PX) or arr.dtype != np.int16:
            raise ValueError(f"Unexpected raster shape/dtype: {arr.shape}, {arr.dtype}")
        downloaded = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".tif.tmp")
        # Store explicit nodata and metadata for recovery after interrupted CSV writes.
        profile.update(driver="GTiff", dtype="int16", count=7, nodata=NODATA, compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256, photometric="MINISBLACK")
        with rasterio.open(temp, "w", **profile) as dst:
            dst.write(arr)
            for i, band in enumerate(BANDS + ["WorldCover_2021"], 1):
                dst.set_band_description(i, band)
            dst.update_tags(timeline_window=window, n_scenes=n_scenes, mean_scene_cloud_pct=cloud,
                            downloaded_utc=downloaded)
        temp.replace(dest)
        action = "downloaded"
    if arr.shape != (7, TILE_PX, TILE_PX) or arr.dtype != np.int16:
        raise ValueError(f"Unexpected existing raster shape/dtype: {dest}: {arr.shape}, {arr.dtype}")
    valid = (arr[:6] != NODATA).all(axis=0)
    if not thumb.exists():
        thumb.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.clip(arr[[2, 1, 0]].astype(np.float32) / 3000.0, 0, 1) ** 0.8
        rgb[:, ~valid] = 0
        temp = thumb.with_suffix(".jpg.tmp")
        Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8)).save(temp, format="JPEG", quality=90)
        temp.replace(thumb)
    ndvi = ndbi = None
    if valid.any():
        b4, b8, b11 = (arr[i][valid].astype(np.float32) for i in (2, 3, 4))
        ndvi = float(np.mean((b8 - b4) / np.maximum(b8 + b4, 1)))
        ndbi = float(np.mean((b11 - b8) / np.maximum(b11 + b8, 1)))
    result = dict(row, year=year, path=str(dest), thumb=str(thumb), valid_fraction=round(float(valid.mean()), 4),
                  ndvi_mean=None if ndvi is None else round(ndvi, 4), ndbi_mean=None if ndbi is None else round(ndbi, 4),
                  n_scenes=n_scenes, mean_scene_cloud_pct=cloud, window=window, downloaded_utc=downloaded)
    return result, action


def build(args):
    states = [state.strip() for state in args.states.split(",") if state.strip()]
    if len(states) != 4 or len(set(states)) != 4:
        raise ValueError("--states must contain exactly four distinct GAUL state names")
    tiles = assign_states(dual_tiles())
    selected = select_tiles(tiles, states, args.tiles_per_state)
    if args.limit_tiles:
        selected = selected[:args.limit_tiles]
    # Preserve previous selections so status can report interrupted and earlier runs.
    manifest_path = OUT / "selected_tiles.csv"
    manifest = {row["tile_id"]: row for row in read_csv(manifest_path)}
    manifest.update({row["tile_id"]: row for row in selected})
    write_csv(manifest_path, SELECTION_FIELDS, manifest.values())
    ee_init()
    index_path = OUT / "tiles_timeline_index.csv"
    index = {(row["tile_id"], int(row["year"])): row for row in read_csv(index_path)}
    failures_path = OUT / "failures.csv"
    failure_fields = ["tile_id", "year", "state", "error", "failed_utc"]
    failures = {(row["tile_id"], int(row["year"])): row for row in read_csv(failures_path)}
    jobs = [(row, year) for row in selected for year in YEARS]
    print(f"Selected tiles: {len(selected)} | tile-years: {len(jobs)} | workers: {args.workers}", flush=True)
    print("Selection by state:", dict(Counter(row["state"] for row in selected)), flush=True)
    started = time.perf_counter()
    totals = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_tile, row, year, index.get((row["tile_id"], year))): (row, year)
                   for row, year in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row, year = futures[future]
            key = row["tile_id"], year
            try:
                result, action = future.result()
                index[key] = result
                failures.pop(key, None)
                totals[action] += 1
                write_csv(index_path, INDEX_FIELDS, [index[k] for k in sorted(index)])
                print(f"{done}/{len(jobs)} {action}: {key[0]} {year} valid_fraction={result['valid_fraction']}", flush=True)
            except Exception as exc:
                totals["failed"] += 1
                error = f"{type(exc).__name__}: {exc}"
                failures[key] = dict(tile_id=key[0], year=year, state=row["state"], error=error,
                                     failed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                print(f"{done}/{len(jobs)} FAILED: {key[0]} {year}: {error}", flush=True)
            write_csv(failures_path, failure_fields, [failures[k] for k in sorted(failures)])
    print(f"Finished: downloaded={totals['downloaded']} existing={totals['existing']} failed={totals['failed']} seconds={time.perf_counter()-started:.2f}")
    return 1 if totals["failed"] else 0


def status():
    rows = read_csv(OUT / "tiles_timeline_index.csv")
    manifest = {row["tile_id"]: row for row in read_csv(OUT / "selected_tiles.csv")}
    indexed = {(row["tile_id"], int(row["year"])): row for row in rows}
    manifest.update({row["tile_id"]: row for row in rows if row["tile_id"] not in manifest})
    print("Tiles per state (selected):", dict(sorted(Counter(row["state"] for row in manifest.values()).items())))
    disk_files = list((OUT / "tiles").glob("*.tif"))
    file_counts = Counter(path.stem.rsplit("_", 1)[-1] for path in disk_files)
    print("Files per year:", {year: file_counts[str(year)] for year in YEARS})
    present = [row for row in indexed.values() if Path(row["path"]).exists()]
    low = sum(float(row["valid_fraction"]) < 0.7 for row in present)
    print(f"Indexed tile-years with files: {len(present)}")
    print(f"valid_fraction < 0.7: {low}/{len(present)} ({low / len(present):.2%})" if present else "valid_fraction < 0.7: N/A (no indexed files)")
    print(f"Unresolved failures: {len(read_csv(OUT / 'failures.csv'))}")
    print("Disk usage (du -sh):", flush=True)
    if OUT.exists():
        subprocess.run(["du", "-sh", str(OUT)], check=True)
    else:
        print(f"{OUT} does not exist")
    incomplete = 0
    for tile_id in sorted(manifest):
        missing = [year for year in YEARS if not (OUT / "tiles" / f"{tile_id}_{year}.tif").exists()]
        if missing:
            incomplete += 1
            print(f"Missing years: {tile_id} ({manifest[tile_id]['state']}): {missing}")
    print(f"Tiles missing one or more of the five years: {incomplete}/{len(manifest)}")


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("availability")
    sub.add_parser("states-report")
    sub.add_parser("score-baselines", help="Create missing rule scores; refuses to overwrite an existing summary")
    command = sub.add_parser("build")
    command.add_argument("--states", required=True)
    command.add_argument("--tiles-per-state", type=positive, default=100)
    command.add_argument("--workers", type=positive, default=8)
    command.add_argument("--limit-tiles", type=positive)
    check = sub.add_parser("selection-check", help="Check 50/30/20 selection without downloading or writing")
    check.add_argument("--states", required=True)
    check.add_argument("--tiles-per-state", type=positive, default=100)
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "build":
            return build(args)
        if args.command == "selection-check":
            selection_check(args)
            return 0
        {"availability": availability, "states-report": states_report,
         "score-baselines": score_baselines, "status": status}[args.command]()
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
