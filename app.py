from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyproj import Transformer

from aoi_jobs import JobManager

FLOAT_FIELDS = ("crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction",
                "ndvi_mean", "ndbi_mean", "lon", "lat")
INT_FIELDS = ("epsg", "year")
TEXT_TEMPLATE = "a satellite image of {}"


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def number(value: Any, integer: bool = False) -> int | float | None:
    try:
        parsed = int(value) if integer else float(value)
        return parsed
    except (TypeError, ValueError):
        return None


class DataStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._transformers: dict[int, Transformer] = {}
        self.load()

    def load(self) -> None:
        self.ids: list[str] = []
        self.emb = np.zeros((0, 512), dtype=np.float32)
        self.sat_ids: list[str] = []
        self.sat_emb = np.zeros((0, 2048), dtype=np.float32)
        self.sat_search = np.zeros((0, 2048), dtype=np.float32)
        self.classifier: dict[str, Any] | None = None
        self.main: dict[str, dict[str, Any]] = {}
        self.years: dict[str, list[int]] = {}
        self.observations: dict[str, dict[int, dict[str, Any]]] = {}
        self.change: dict[str, dict[str, Any]] = {}
        self.change_ml: dict[str, dict[str, Any]] = {}
        self.bounds: dict[str, list[list[float | None]]] = {}
        index = self.data_dir / "tiles_index.csv"
        rows: dict[str, list[dict[str, Any]]] = {}
        if index.exists():
            with index.open(newline="", encoding="utf-8") as stream:
                for raw in csv.DictReader(stream):
                    tile_id = raw.get("tile_id")
                    if not tile_id:
                        continue
                    row = self._convert(raw)
                    rows.setdefault(tile_id, []).append(row)
                    if row.get("year") is not None:
                        self.years.setdefault(tile_id, []).append(row["year"])
                        self.observations.setdefault(tile_id, {})[row["year"]] = row
                    if row.get("year") == 2025 and raw.get("thumb"):
                        thumb = Path(raw["thumb"])
                        thumb_exists = thumb.exists() if thumb.is_absolute() else (
                            (self.data_dir.parent / thumb).exists() or (self.data_dir / thumb).exists()
                        )
                        if thumb_exists:
                            self.main[tile_id] = row
        self.years = {key: sorted(set(value)) for key, value in self.years.items()}
        emb_path = self.data_dir / "embeddings" / "embeddings.npy"
        ids_path = self.data_dir / "embeddings" / "tile_ids.json"
        if emb_path.exists() and ids_path.exists():
            try:
                emb = np.asarray(np.load(emb_path), dtype=np.float32)
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                keep = [i for i, tile_id in enumerate(ids) if i < len(emb) and tile_id in self.main]
                self.ids = [ids[i] for i in keep]
                self.emb = emb[keep]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.ids, self.emb = [], np.zeros((0, 512), dtype=np.float32)
        sat_emb_path = self.data_dir / "embeddings" / "sat_embeddings.npy"
        sat_ids_path = self.data_dir / "embeddings" / "sat_tile_ids.json"
        if sat_emb_path.exists() and sat_ids_path.exists():
            try:
                sat_emb = np.asarray(np.load(sat_emb_path), dtype=np.float32)
                sat_ids = json.loads(sat_ids_path.read_text(encoding="utf-8"))
                keep = []
                seen_sat_ids: set[str] = set()
                for i, tile_id in enumerate(sat_ids):
                    if i < len(sat_emb) and tile_id in self.main and tile_id not in seen_sat_ids:
                        keep.append(i)
                        seen_sat_ids.add(tile_id)
                self.sat_ids = [sat_ids[i] for i in keep]
                self.sat_emb = sat_emb[keep]
                if keep:
                    unit = self.sat_emb / np.maximum(np.linalg.norm(self.sat_emb, axis=1, keepdims=True), 1e-12)
                    unit -= unit.mean(axis=0, keepdims=True)
                    self.sat_search = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.sat_ids, self.sat_emb = [], np.zeros((0, 2048), dtype=np.float32)
                self.sat_search = np.zeros((0, 2048), dtype=np.float32)
        classifier_path = self.data_dir / "embeddings" / "category_classifier.pkl"
        if classifier_path.exists():
            try:
                candidate = joblib.load(classifier_path)
                if isinstance(candidate, dict) and {"scaler", "model", "classes"}.issubset(candidate):
                    self.classifier = candidate
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                self.classifier = None
        for row in self.main.values():
            row["ml_category"] = None
            row["ml_confidence"] = None
        if self.classifier is not None:
            try:
                scaler, model = self.classifier["scaler"], self.classifier["model"]
                for tile_id, embedding in zip(self.sat_ids, self.sat_emb):
                    probabilities = model.predict_proba(scaler.transform(embedding.reshape(1, -1)))[0]
                    best = int(np.argmax(probabilities))
                    self.main[tile_id]["ml_category"] = str(model.classes_[best])
                    self.main[tile_id]["ml_confidence"] = float(probabilities[best])
            except (ValueError, TypeError, AttributeError, KeyError):
                # A stale/incompatible optional artifact must not prevent the app from starting.
                for row in self.main.values():
                    row["ml_category"] = None
                    row["ml_confidence"] = None
        for tile_id in set(self.ids) | set(self.sat_ids):
            self.bounds[tile_id] = self._bounds(self.main[tile_id])
        summary = self.data_dir / "change" / "change_summary.csv"
        if summary.exists():
            with summary.open(newline="", encoding="utf-8") as stream:
                for raw in csv.DictReader(stream):
                    if raw.get("status") == "ok" and raw.get("year_a") == "2021" and raw.get("year_b") == "2025":
                        self.change[raw["tile_id"]] = self._convert_change(raw)
        summary_ml = self.data_dir / "change" / "change_summary_ml.csv"
        if summary_ml.exists():
            with summary_ml.open(newline="", encoding="utf-8") as stream:
                for raw in csv.DictReader(stream):
                    if raw.get("status") == "ok" and raw.get("year_a") == "2021" and raw.get("year_b") == "2025":
                        self.change_ml[raw["tile_id"]] = self._convert_change(raw)

    @staticmethod
    def _convert(raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        for key in FLOAT_FIELDS:
            row[key] = number(raw.get(key))
        for key in INT_FIELDS:
            row[key] = number(raw.get(key), integer=True)
        return row

    @staticmethod
    def _convert_change(raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        for key in ("year_a", "year_b"):
            row[key] = number(raw.get(key), integer=True)
        for key in ("valid_fraction", "changed_fraction", "new_built_ha", "vegetation_loss_ha",
                    "vegetation_gain_ha", "water_change_ha", "other_change_ha", "total_change_ha",
                    "mean_dndvi", "mean_dndbi"):
            row[key] = number(raw.get(key))
        row["suspicious"] = str(raw.get("suspicious", "")).lower() == "true"
        return row

    def _bounds(self, row: dict[str, Any]) -> list[list[float | None]]:
        epsg = row.get("epsg")
        coords = [(row.get("x0"), row.get("y0")), (row.get("x1"), row.get("y0")),
                  (row.get("x1"), row.get("y1")), (row.get("x0"), row.get("y1"))]
        if epsg is None or any(x is None or y is None for x, y in coords):
            return [[None, None], [None, None]]
        transformer = self._transformers.setdefault(epsg, Transformer.from_crs(epsg, 4326, always_xy=True))
        points = [transformer.transform(x, y) for x, y in coords]
        lons, lats = zip(*points)
        return [[min(lats), min(lons)], [max(lats), max(lons)]]

    def row(self, tile_id: str) -> dict[str, Any] | None:
        return self.main.get(tile_id)


class Filters(BaseModel):
    category: list[str] | None = None
    min_built: float | None = None
    max_built: float | None = None
    min_crop: float | None = None
    min_open: float | None = None
    max_water: float | None = None
    min_valid: float = 0.9


class SearchRequest(BaseModel):
    text: str | None = None
    like_tile_id: str | None = None
    k: int = Field(default=12)
    filters: Filters = Field(default_factory=Filters)


class AoiJobRequest(BaseModel):
    aoi: dict[str, Any]
    name: str | None = None


class TemporalAnalysisRequest(BaseModel):
    aoi: dict[str, Any]
    year_start: int = 2021
    year_end: int = 2025
    max_pairs: int = Field(default=100, ge=1, le=1000)
    name: str | None = None


class DiscoveryClustersRequest(BaseModel):
    category: str | None = None
    bbox: list[float] | None = None  # [west, south, east, north]
    n_clusters: int = Field(default=12, ge=2, le=20)
    refresh: bool = False


class AoiDecisionRequest(BaseModel):
    tile_id: str
    decision: str
    reviewer: str = "analyst"
    note: str = ""


def apply_filters(store: DataStore, filters: Filters, ids: list[str] | None = None) -> np.ndarray:
    ids = store.ids if ids is None else ids
    mask = np.ones(len(ids), dtype=bool)
    for i, tile_id in enumerate(ids):
        row = store.main[tile_id]
        if filters.category and row.get("category") not in filters.category:
            mask[i] = False
            continue
        checks = (("built_frac", filters.min_built, lambda a, b: a >= b),
                  ("built_frac", filters.max_built, lambda a, b: a <= b),
                  ("crop_frac", filters.min_crop, lambda a, b: a >= b),
                  ("open_frac", filters.min_open, lambda a, b: a >= b),
                  ("water_frac", filters.max_water, lambda a, b: a <= b),
                  ("valid_fraction", filters.min_valid, lambda a, b: a >= b))
        for key, bound, predicate in checks:
            value = row.get(key)
            if bound is not None and (value is None or not predicate(value, bound)):
                mask[i] = False
                break
    return mask


def create_app(data_dir: Path = Path("india_tiles"), text_embedder: Callable | None = None) -> FastAPI:
    store = DataStore(Path(data_dir))
    aoi_manager = JobManager(Path(data_dir))
    app = FastAPI(title="SURAG Tile Search")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/files/thumbs", StaticFiles(directory=str(store.data_dir / "thumbs"), check_dir=False), name="thumbs")
    app.mount("/files/change", StaticFiles(directory=str(store.data_dir / "change"), check_dir=False), name="change")
    dist = Path(__file__).parent / "frontend" / "dist"

    def health() -> dict[str, Any]:
        return {"status": "ok", "tiles_indexed": len(store.main), "tiles_embedded": len(store.ids),
                "tiles_with_change": len(store.change)}

    def public_result(tile_id: str, score: float, rank: int) -> dict[str, Any]:
        row = store.main[tile_id]
        change = store.change_ml.get(tile_id, store.change.get(tile_id, {}))
        return clean({"rank": rank, "tile_id": tile_id, "score": float(score),
                      "category": row.get("category"), "lat": row.get("lat"), "lon": row.get("lon"),
                      **{key: row.get(key) for key in ("crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction", "ndvi_mean")},
                      "thumb_url": f"/files/thumbs/{tile_id}.jpg", "bounds": store.bounds[tile_id],
                      "has_change": tile_id in store.change_ml or tile_id in store.change,
                      "change_total_ha": change.get("total_change_ha")})

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return health()

    @app.get("/api/stats")
    def api_stats() -> dict[str, Any]:
        categories: dict[str, int] = {}
        for row in store.main.values():
            categories[row["category"]] = categories.get(row["category"], 0) + 1
        years = {str(year): sum(year in values for values in store.years.values()) for year in (2021, 2025)}
        return {"embedded_by_category": categories, "with_change": len(store.change), "years": years}

    @app.post("/api/search")
    def api_search(request: SearchRequest) -> dict[str, Any]:
        has_text = request.text is not None
        has_like = request.like_tile_id is not None
        if has_text == has_like:
            raise HTTPException(400, "Provide exactly one of text or like_tile_id")
        if has_text and not request.text.strip():
            raise HTTPException(400, "Text query cannot be empty")
        # TorchGeo's encoder has no paired text model, so text search stays CLIP-based.
        search_ids, search_emb = store.ids, store.emb
        if has_like and len(store.sat_search) and request.like_tile_id in store.sat_ids:
            search_ids, search_emb = store.sat_ids, store.sat_search
        if has_like and request.like_tile_id not in search_ids:
            raise HTTPException(404, "Tile is not embedded")
        if not len(search_ids):
            return {"query": {"text": request.text, "like_tile_id": request.like_tile_id, "k": max(1, min(request.k, 100))},
                    "searched_tiles": 0, "results": []}
        try:
            if has_text:
                embed = text_embedder
                if embed is None:
                    from semantic_search import embed_texts
                    embed = embed_texts
                query = np.asarray(embed([TEXT_TEMPLATE.format(request.text.strip())]))[0]
            else:
                query = search_emb[search_ids.index(request.like_tile_id)]
        except Exception as exc:
            if has_text:
                raise HTTPException(503, f"Text search is unavailable: {str(exc)[:120]}")
            raise
        scores = np.clip(search_emb @ query, -1.0, 1.0) if has_like else search_emb @ query
        mask = apply_filters(store, request.filters, search_ids)
        if has_like:
            mask[search_ids.index(request.like_tile_id)] = False
        valid = np.where(mask)[0]
        k = min(max(1, min(request.k, 100)), len(valid))
        order = valid[np.argsort(scores[valid])[::-1][:k]] if k else []
        results = [public_result(search_ids[index], scores[index], rank) for rank, index in enumerate(order, 1)]
        return clean({"query": {"text": request.text, "like_tile_id": request.like_tile_id, "k": k},
                      "searched_tiles": len(search_ids), "results": results})

    @app.get("/api/discovery/similar/{tile_id}")
    def api_discovery_similar(tile_id: str, k: int = 20, category: str | None = None) -> dict[str, Any]:
        if tile_id not in store.sat_ids or not len(store.sat_search):
            raise HTTPException(404, "Satellite embedding not found for tile")
        query_index = store.sat_ids.index(tile_id)
        matrix = store.sat_search
        query = matrix[query_index]
        scores = np.clip(matrix @ query, -1.0, 1.0)
        mask = np.array([candidate != tile_id and candidate in store.main and
                         (category is None or store.main[candidate].get("category") == category)
                         for candidate in store.sat_ids], dtype=bool)
        positions = np.flatnonzero(mask)
        positions = positions[np.argsort(scores[positions])[::-1][:max(1, min(k, 100))]]
        results = [public_result(store.sat_ids[pos], float(scores[pos]), rank)
                   for rank, pos in enumerate(positions, 1)]
        return clean({"query_tile_id": tile_id, "method": "mean-centered TorchGeo Sentinel-2 embedding cosine similarity",
                      "searched_tiles": int(len(store.sat_ids)), "results": results})

    @app.post("/api/discovery/clusters")
    def api_discovery_clusters(request: DiscoveryClustersRequest) -> dict[str, Any]:
        if not (store.data_dir / "embeddings" / "sat_embeddings.npy").exists():
            raise HTTPException(503, "Satellite embeddings are unavailable")
        if request.bbox is not None and (len(request.bbox) != 4 or request.bbox[0] >= request.bbox[2]
                                         or request.bbox[1] >= request.bbox[3]):
            raise HTTPException(400, "bbox must be [west, south, east, north]")
        try:
            from discovery_clustering import get_or_build_clusters
            artifact = get_or_build_clusters(store.data_dir, request.n_clusters, request.refresh)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        groups = []
        visible_counts = {str(index): 0 for index in range(request.n_clusters)}
        for tile_id, cluster_id in artifact["cluster_of"].items():
            row = store.main.get(tile_id) or store.observations.get(tile_id, {}).get(2021)
            if row is None or (request.category and row.get("category") != request.category):
                continue
            if request.bbox:
                west, south, east, north = request.bbox
                lon, lat = row.get("lon"), row.get("lat")
                if lon is None or lat is None or not (west <= lon <= east and south <= lat <= north):
                    continue
            visible_counts[str(cluster_id)] += 1
        for cluster_id in range(request.n_clusters):
            key = str(cluster_id)
            reps = artifact["representatives"][key]
            category_counts: dict[str, int] = {}
            category_counts.update(artifact["quality"]["purity_by_cluster"][key]["category_counts"])
            sites = [{"tile_id": tile_id, "category": store.main[tile_id].get("category"),
                      "score": score, "bounds": store.bounds[tile_id]}
                     for tile_id, score in zip(reps, artifact["representative_scores"][key])
                     if tile_id in store.main]
            groups.append({"cluster_id": cluster_id, "count": visible_counts[key],
                           "full_catalog_count": artifact["sizes"][key],
                           "category_counts": category_counts, "representatives": reps,
                           "representative_tile_id": reps[0] if reps else None,
                           "dominant_category_share": artifact["quality"]["purity_by_cluster"][key]["dominant_category_share"],
                           "sites": sites})
        return clean({"method": "L2 + PCA(64) + MiniBatchKMeans on all satellite embedding rows",
                      "n_tiles": sum(visible_counts.values()),
                      "full_catalog_tiles": len(artifact["cluster_of"]),
                      "fit_embedding_rows": artifact["params"]["fit_embedding_rows"],
                      "n_clusters": request.n_clusters, "sampled": False,
                      "category": request.category, "bbox": request.bbox,
                      "quality": {"silhouette_2000_tile_sample": artifact["quality"]["silhouette_2000_tile_sample"],
                                  "sample_size": artifact["quality"]["sample_size"],
                                  "overall_weighted_purity": artifact["quality"]["overall_weighted_purity"]},
                      "comparisons": {k: {"silhouette_2000_tile_sample": value["silhouette_2000_tile_sample"],
                                          "overall_weighted_purity": value["overall_weighted_purity"]}
                                      for k, value in artifact["comparisons"].items()},
                      "clusters": groups})

    @app.get("/api/discovery/cluster/{tile_id}")
    def api_discovery_tile_cluster(tile_id: str) -> dict[str, Any]:
        from discovery_clustering import DEFAULT_K, get_or_build_clusters
        path = store.data_dir / "embeddings" / "clusters.json"
        configured_k = DEFAULT_K
        if path.exists():
            configured_k = int(json.loads(path.read_text(encoding="utf-8"))["params"]["n_clusters"])
        try:
            artifact = get_or_build_clusters(store.data_dir, configured_k)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if tile_id not in artifact["cluster_of"]:
            raise HTTPException(404, "Tile has no satellite embedding")
        cluster_id = artifact["cluster_of"][tile_id]
        return {"tile_id": tile_id, "cluster_id": cluster_id,
                "cluster_size": artifact["sizes"][str(cluster_id)],
                "representatives": artifact["representatives"][str(cluster_id)],
                "n_clusters": configured_k}

    @app.get("/api/tiles/{tile_id}")
    def api_tile(tile_id: str) -> dict[str, Any]:
        row = store.row(tile_id)
        if row is None:
            raise HTTPException(404, "Tile not found")
        return clean({"tile_id": tile_id, "category": row.get("category"), "lat": row.get("lat"), "lon": row.get("lon"),
                      "epsg": row.get("epsg"), "smod_code": row.get("smod_code"),
                      **{key: row.get(key) for key in ("crop_frac", "built_frac", "open_frac", "water_frac", "valid_fraction", "ndvi_mean", "ndbi_mean")},
                      "ml_category": row.get("ml_category"), "ml_category_confidence": row.get("ml_confidence"),
                      "years_available": store.years.get(tile_id, []), "thumb_url": f"/files/thumbs/{tile_id}.jpg",
                      "bounds": store.bounds.get(tile_id), "maps_url": f"https://www.google.com/maps/@{row.get('lat')},{row.get('lon')},15z/data=!3m1!1e3",
                      "change": store.change_ml.get(tile_id, store.change.get(tile_id))})

    @app.get("/api/tiles/{tile_id}/change")
    def api_change(tile_id: str) -> dict[str, Any]:
        summary = store.change_ml.get(tile_id, store.change.get(tile_id))
        if summary is None:
            raise HTTPException(404, "No change result for this tile")
        category = summary.get("category", store.main.get(tile_id, {}).get("category", ""))
        base = store.data_dir / "change" / str(category)
        ml = tile_id in store.change_ml
        suffix = "_ml" if ml else ""
        polygons_path = Path(summary.get("polygons_path") or base / f"{tile_id}_2021_2025{suffix}_polygons.geojson")
        polygons = {"type": "FeatureCollection", "features": []}
        if polygons_path.exists():
            polygons = json.loads(polygons_path.read_text(encoding="utf-8"))
        preview_path = Path(summary.get("preview_path") or base / f"{tile_id}_2021_2025{suffix}_preview.png")
        try:
            preview_relative = preview_path.resolve().relative_to((store.data_dir / "change").resolve())
        except ValueError:
            preview_relative = Path(str(category)) / preview_path.name
        return clean({"summary": summary, "preview_url": f"/files/change/{preview_relative.as_posix()}",
                      "polygons": polygons})

    @app.post("/api/reload")
    def api_reload() -> dict[str, Any]:
        store.load()
        return health()

    @app.post("/api/aoi/jobs", status_code=201)
    def api_create_aoi_job(request: AoiJobRequest) -> dict[str, Any]:
        try:
            return aoi_manager.create_job(request.aoi, request.name, store)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/aoi/change-analysis", status_code=202)
    def api_create_temporal_analysis(request: TemporalAnalysisRequest) -> dict[str, Any]:
        try:
            return aoi_manager.create_temporal_job(request.aoi, request.name, store,
                                                   request.year_start, request.year_end, request.max_pairs)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/aoi/jobs/upload", status_code=201)
    async def api_upload_aoi_job(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "aoi.geojson").suffix or ".geojson"
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(await file.read())
        try:
            payload = json.loads(temp_path.read_text(encoding="utf-8"))
            result, used_feature_index = aoi_manager.create_job_from_geojson(payload, store)
            if used_feature_index is not None:
                result["used_feature_index"] = used_feature_index
            return result
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid GeoJSON: {exc}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        finally:
            temp_path.unlink(missing_ok=True)

    @app.get("/api/aoi/jobs/{job_id}")
    def api_aoi_job_status(job_id: str) -> dict[str, Any]:
        try:
            return aoi_manager.status(job_id)
        except KeyError:
            raise HTTPException(404, "AOI job not found")

    @app.get("/api/aoi/jobs/{job_id}/results")
    def api_aoi_job_results(job_id: str) -> dict[str, Any]:
        try:
            return clean(aoi_manager.results(job_id))
        except KeyError:
            raise HTTPException(404, "AOI job not found")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/aoi/jobs/{job_id}/decision")
    def api_aoi_decision(job_id: str, request: AoiDecisionRequest) -> dict[str, Any]:
        if request.decision not in {"confirmed", "rejected"}:
            raise HTTPException(400, "decision must be confirmed or rejected")
        try:
            return clean(aoi_manager.decide(job_id, request.tile_id, request.decision, request.reviewer, request.note))
        except KeyError:
            raise HTTPException(404, "AOI job not found")
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/aoi/reviews")
    def api_aoi_reviews() -> dict[str, Any]:
        return clean(aoi_manager.reviews())

    @app.get("/")
    def root() -> Any:
        if dist.exists() and (dist / "index.html").exists():
            from fastapi.responses import FileResponse
            return FileResponse(dist / "index.html")
        return JSONResponse({"detail": "Frontend not built yet. Run: cd frontend && npm install && npm run build"})

    app.state.store = store
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


if os.environ.get("SURAG_NO_AUTOAPP") != "1":
    app = create_app()
