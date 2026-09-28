from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    if geometry.get("type") == "Polygon":
        return [geometry.get("coordinates", [[]])[0]]
    if geometry.get("type") == "MultiPolygon":
        return [polygon[0] for polygon in geometry.get("coordinates", []) if polygon]
    return []


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    j = len(ring) - 1
    for i, point in enumerate(ring):
        xi, yi = point[:2]
        xj, yj = ring[j][:2]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_intersect = (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def point_in_aoi(lon: float | None, lat: float | None, aoi: dict[str, Any]) -> bool:
    if lon is None or lat is None:
        return False
    return any(_point_in_ring(float(lon), float(lat), ring) for ring in _as_rings(aoi))


class JobManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._reviews: list[dict[str, Any]] = []
        self._counter = 0
        # results() calls status() while holding this lock, so this must permit
        # that same thread to re-enter instead of deadlocking the API request.
        self._lock = threading.RLock()
        self._load_reviews()

    def _load_reviews(self) -> None:
        path = self.data_dir / "change" / "aoi_review_log.csv"
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as stream:
            self._reviews = list(csv.DictReader(stream))

    def _append_review(self, row: dict[str, Any]) -> None:
        path = self.data_dir / "change" / "aoi_review_log.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fields = ["timestamp_utc", "job_id", "tile_id", "category", "decision", "reviewer",
                  "note", "total_change_ha", "source"]
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fields})

    def create_job(self, aoi: dict[str, Any], name: str | None, store: Any) -> dict[str, Any]:
        if not _as_rings(aoi):
            raise ValueError("AOI must be a GeoJSON Polygon or MultiPolygon")
        with self._lock:
            self._counter += 1
            job_id = f"aoi_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{self._counter}"
            self._jobs[job_id] = {"job_id": job_id, "status": "queued", "name": name or "",
                                  "total": 0, "done": 0, "error": None, "aoi": aoi, "results": []}
        thread = threading.Thread(target=self._run_job, args=(job_id, store), daemon=True)
        thread.start()
        return self.status(job_id)

    def create_temporal_job(self, aoi: dict[str, Any], name: str | None, store: Any,
                            year_start: int, year_end: int, max_pairs: int) -> dict[str, Any]:
        if not _as_rings(aoi):
            raise ValueError("AOI must be a GeoJSON Polygon or MultiPolygon")
        if year_start >= year_end:
            raise ValueError("year_start must be earlier than year_end")
        with self._lock:
            self._counter += 1
            job_id = f"temporal_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{self._counter}"
            self._jobs[job_id] = {"job_id": job_id, "status": "queued", "name": name or "",
                                  "total": 0, "done": 0, "error": None, "aoi": aoi,
                                  "results": [], "kind": "temporal", "year_start": year_start,
                                  "year_end": year_end, "max_pairs": max_pairs, "candidate_pairs": 0}
        thread = threading.Thread(target=self._run_temporal_job, args=(job_id, store), daemon=True)
        thread.start()
        return self.status(job_id)

    def _run_temporal_job(self, job_id: str, store: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
        try:
            from change_detection_ml import detect_change_ml
            from multi_temporal import enrich_change_result

            aoi = self._jobs[job_id]["aoi"]
            year_start, year_end = self._jobs[job_id]["year_start"], self._jobs[job_id]["year_end"]
            candidates: list[tuple[str, str, int, int]] = []
            for tile_id, observations in store.observations.items():
                main = store.main.get(tile_id, {})
                if not point_in_aoi(main.get("lon"), main.get("lat"), aoi):
                    continue
                years = sorted(year for year in observations if year_start <= year <= year_end
                               and observations[year].get("path"))
                for year_a, year_b in zip(years, years[1:]):
                    candidates.append((tile_id, str(main.get("category", "")), year_a, year_b))
            candidates.sort(key=lambda pair: (pair[0], pair[2], pair[3]))
            candidate_count = len(candidates)
            scheduled = candidates[:self._jobs[job_id]["max_pairs"]]
            with self._lock:
                self._jobs[job_id]["candidate_pairs"] = candidate_count
                self._jobs[job_id]["total"] = len(scheduled)
                self._jobs[job_id]["truncated"] = candidate_count > len(scheduled)

            results: list[dict[str, Any]] = []
            for tile_id, category, year_a, year_b in scheduled:
                observations = store.observations[tile_id]
                path_a = self._resolve_image_path(store, observations[year_a].get("path"))
                path_b = self._resolve_image_path(store, observations[year_b].get("path"))
                row = detect_change_ml(path_a, path_b, tile_id, category,
                                       str(self.data_dir / "change"), year_a, year_b)
                row = enrich_change_result(row, path_a, path_b)
                row["bounds"] = store.bounds.get(tile_id)
                if row.get("status") == "ok":
                    details_path = (self.data_dir / "change" / category /
                                    f"{tile_id}_{year_a}_{year_b}_ml_analysis.json")
                    details_path.write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
                    row["analysis_details_path"] = str(details_path)
                results.append(row)
                with self._lock:
                    self._jobs[job_id]["done"] += 1
                    self._jobs[job_id]["results"] = list(results)
            with self._lock:
                self._jobs[job_id]["results"] = results
                self._jobs[job_id]["status"] = "done"
        except Exception as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)

    @staticmethod
    def _resolve_image_path(store: Any, value: str | None) -> str:
        if not value:
            raise FileNotFoundError("Observation row has no image path")
        path = Path(value)
        if path.is_absolute():
            return str(path)
        candidates = (Path.cwd() / path, store.data_dir.parent / path, store.data_dir / path)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"Observation image not found: {value}")

    def create_job_from_geojson(self, payload: dict[str, Any], store: Any) -> tuple[dict[str, Any], int | None]:
        used_feature_index: int | None = None
        name = payload.get("name") if isinstance(payload.get("name"), str) else None
        if payload.get("type") == "FeatureCollection":
            features = payload.get("features", [])
            for index, feature in enumerate(features):
                geometry = feature.get("geometry") if isinstance(feature, dict) else None
                if isinstance(geometry, dict) and _as_rings(geometry):
                    used_feature_index = index
                    name = name or feature.get("properties", {}).get("name")
                    return self.create_job(geometry, name, store), used_feature_index
            raise ValueError("FeatureCollection does not contain a Polygon or MultiPolygon")
        if payload.get("type") == "Feature":
            geometry = payload.get("geometry")
            if not isinstance(geometry, dict):
                raise ValueError("GeoJSON Feature has no geometry")
            name = name or payload.get("properties", {}).get("name")
            return self.create_job(geometry, name, store), used_feature_index
        return self.create_job(payload, name, store), used_feature_index

    def _run_job(self, job_id: str, store: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
        try:
            rows = list(store.change_ml.values()) or list(store.change.values())
            results: list[dict[str, Any]] = []
            with self._lock:
                self._jobs[job_id]["total"] = len(rows)
            for row in rows:
                tile_id = row.get("tile_id")
                main = store.main.get(tile_id, {})
                if point_in_aoi(main.get("lon"), main.get("lat"), self._jobs[job_id]["aoi"]):
                    result = dict(row)
                    result["confirmed"] = None
                    result["note"] = ""
                    result["bounds"] = store.bounds.get(tile_id)
                    results.append(result)
                with self._lock:
                    self._jobs[job_id]["done"] += 1
            with self._lock:
                self._jobs[job_id]["results"] = results
                self._jobs[job_id]["status"] = "done"
        except Exception as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            keys = ("job_id", "status", "name", "total", "done", "error", "kind",
                    "year_start", "year_end", "candidate_pairs", "truncated")
            return {key: job.get(key) for key in keys if key in job}

    def results(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            if job["status"] != "done":
                raise RuntimeError("AOI job is still running")
            return {**self.status(job_id), "results": list(job["results"])}

    def decide(self, job_id: str, tile_id: str, decision: str, reviewer: str, note: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            for row in job["results"]:
                if row.get("tile_id") == tile_id:
                    row["confirmed"] = decision == "confirmed"
                    row["note"] = note
                    review = {"timestamp_utc": _utc_now(), "job_id": job_id, "tile_id": tile_id,
                              "category": row.get("category"), "decision": decision,
                              "reviewer": reviewer, "note": note,
                              "total_change_ha": row.get("total_change_ha"),
                              "source": row.get("change_source") or row.get("source")}
                    self._reviews.insert(0, review)
                    self._append_review(review)
                    return dict(row)
            raise ValueError("Tile is not part of this AOI job")

    def reviews(self) -> dict[str, Any]:
        with self._lock:
            return {"decisions": list(self._reviews)}


def load_geojson_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
