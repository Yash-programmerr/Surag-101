"""Full-catalog clustering of TorchGeo Sentinel-2 tile embeddings."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits


DEFAULT_K = 12
COMPARISON_KS = (8, 12, 16)
PCA_DIMS = 64
SILHOUETTE_SAMPLE = 2000


def load_catalog(emb_dir: Path, index_csv: Path) -> tuple[np.ndarray, list[str], dict[str, str]]:
    ids = [str(value) for value in json.loads((emb_dir / "sat_tile_ids.json").read_text(encoding="utf-8"))]
    vectors = np.load(emb_dir / "sat_embeddings.npy")
    if vectors.ndim != 2 or vectors.shape[1] != 2048 or vectors.shape[0] != len(ids):
        raise ValueError(f"Expected matching (N, 2048) embeddings and IDs, got {vectors.shape} and {len(ids)} IDs")
    if not np.isfinite(vectors).all():
        raise ValueError("Satellite embedding array contains non-finite values")
    categories: dict[str, str] = {}
    with index_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            tile_id = row.get("tile_id")
            if tile_id and row.get("category") and tile_id not in categories:
                categories[tile_id] = row["category"]
    return np.asarray(vectors, dtype=np.float32), ids, categories


def _cluster_metrics(tile_vectors: np.ndarray, tile_labels: np.ndarray,
                     categories: np.ndarray, sample_indices: np.ndarray,
                     n_clusters: int) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    dominant_total = 0
    labeled_total = 0
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(tile_labels == cluster_id)
        counts = Counter(categories[members].tolist())
        counts.pop("unknown", None)
        labeled = sum(counts.values())
        dominant_name, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
        dominant_total += dominant_count
        labeled_total += labeled
        clusters[str(cluster_id)] = {
            "size": int(len(members)),
            "dominant_category": dominant_name,
            "dominant_category_share": round(dominant_count / labeled, 6) if labeled else None,
            "category_counts": dict(sorted(counts.items())),
        }
    sampled_labels = tile_labels[sample_indices]
    silhouette = (float(silhouette_score(tile_vectors[sample_indices], sampled_labels))
                  if len(np.unique(sampled_labels)) > 1 else None)
    return {
        "silhouette_2000_tile_sample": round(silhouette, 6) if silhouette is not None else None,
        "sample_size": int(len(sample_indices)),
        "overall_weighted_purity": round(dominant_total / labeled_total, 6) if labeled_total else None,
        "purity_by_cluster": clusters,
    }


def build_clusters(emb_dir: Path, index_csv: Path, n_clusters: int = DEFAULT_K) -> dict[str, Any]:
    """Fit on every embedding row, then resolve duplicate years to one ID label."""
    if not 2 <= n_clusters <= 20:
        raise ValueError("n_clusters must be between 2 and 20")
    vectors, ids, category_of = load_catalog(emb_dir, index_csv)
    unique_ids, inverse = np.unique(np.asarray(ids, dtype=str), return_inverse=True)
    if len(unique_ids) < max(n_clusters, PCA_DIMS + 1):
        raise ValueError(f"Need at least {max(n_clusters, PCA_DIMS + 1)} unique tiles for 64-D PCA")
    categories = np.asarray([category_of.get(tile_id, "unknown") for tile_id in unique_ids])
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(unique_ids), size=min(SILHOUETTE_SAMPLE, len(unique_ids)), replace=False)
    compare_ks = tuple(sorted(set(COMPARISON_KS + (n_clusters,))))
    with threadpool_limits(limits=4):
        # All 16,847 embedding rows participate. Repeated tile IDs correspond
        # to different year rows; their PCA coordinates are averaged only when
        # assigning the single cluster_of entry required for each tile ID.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        projected = PCA(n_components=PCA_DIMS, svd_solver="randomized", random_state=42).fit_transform(vectors)
        tile_vectors = np.zeros((len(unique_ids), PCA_DIMS), dtype=np.float32)
        np.add.at(tile_vectors, inverse, projected)
        tile_vectors /= np.maximum(np.bincount(inverse)[:, None], 1)
        quality_by_k: dict[str, dict[str, Any]] = {}
        selected_labels = None
        selected_model = None
        for k in compare_ks:
            model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3,
                                    batch_size=1024, max_iter=100)
            row_labels = model.fit_predict(projected)
            # The nearest learned centroid in mean PCA space gives one stable
            # assignment per tile, including all IDs repeated across years.
            tile_labels = model.predict(tile_vectors)
            metrics = _cluster_metrics(tile_vectors, tile_labels, categories, sample_indices, k)
            quality_by_k[str(k)] = metrics
            print(f"k={k}: silhouette_2000={metrics['silhouette_2000_tile_sample']} "
                  f"weighted_purity={metrics['overall_weighted_purity']}", flush=True)
            for cluster_id, info in metrics["purity_by_cluster"].items():
                print(f"  cluster {cluster_id}: size={info['size']} "
                      f"dominant_category={info['dominant_category']} "
                      f"dominant_share={info['dominant_category_share']}", flush=True)
            if k == n_clusters:
                selected_labels = tile_labels
                selected_model = model
            del row_labels, model
    assert selected_labels is not None and selected_model is not None
    cluster_of = {str(tile_id): int(cluster_id) for tile_id, cluster_id in zip(unique_ids, selected_labels)}
    sizes = {str(cluster_id): int(np.sum(selected_labels == cluster_id)) for cluster_id in range(n_clusters)}
    representatives: dict[str, list[str]] = {}
    representative_scores: dict[str, list[float]] = {}
    for cluster_id, center in enumerate(selected_model.cluster_centers_):
        positions = np.flatnonzero(selected_labels == cluster_id)
        distances = np.linalg.norm(tile_vectors[positions] - center, axis=1)
        nearest = np.argsort(distances)[:5]
        representatives[str(cluster_id)] = [str(unique_ids[positions[i]]) for i in nearest]
        representative_scores[str(cluster_id)] = [round(float(1.0 / (1.0 + distances[i])), 6) for i in nearest]
    artifact = {
        "params": {
            "n_clusters": n_clusters,
            "normalization": "L2 per embedding row",
            "pca_components": PCA_DIMS,
            "pca_random_state": 42,
            "model": "MiniBatchKMeans",
            "random_state": 42,
            "n_init": 3,
            "fit_embedding_rows": len(ids),
            "unique_tile_ids": len(unique_ids),
            "duplicate_year_rows": len(ids) - len(unique_ids),
            "category_label_source": "tiles_index.csv category; WorldCover/GHSL-derived proxy, not independent ground truth",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "cluster_of": cluster_of,
        "sizes": sizes,
        "representatives": representatives,
        "representative_scores": representative_scores,
        "quality": quality_by_k[str(n_clusters)],
        "comparisons": quality_by_k,
    }
    if len(artifact["cluster_of"]) != len(unique_ids) or set(artifact["cluster_of"]) != set(ids):
        raise RuntimeError("Cluster coverage check failed: not every embedded tile ID was assigned")
    return artifact


def save_clusters(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def get_or_build_clusters(data_dir: Path, n_clusters: int = DEFAULT_K,
                          refresh: bool = False) -> dict[str, Any]:
    data_dir = Path(data_dir)
    path = data_dir / "embeddings" / "clusters.json"
    if path.exists() and not refresh:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("params", {}).get("n_clusters") != n_clusters:
            raise ValueError("Cached cluster count differs; pass refresh=true to recompute")
        current_ids = set(json.loads((data_dir / "embeddings" / "sat_tile_ids.json").read_text(encoding="utf-8")))
        if set(artifact.get("cluster_of", {})) != current_ids:
            raise ValueError("Cached clusters do not cover current satellite embeddings; pass refresh=true")
        return artifact
    artifact = build_clusters(data_dir / "embeddings", data_dir / "tiles_index.csv", n_clusters)
    save_clusters(artifact, path)
    return artifact
