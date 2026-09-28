#!/usr/bin/env python3
"""Train a supervised category classifier over Sentinel-2 tile embeddings."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_CACHED_MODEL_PATH: str | None = None
_CACHED_ARTIFACT: dict | None = None


def load_data(
    index_csv: str = "india_tiles/tiles_index.csv",
    emb_dir: str = "india_tiles/embeddings",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return embeddings joined to main index rows, preserving embedding order."""
    emb_path = Path(emb_dir)
    embeddings = np.load(emb_path / "sat_embeddings.npy")
    with (emb_path / "sat_tile_ids.json").open(encoding="utf-8") as handle:
        embedding_ids = json.load(handle)
    if embeddings.ndim != 2 or embeddings.shape[1] != 2048:
        raise ValueError(f"Expected embeddings shaped (N, 2048), got {embeddings.shape}")
    if len(embeddings) != len(embedding_ids):
        raise ValueError("sat_embeddings.npy and sat_tile_ids.json lengths differ")

    index = pd.read_csv(index_csv, dtype={"tile_id": str})
    main_rows = index[index["thumb"].fillna("").ne("")][["tile_id", "category"]]
    # The source index can repeat an otherwise identical main row.  Retain the
    # first row per tile so every embedding remains one supervised example.
    main_rows = main_rows.drop_duplicates("tile_id", keep="first")
    order = pd.DataFrame({"tile_id": [str(value) for value in embedding_ids], "embedding_row": range(len(embedding_ids))})
    # Repeated source-index rows also produced repeated embedding IDs.  Keeping
    # the first ensures a physical tile cannot be present in both split sets.
    order = order.drop_duplicates("tile_id", keep="first")
    matched = order.merge(main_rows, on="tile_id", how="inner", validate="one_to_one", sort=False)
    print(f"Matched {len(matched)} embeddings to main index rows")
    return (
        embeddings[matched["embedding_row"].to_numpy()].astype(np.float32, copy=False),
        matched["category"].to_numpy(dtype=str),
        matched["tile_id"].tolist(),
    )


def _print_evaluation(y_test: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> tuple[float, dict, np.ndarray]:
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, labels=classes, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=classes, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    print("=== HELD-OUT TEST ACCURACY (not train) ===")
    print(f"overall accuracy: {accuracy:.6f}")
    print(report_text)
    print("Confusion matrix (rows = true category, columns = predicted category)")
    print("true\\pred\t" + "\t".join(classes))
    for category, row in zip(classes, matrix):
        print(category + "\t" + "\t".join(str(value) for value in row))
    return accuracy, report_dict, matrix


def train(
    index_csv: str = "india_tiles/tiles_index.csv",
    emb_dir: str = "india_tiles/embeddings",
    out_dir: str = "india_tiles/embeddings",
) -> None:
    """Fit and evaluate a balanced logistic-regression classifier."""
    X, y, tile_ids = load_data(index_csv, emb_dir)
    classes = sorted(np.unique(y).tolist())
    ids_train, ids_test, y_train, y_test = train_test_split(
        tile_ids, y, test_size=0.2, random_state=42, stratify=y
    )
    row_by_id = {tile_id: row for row, tile_id in enumerate(tile_ids)}
    X_train = X[[row_by_id[tile_id] for tile_id in ids_train]]
    X_test = X[[row_by_id[tile_id] for tile_id in ids_test]]
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))
    accuracy, report_dict, matrix = _print_evaluation(y_test, y_pred, classes)
    if accuracy < 0.60:
        counts = pd.Series(y).value_counts().sort_index()
        print("WARNING: accuracy below 0.60")
        print("Per-category sample counts: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
        print("Likely causes: too few samples for some category; embeddings may not separate these categories well;")
        print("open_land vs agriculture and urban_areas vs city_structures are inherently similar categories.")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifact = {"scaler": scaler, "model": model, "classes": classes}
    joblib.dump(artifact, out_path / "category_classifier.pkl")
    per_class = {category: report_dict[category] for category in classes}
    report = {
        "overall_accuracy": float(accuracy),
        "per_class_metrics": per_class,
        "confusion_matrix": matrix.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "per_category_counts_train": {key: int(value) for key, value in pd.Series(y_train).value_counts().sort_index().items()},
        "per_category_counts_test": {key: int(value) for key, value in pd.Series(y_test).value_counts().sort_index().items()},
    }
    with (out_path / "category_classifier_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def predict_category(
    embedding_2048: np.ndarray,
    model_path: str = "india_tiles/embeddings/category_classifier.pkl",
) -> tuple[str, float, dict[str, float]]:
    """Return the predicted category and calibrated class probabilities."""
    global _CACHED_MODEL_PATH, _CACHED_ARTIFACT
    resolved_path = str(Path(model_path).resolve())
    if _CACHED_ARTIFACT is None or _CACHED_MODEL_PATH != resolved_path:
        _CACHED_ARTIFACT = joblib.load(resolved_path)
        _CACHED_MODEL_PATH = resolved_path
    vector = np.asarray(embedding_2048, dtype=np.float32)
    if vector.shape != (2048,):
        raise ValueError(f"Expected one embedding shaped (2048,), got {vector.shape}")
    scaler = _CACHED_ARTIFACT["scaler"]
    model = _CACHED_ARTIFACT["model"]
    probabilities = model.predict_proba(scaler.transform(vector.reshape(1, -1)))[0]
    all_probs = {str(name): float(probability) for name, probability in zip(model.classes_, probabilities)}
    best_index = int(np.argmax(probabilities))
    return str(model.classes_[best_index]), float(probabilities[best_index]), all_probs


def _predict_from_tile_id(tile_id: str) -> None:
    emb_dir = Path("india_tiles/embeddings")
    embeddings = np.load(emb_dir / "sat_embeddings.npy")
    with (emb_dir / "sat_tile_ids.json").open(encoding="utf-8") as handle:
        embedding_ids = [str(value) for value in json.load(handle)]
    try:
        row = embedding_ids.index(tile_id)
    except ValueError as exc:
        raise ValueError(f"tile_id not found in embedding files: {tile_id}") from exc
    category, confidence, probabilities = predict_category(embeddings[row])
    index = pd.read_csv("india_tiles/tiles_index.csv", dtype={"tile_id": str})
    matched = index[(index["tile_id"] == tile_id) & index["thumb"].fillna("").ne("")]
    if matched.empty:
        raise ValueError(f"tile_id not found in a main index row: {tile_id}")
    threshold_category = str(matched.iloc[0]["category"])
    print(f"ML predicted category: {category}")
    print(f"ML confidence: {confidence:.6f}")
    print("All probabilities: " + json.dumps(probabilities, sort_keys=True))
    print(f"Threshold-based category: {threshold_category}")
    print(f"Agree: {'yes' if category == threshold_category else 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--tile-id")
    args = parser.parse_args()
    if args.command == "train":
        train()
    elif args.command == "predict":
        if not args.tile_id:
            parser.error("predict requires --tile-id")
        _predict_from_tile_id(args.tile_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
