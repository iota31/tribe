"""Path C: Learned manipulation classifier from raw fMRI activations.

Collects raw 20,484-vertex activation vectors from labeled datasets,
reduces dimensionality with PCA, trains a logistic regression classifier.

The classifier learns WHICH activation patterns correspond to manipulation
from data, rather than using hand-tuned region weights.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CLASSIFIER_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"


def collect_activations(
    dataset_name: str,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect raw activation vectors from a labeled dataset.

    Saves incrementally to .npy files with checkpoint support.
    Returns (activations, labels, ids).
    """
    if output_dir is None:
        output_dir = CLASSIFIER_DIR
    if data_dir is None:
        data_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    from tribe.benchmarks.runner import _item_label, _load_dataset

    items = _load_dataset(dataset_name, data_dir)

    # Checkpoint: activations saved as individual .npy files
    checkpoint_dir = output_dir / f"{dataset_name}_activations"
    checkpoint_dir.mkdir(exist_ok=True)

    # Find already-completed items
    done_ids = {f.stem for f in checkpoint_dir.glob("*.npy")}
    remaining = [item for item in items if item["id"] not in done_ids]

    logger.info(
        "%s: %d total, %d done, %d remaining",
        dataset_name,
        len(items),
        len(done_ids),
        len(remaining),
    )

    if remaining:
        from tribe.backends.router import get_backend

        backend = get_backend()
        total = len(items)
        already_done = len(done_ids)

        for i, item in enumerate(remaining):
            idx = already_done + i + 1
            start = time.monotonic()

            try:
                activation = backend.get_raw_activation(item["text"])
                elapsed = time.monotonic() - start

                # Save individual activation
                np.save(str(checkpoint_dir / f"{item['id']}.npy"), activation)

                logger.info(
                    "[%d/%d] %s - %.1fs",
                    idx,
                    total,
                    item["id"],
                    elapsed,
                )
            except Exception as e:
                logger.error("[%d/%d] %s - FAILED: %s", idx, total, item["id"], e)

            gc.collect()

    # Load all activations
    activations = []
    labels = []
    ids = []

    for item in items:
        npy_path = checkpoint_dir / f"{item['id']}.npy"
        if npy_path.exists():
            act = np.load(str(npy_path))
            activations.append(act)
            labels.append(_item_label(item))
            ids.append(item["id"])

    if not activations:
        return np.array([]), np.array([]), []

    X = np.stack(activations)
    y = np.array(labels)

    logger.info("Collected %d activations, shape %s", len(X), X.shape)
    return X, y, ids


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 50,
    output_dir: Path | None = None,
) -> dict:
    """Train a logistic regression on PCA-reduced activations.

    Returns dict with model, metrics, and feature importances.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    if output_dir is None:
        output_dir = CLASSIFIER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training on %d samples, %d features", X.shape[0], X.shape[1])

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])

    # CRITICAL: PCA must be fit INSIDE the CV loop to avoid leakage
    # Use sklearn Pipeline to wrap PCA + LogisticRegression as a single estimator
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("pca", PCA(n_components=n_comp)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ]
    )

    # Cross-validated AUC (PCA refit on each fold's training data only)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    logger.info(
        "Leak-free CV AUC: %.4f +/- %.4f",
        cv_scores.mean(),
        cv_scores.std(),
    )

    # Now fit the final pipeline on all data for deployment
    pipeline.fit(X, y)
    pca = pipeline.named_steps["pca"]
    clf = pipeline.named_steps["clf"]
    explained = sum(pca.explained_variance_ratio_)
    logger.info(
        "Final fit: PCA %d -> %d, %.1f%% variance",
        X.shape[1],
        n_comp,
        explained * 100,
    )

    # Full-data AUC (overfit indicator vs CV)
    y_proba = pipeline.predict_proba(X)[:, 1]
    full_auc = roc_auc_score(y, y_proba)

    # Save model
    import joblib

    model_path = output_dir / "manipulation_classifier.pkl"
    joblib.dump({"pca": pca, "clf": clf}, str(model_path))
    logger.info("Model saved to %s", model_path)

    return {
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "full_auc": round(float(full_auc), 4),
        "n_samples": int(X.shape[0]),
        "n_components": int(n_comp),
        "variance_explained": round(float(explained), 4),
        "model_path": str(model_path),
    }


def predict(text: str, model_path: Path | None = None) -> float:
    """Predict manipulation probability for a text using the trained classifier.

    Returns probability 0.0 to 1.0.
    """
    import joblib

    if model_path is None:
        model_path = CLASSIFIER_DIR / "manipulation_classifier.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained classifier at {model_path}. Run: tribe bench train-classifier"
        )

    model = joblib.load(str(model_path))
    pca = model["pca"]
    clf = model["clf"]

    from tribe.backends.router import get_backend

    backend = get_backend()
    activation = backend.get_raw_activation(text)
    X = pca.transform(activation.reshape(1, -1))
    proba = clf.predict_proba(X)[0, 1]

    return float(proba)
