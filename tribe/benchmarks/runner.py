"""Benchmark runner -- evaluate TRIBE v2 against manipulation datasets."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from tribe.backends.router import get_backend

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"


def run_benchmark(
    dataset_name: str,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run a single benchmark.

    Args:
        dataset_name: "semeval", "mentalmanip", or "paired".
        data_dir: Where to store/find downloaded datasets.
        results_dir: Where to save results JSON.

    Returns:
        Results dict with scores and metrics.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    items = _load_dataset(dataset_name, data_dir)
    scores = _run_analysis(items)
    metrics = _compute_metrics(dataset_name, items, scores)

    result = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total": len(items),
        "n_successful": len([s for s in scores if s["manipulation_score"] is not None]),
        "n_failed": len([s for s in scores if s["manipulation_score"] is None]),
        "scores": scores,
        "metrics": metrics,
    }

    output_path = results_dir / f"{dataset_name}_results.json"
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Results saved to %s", output_path)

    return result


def run_all(
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run all three benchmarks.

    Args:
        data_dir: Where to store/find downloaded datasets.
        results_dir: Where to save results JSON.

    Returns:
        Mapping of dataset name to results dict.
    """
    results: dict = {}
    for name in ["paired", "mentalmanip", "semeval"]:
        logger.info("=== Running %s benchmark ===", name)
        results[name] = run_benchmark(name, data_dir, results_dir)
    return results


def _load_dataset(dataset_name: str, data_dir: Path) -> list[dict]:
    """Load a dataset by name.

    Args:
        dataset_name: One of "semeval", "mentalmanip", or "paired".
        data_dir: Directory for downloaded datasets.

    Returns:
        List of item dicts with at least "id" and "text" keys.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name == "semeval":
        from tribe.benchmarks.datasets.semeval import download, load

        download(data_dir)
        return load(data_dir)
    elif dataset_name == "mentalmanip":
        from tribe.benchmarks.datasets.mentalmanip import download, load

        download(data_dir)
        return load(data_dir)
    elif dataset_name == "paired":
        from tribe.benchmarks.datasets.paired import load

        return load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _run_analysis(items: list[dict]) -> list[dict]:
    """Run TRIBE analysis on each item.

    Args:
        items: List of dataset items with "id" and "text" keys.

    Returns:
        List of score dicts, one per item.
    """
    backend = get_backend()
    scores: list[dict] = []
    total = len(items)

    for i, item in enumerate(items):
        start = time.monotonic()
        try:
            result = backend.analyze_text(item["text"])
            elapsed = time.monotonic() - start
            scores.append(
                {
                    "id": item["id"],
                    "text_preview": item["text"][:100],
                    "manipulation_score": result.manipulation_score,
                    "manipulation_ratio": (
                        result.neural.manipulation_ratio if result.neural else None
                    ),
                    "dominant_network": (result.neural.dominant_network if result.neural else None),
                    "primary_trigger": result.primary_trigger,
                    "label": _item_label(item),
                    "processing_time_ms": result.processing_time_ms,
                }
            )
            logger.info(
                "[%d/%d] %s -- score: %.1f, time: %.1fs",
                i + 1,
                total,
                item["id"],
                result.manipulation_score,
                elapsed,
            )
        except Exception as e:
            logger.error("[%d/%d] %s -- FAILED: %s", i + 1, total, item["id"], e)
            scores.append(
                {
                    "id": item["id"],
                    "text_preview": item["text"][:100],
                    "manipulation_score": None,
                    "error": str(e),
                    "label": _item_label(item),
                }
            )

    return scores


def _item_label(item: dict) -> int:
    """Extract a binary label from a dataset item.

    Args:
        item: Dataset item dict.

    Returns:
        1 for manipulative, 0 for non-manipulative.
    """
    if "manipulative" in item:
        return 1 if item["manipulative"] else 0
    if "propaganda_density" in item:
        return 1 if item["propaganda_density"] > 0.1 else 0
    return 0


def _compute_metrics(
    dataset_name: str,
    items: list[dict],
    scores: list[dict],
) -> dict:
    """Compute dataset-specific metrics from analysis scores.

    Args:
        dataset_name: Name of the dataset.
        items: Original dataset items.
        scores: Analysis score dicts from _run_analysis.

    Returns:
        Metrics dict appropriate for the dataset type.
    """
    valid_scores = [s for s in scores if s["manipulation_score"] is not None]

    if not valid_scores:
        return {}

    if dataset_name == "paired":
        return _metrics_paired(valid_scores)
    elif dataset_name == "mentalmanip":
        return _metrics_separation(valid_scores)
    elif dataset_name == "semeval":
        return _metrics_semeval(items, valid_scores)
    return {}


def _metrics_paired(valid_scores: list[dict]) -> dict:
    """Compute paired comparison metrics.

    Args:
        valid_scores: Score dicts with non-null manipulation_score.

    Returns:
        Dict with win_rate, mean_diff, t_statistic, p_value, n_pairs.
    """
    from tribe.benchmarks.metrics import compute_paired

    manip_scores = [s["manipulation_score"] for s in valid_scores if s["label"] == 1]
    neutral_scores = [s["manipulation_score"] for s in valid_scores if s["label"] == 0]
    paired = compute_paired(manip_scores, neutral_scores)

    return {
        "win_rate": paired.win_rate,
        "mean_diff": paired.mean_diff,
        "t_statistic": paired.t_statistic,
        "p_value": paired.p_value,
        "n_pairs": paired.n_pairs,
    }


def _metrics_separation(valid_scores: list[dict]) -> dict:
    """Compute binary separation metrics.

    Args:
        valid_scores: Score dicts with non-null manipulation_score.

    Returns:
        Dict with AUC-ROC, Cohen's d, and distribution stats.
    """
    from tribe.benchmarks.metrics import compute_separation

    manip_scores = [s["manipulation_score"] for s in valid_scores if s["label"] == 1]
    neutral_scores = [s["manipulation_score"] for s in valid_scores if s["label"] == 0]
    sep = compute_separation(manip_scores, neutral_scores)

    return {
        "auc_roc": sep.auc_roc,
        "cohens_d": sep.cohens_d,
        "mean_manipulative": sep.mean_positive,
        "std_manipulative": sep.std_positive,
        "mean_neutral": sep.mean_negative,
        "std_neutral": sep.std_negative,
        "n_manipulative": sep.n_positive,
        "n_neutral": sep.n_negative,
    }


def _metrics_semeval(items: list[dict], valid_scores: list[dict]) -> dict:
    """Compute SemEval correlation and quartile separation metrics.

    Args:
        items: Original dataset items with propaganda_density.
        valid_scores: Score dicts with non-null manipulation_score.

    Returns:
        Dict with Spearman, Pearson, and quartile-based separation.
    """
    from tribe.benchmarks.metrics import compute_correlation, compute_separation

    # Build aligned arrays of density and score
    score_by_id = {s["id"]: s["manipulation_score"] for s in valid_scores}
    densities: list[float] = []
    m_scores: list[float] = []
    for item in items:
        score_val = score_by_id.get(item["id"])
        if score_val is not None:
            densities.append(item["propaganda_density"])
            m_scores.append(score_val)

    if len(densities) < 4:
        return {}

    corr = compute_correlation(densities, m_scores)

    # Quartile separation: top vs bottom quartile by density
    sorted_pairs = sorted(zip(densities, m_scores), key=lambda x: x[0])
    q1 = len(sorted_pairs) // 4
    low_scores = [s for _, s in sorted_pairs[:q1]]
    high_scores = [s for _, s in sorted_pairs[-q1:]]
    sep = compute_separation(high_scores, low_scores)

    return {
        "spearman_rho": corr.spearman_rho,
        "spearman_p": corr.spearman_p,
        "pearson_r": corr.pearson_r,
        "auc_roc_quartile": sep.auc_roc,
        "cohens_d_quartile": sep.cohens_d,
        "mean_high_propaganda": sep.mean_positive,
        "mean_low_propaganda": sep.mean_negative,
        "n": corr.n,
    }
