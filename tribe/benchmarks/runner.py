"""Benchmark runner - evaluate TRIBE v2 against manipulation datasets.

Designed for long runs (hours) with crash resilience:
- Results written incrementally to JSONL (one line per item)
- Automatic resume from checkpoint on restart
- Low memory footprint (no accumulation in RAM)
- GC after each inference to prevent memory buildup
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"


def run_benchmark(
    dataset_name: str,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run a single benchmark with checkpoint/resume support.

    Results are written incrementally to a .jsonl file (one line per item).
    If the run crashes and restarts, it resumes from the last completed item.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    items = _load_dataset(dataset_name, data_dir)

    # Checkpoint file: one JSON object per line
    jsonl_path = results_dir / f"{dataset_name}_scores.jsonl"
    completed_ids = _load_checkpoint(jsonl_path)

    remaining = [item for item in items if item["id"] not in completed_ids]
    logger.info(
        "%s: %d total, %d already done, %d remaining",
        dataset_name,
        len(items),
        len(completed_ids),
        len(remaining),
    )

    if remaining:
        _run_incremental(remaining, jsonl_path, len(items), len(completed_ids))

    # Compute final metrics from the JSONL file
    all_scores = _read_all_scores(jsonl_path)
    metrics = _compute_metrics(dataset_name, items, all_scores)

    result = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total": len(items),
        "n_successful": len([s for s in all_scores if s.get("manipulation_score") is not None]),
        "n_failed": len([s for s in all_scores if s.get("manipulation_score") is None]),
        "scores": all_scores,
        "metrics": metrics,
    }

    # Write final summary JSON
    output_path = results_dir / f"{dataset_name}_results.json"
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Final results saved to %s", output_path)

    return result


def run_all(
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run all benchmarks sequentially."""
    results: dict = {}
    for name in ["paired", "mentalmanip", "semeval"]:
        logger.info("=== Running %s benchmark ===", name)
        results[name] = run_benchmark(name, data_dir, results_dir)
    return results


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def _load_checkpoint(jsonl_path: Path) -> set[str]:
    """Load IDs of already-completed items from the JSONL checkpoint file."""
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completed.add(obj["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _read_all_scores(jsonl_path: Path) -> list[dict]:
    """Read all score entries from the JSONL file."""
    scores: list[dict] = []
    if not jsonl_path.exists():
        return scores
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                scores.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return scores


# ---------------------------------------------------------------------------
# Incremental analysis (low memory)
# ---------------------------------------------------------------------------


def _run_incremental(
    items: list[dict],
    jsonl_path: Path,
    total: int,
    already_done: int,
) -> None:
    """Run TRIBE on items one at a time, appending each result to JSONL.

    After each inference:
    - Result is flushed to disk immediately
    - References to the result are dropped
    - gc.collect() frees memory from the subprocess
    """
    from tribe.backends.router import get_backend

    backend = get_backend()

    with open(jsonl_path, "a") as f:
        for i, item in enumerate(items):
            idx = already_done + i + 1
            start = time.monotonic()

            try:
                result = backend.analyze_text(item["text"])
                elapsed = time.monotonic() - start

                score_entry = {
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

                logger.info(
                    "[%d/%d] %s - score: %.1f, time: %.1fs",
                    idx,
                    total,
                    item["id"],
                    result.manipulation_score,
                    elapsed,
                )
            except Exception as e:
                logger.error("[%d/%d] %s - FAILED: %s", idx, total, item["id"], e)
                score_entry = {
                    "id": item["id"],
                    "text_preview": item["text"][:100],
                    "manipulation_score": None,
                    "error": str(e),
                    "label": _item_label(item),
                }

            # Write immediately and flush
            f.write(json.dumps(score_entry) + "\n")
            f.flush()

            # Free memory from inference
            del score_entry
            gc.collect()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(dataset_name: str, data_dir: Path) -> list[dict]:
    """Load a dataset by name."""
    if dataset_name == "semeval":
        from tribe.benchmarks.datasets.semeval import download, load

        download(data_dir)
        return load(data_dir)
    elif dataset_name == "mentalmanip":
        from tribe.benchmarks.datasets.mentalmanip import download, load

        download(data_dir)
        return load(data_dir)
    elif dataset_name == "qbias":
        from tribe.benchmarks.datasets.qbias import download, load

        download(data_dir)
        # Sample 3000 items (1500 biased + 1500 center) for reasonable GPU time
        return load(data_dir, sample_size=3000, balanced=True)
    elif dataset_name == "paired":
        from tribe.benchmarks.datasets.paired import load

        return load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _item_label(item: dict) -> int:
    """Extract a binary label from a dataset item."""
    if "manipulative" in item:
        return 1 if item["manipulative"] else 0
    if "propaganda_density" in item:
        return 1 if item["propaganda_density"] > 0.1 else 0
    return 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    dataset_name: str,
    items: list[dict],
    scores: list[dict],
) -> dict:
    """Compute dataset-specific metrics from analysis scores."""
    valid_scores = [s for s in scores if s.get("manipulation_score") is not None]

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
    """Compute paired comparison metrics."""
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
    """Compute binary separation metrics."""
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
    """Compute SemEval correlation and quartile separation metrics."""
    from tribe.benchmarks.metrics import compute_correlation, compute_separation

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
