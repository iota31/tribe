"""Benchmark metrics for evaluating manipulation detection quality.

Provides statistical measures for binary separation (AUC-ROC, Cohen's d),
correlation (Spearman, Pearson), and paired comparisons (win rate, t-test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SeparationResult:
    """Results from a binary separation test."""

    auc_roc: float
    cohens_d: float
    mean_positive: float
    std_positive: float
    mean_negative: float
    std_negative: float
    n_positive: int
    n_negative: int


@dataclass
class CorrelationResult:
    """Results from a correlation test."""

    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n: int


@dataclass
class PairedResult:
    """Results from a paired comparison test."""

    win_rate: float
    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value: float
    n_pairs: int


def compute_separation(
    scores_positive: list[float],
    scores_negative: list[float],
) -> SeparationResult:
    """Compute AUC-ROC and Cohen's d for binary separation.

    Args:
        scores_positive: Manipulation scores for positive (manipulative) samples.
        scores_negative: Manipulation scores for negative (non-manipulative) samples.

    Returns:
        SeparationResult with AUC-ROC, Cohen's d, and distribution stats.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = [1] * len(scores_positive) + [0] * len(scores_negative)
    scores = scores_positive + scores_negative

    auc = roc_auc_score(labels, scores)

    # Cohen's d with pooled standard deviation
    mean_p = float(np.mean(scores_positive))
    mean_n = float(np.mean(scores_negative))
    std_p = float(np.std(scores_positive, ddof=1))
    std_n = float(np.std(scores_negative, ddof=1))

    n_p = len(scores_positive)
    n_n = len(scores_negative)
    pooled_std = math.sqrt(((n_p - 1) * std_p**2 + (n_n - 1) * std_n**2) / (n_p + n_n - 2))
    d = (mean_p - mean_n) / pooled_std if pooled_std > 0 else 0.0

    return SeparationResult(
        auc_roc=round(auc, 4),
        cohens_d=round(d, 4),
        mean_positive=round(mean_p, 4),
        std_positive=round(std_p, 4),
        mean_negative=round(mean_n, 4),
        std_negative=round(std_n, 4),
        n_positive=n_p,
        n_negative=n_n,
    )


def compute_correlation(
    x: list[float],
    y: list[float],
) -> CorrelationResult:
    """Compute Spearman and Pearson correlation.

    Args:
        x: First variable values.
        y: Second variable values.

    Returns:
        CorrelationResult with both correlation coefficients and p-values.
    """
    from scipy import stats

    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)

    return CorrelationResult(
        spearman_rho=round(float(spearman.statistic), 4),
        spearman_p=round(float(spearman.pvalue), 6),
        pearson_r=round(float(pearson.statistic), 4),
        pearson_p=round(float(pearson.pvalue), 6),
        n=len(x),
    )


def compute_paired(
    scores_manipulative: list[float],
    scores_neutral: list[float],
) -> PairedResult:
    """Compute paired comparison metrics.

    Args:
        scores_manipulative: Manipulation scores for manipulative samples.
        scores_neutral: Manipulation scores for paired neutral samples.

    Returns:
        PairedResult with win rate, mean difference, t-statistic, and p-value.
    """
    import numpy as np
    from scipy import stats

    diffs = [m - n for m, n in zip(scores_manipulative, scores_neutral)]
    wins = sum(1 for d in diffs if d > 0)

    t_stat, p_val = stats.ttest_rel(scores_manipulative, scores_neutral)

    return PairedResult(
        win_rate=round(wins / len(diffs), 4),
        mean_diff=round(float(np.mean(diffs)), 4),
        std_diff=round(float(np.std(diffs, ddof=1)), 4),
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_val), 6),
        n_pairs=len(diffs),
    )
