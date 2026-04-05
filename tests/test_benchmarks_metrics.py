"""Tests for the benchmarks metrics module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# compute_separation
# ---------------------------------------------------------------------------


class TestComputeSeparation:
    """Tests for binary separation metrics."""

    def test_perfect_separation(self):
        """Perfect separation yields AUC = 1.0 and large Cohen's d."""
        from tribe.benchmarks.metrics import compute_separation

        pos = [8.0, 9.0, 10.0, 7.0, 8.5]
        neg = [1.0, 2.0, 1.5, 0.5, 2.5]
        result = compute_separation(pos, neg)

        assert result.auc_roc == 1.0
        assert result.cohens_d > 3.0
        assert result.n_positive == 5
        assert result.n_negative == 5

    def test_no_separation(self):
        """Identical distributions yield AUC near 0.5 and d near 0."""
        from tribe.benchmarks.metrics import compute_separation

        pos = [5.0, 5.0, 5.0, 5.0]
        neg = [5.0, 5.0, 5.0, 5.0]
        result = compute_separation(pos, neg)

        assert result.auc_roc == 0.5
        assert result.cohens_d == 0.0

    def test_result_fields(self):
        """All SeparationResult fields are populated correctly."""
        from tribe.benchmarks.metrics import compute_separation

        pos = [6.0, 7.0, 8.0]
        neg = [2.0, 3.0, 4.0]
        result = compute_separation(pos, neg)

        assert 0.0 <= result.auc_roc <= 1.0
        assert result.mean_positive > result.mean_negative
        assert result.std_positive >= 0
        assert result.std_negative >= 0
        assert result.n_positive == 3
        assert result.n_negative == 3

    def test_moderate_separation(self):
        """Overlapping distributions yield AUC between 0.5 and 1.0."""
        from tribe.benchmarks.metrics import compute_separation

        pos = [4.0, 5.0, 6.0, 7.0, 5.5]
        neg = [3.0, 4.0, 5.0, 2.0, 3.5]
        result = compute_separation(pos, neg)

        assert 0.5 < result.auc_roc < 1.0
        assert result.cohens_d > 0


# ---------------------------------------------------------------------------
# compute_correlation
# ---------------------------------------------------------------------------


class TestComputeCorrelation:
    """Tests for correlation metrics."""

    def test_perfect_positive_correlation(self):
        """Perfectly correlated data yields r = 1.0."""
        from tribe.benchmarks.metrics import compute_correlation

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = compute_correlation(x, y)

        assert result.pearson_r == 1.0
        assert result.spearman_rho == 1.0
        assert result.n == 5

    def test_negative_correlation(self):
        """Inversely correlated data yields r = -1.0."""
        from tribe.benchmarks.metrics import compute_correlation

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = compute_correlation(x, y)

        assert result.pearson_r == -1.0
        assert result.spearman_rho == -1.0

    def test_p_values_present(self):
        """p-values are present and non-negative."""
        from tribe.benchmarks.metrics import compute_correlation

        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [1.1, 2.2, 2.9, 4.1, 5.0, 6.2]
        result = compute_correlation(x, y)

        assert result.spearman_p >= 0
        assert result.pearson_p >= 0


# ---------------------------------------------------------------------------
# compute_paired
# ---------------------------------------------------------------------------


class TestComputePaired:
    """Tests for paired comparison metrics."""

    def test_clear_winner(self):
        """When manipulative always scores higher, win_rate = 1.0."""
        from tribe.benchmarks.metrics import compute_paired

        manip = [8.0, 9.0, 7.0, 8.5, 9.5]
        neutral = [2.0, 3.0, 1.0, 2.5, 3.5]
        result = compute_paired(manip, neutral)

        assert result.win_rate == 1.0
        assert result.mean_diff > 0
        assert result.p_value < 0.05
        assert result.n_pairs == 5

    def test_no_difference(self):
        """Identical pairs yield win_rate = 0.0 and high p-value."""
        from tribe.benchmarks.metrics import compute_paired

        same = [5.0, 5.0, 5.0, 5.0]
        result = compute_paired(same, same)

        assert result.win_rate == 0.0
        assert result.mean_diff == 0.0

    def test_mixed_results(self):
        """Mixed results yield 0 < win_rate < 1."""
        from tribe.benchmarks.metrics import compute_paired

        manip = [6.0, 3.0, 7.0, 2.0]
        neutral = [4.0, 5.0, 3.0, 6.0]
        result = compute_paired(manip, neutral)

        assert 0.0 < result.win_rate < 1.0
        assert result.n_pairs == 4

    def test_t_statistic_sign(self):
        """t-statistic is positive when manipulative scores higher on average."""
        from tribe.benchmarks.metrics import compute_paired

        manip = [7.0, 8.0, 9.0]
        neutral = [3.0, 4.0, 5.0]
        result = compute_paired(manip, neutral)

        assert result.t_statistic > 0
