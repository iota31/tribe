"""Tests for neural interpretation layer."""

from pathlib import Path

import numpy as np

from tribe.interpretation.neural import (
    YEO7_NETWORKS,
    compute_manipulation_ratio,
    compute_network_scores,
    load_yeo7_network_ids,
)


def test_manipulation_ratio_calculation():
    """High emotional / low rational = high ratio."""
    network_scores = {
        "Salience": 0.8,
        "Default_Mode": 0.7,
        "Limbic": 0.6,
        "Executive_Control": 0.2,
        "Dorsal_Attention": 0.3,
        "Visual": 0.5,
        "Somatomotor": 0.4,
    }
    ratio = compute_manipulation_ratio(network_scores)
    # (0.8 + 0.7 + 0.6) / (0.2 + 0.3) = 2.1 / 0.5 = 4.2
    assert 4.0 <= ratio <= 4.5


def test_manipulation_ratio_balanced():
    """Balanced networks give ratio of 1.5.

    emotional_sum = Salience(0.5) + Limbic(0.5) + Default_Mode(0.5) = 1.5
    rational_sum = Executive_Control(0.5) + Dorsal_Attention(0.5) = 1.0
    ratio = 1.5 / 1.0 = 1.5
    """
    network_scores = {net: 0.5 for net in YEO7_NETWORKS.values()}
    ratio = compute_manipulation_ratio(network_scores)
    assert 1.3 <= ratio <= 1.7


def test_network_scores():
    """Network scores computed from activation vector."""
    lh = [4] * 1024 + [6] * 1024 + [7] * 1024 + [1] * 1024 + [0] * 6146
    rh = [4] * 1024 + [6] * 1024 + [7] * 1024 + [1] * 1024 + [0] * 6146
    network_ids = np.array(lh + rh, dtype=np.int32)

    activation = np.zeros(20484)
    activation[:1024] = 0.8  # Salience (LH)
    activation[1024:2048] = 0.2  # ExecControl (LH)
    activation[2048:3072] = 0.6  # DefaultMode (LH)
    activation[10242:11266] = 0.8  # Salience (RH)
    activation[11266:12290] = 0.2  # ExecControl (RH)
    activation[12290:13314] = 0.6  # DefaultMode (RH)

    scores = compute_network_scores(activation, network_ids)
    assert 0.78 <= scores["Salience"] <= 0.82
    assert 0.18 <= scores["Executive_Control"] <= 0.22
    assert 0.58 <= scores["Default_Mode"] <= 0.62


def test_load_yeo7_network_ids():
    """load_yeo7_network_ids returns correct shape and valid network IDs."""
    atlas_dir = Path(__file__).parent.parent / "tribe" / "interpretation" / "atlas"
    ids = load_yeo7_network_ids(atlas_dir)

    assert ids.shape == (20484,), f"Expected (20484,), got {ids.shape}"

    unique = sorted(set(ids))
    assert unique == [0, 1, 2, 3, 4, 5, 6, 7], f"Unexpected values: {unique}"

    lh = ids[:10242]
    rh = ids[10242:]
    assert lh.shape == (10242,)
    assert rh.shape == (10242,)

    assert len(sorted(set(lh))) == 8
    assert len(sorted(set(rh))) == 8


def test_manipulation_ratio_zero_rational():
    """Zero rational activation gives maximum ratio."""
    network_scores = {
        "Salience": 0.8,
        "Default_Mode": 0.7,
        "Limbic": 0.6,
        "Executive_Control": 0.0,
        "Dorsal_Attention": 0.0,
        "Visual": 0.5,
        "Somatomotor": 0.4,
    }
    ratio = compute_manipulation_ratio(network_scores)
    assert ratio == 10.0


def test_yeo7_networks_has_seven_networks():
    """YEO7_NETWORKS maps IDs 1-7 to network names."""
    assert len(YEO7_NETWORKS) == 7
    for i in range(1, 8):
        assert i in YEO7_NETWORKS
