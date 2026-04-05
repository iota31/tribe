"""Neural interpretation — map TRIBE v2 cortical activations to emotions.

Uses the Yeo 2011 7-network parcellation to group predicted brain activations
into functional networks, then computes manipulation ratio from network imbalance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tribe.schema import NeuralAnalysis

# Yeo 7-network IDs and their names
YEO7_NETWORKS: dict[int, str] = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal_Attention",
    4: "Salience",  # Ventral Attention / Salience Network
    5: "Limbic",
    6: "Executive_Control",  # Frontoparietal / Executive Control
    7: "Default_Mode",
}

# Networks that indicate emotional/attention-capture processing
EMOTIONAL_NETWORKS = {"Salience", "Default_Mode", "Limbic"}

# Networks that indicate rational/analytical processing
RATIONAL_NETWORKS = {"Executive_Control", "Dorsal_Attention"}

# Desikan-Killiany regions within networks that map to specific emotions
REGION_EMOTION_MAP: dict[str, str] = {
    "insula": "disgust/aversion",
    "rostralanteriorcingulate": "anxiety/moral outrage",
    "caudalanteriorcingulate": "conflict/distress",
    "medialorbitofrontal": "desire/value judgment",
    "temporalpole": "social emotion/tribal outrage",
    "lateralorbitofrontal": "impulsive decision pressure",
}

# Atlas directory (bundled with package)
ATLAS_DIR = Path(__file__).parent / "atlas"


def load_yeo7_network_ids(atlas_dir: Path | None = None) -> np.ndarray:
    """Load Yeo 7-network parcellation for fsaverage5.

    Returns:
        Array of shape (20484,) with network IDs (0-7, 0=unlabeled).
    """
    if atlas_dir is None:
        atlas_dir = ATLAS_DIR

    import nibabel as nib

    lh_path = atlas_dir / "lh.Yeo2011_7Networks_N1000.annot"
    rh_path = atlas_dir / "rh.Yeo2011_7Networks_N1000.annot"

    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(
            f"Yeo 7-network atlas files not found in {atlas_dir}. "
            "Run `tribe setup` to download atlas files."
        )

    lh_labels, _, lh_names = nib.freesurfer.read_annot(str(lh_path))
    rh_labels, _, rh_names = nib.freesurfer.read_annot(str(rh_path))

    # Convert annotation labels to network IDs (1-7)
    lh_network_ids = _annot_labels_to_network_ids(lh_labels, lh_names)
    rh_network_ids = _annot_labels_to_network_ids(rh_labels, rh_names)

    return np.concatenate([lh_network_ids, rh_network_ids])


def _annot_labels_to_network_ids(
    labels: np.ndarray, names: list[bytes]
) -> np.ndarray:
    """Convert annotation label indices to Yeo network IDs (1-7)."""
    name_to_netid: dict[int, int] = {}
    for idx, name in enumerate(names):
        n = name.decode("utf-8") if isinstance(name, bytes) else name
        if "7Networks_" in n:
            net_num = int(n.split("7Networks_")[1])
            name_to_netid[idx] = net_num
        else:
            name_to_netid[idx] = 0  # unlabeled

    return np.array([name_to_netid.get(int(l), 0) for l in labels])


def compute_network_scores(
    activation: np.ndarray,
    network_ids: np.ndarray,
) -> dict[str, float]:
    """Compute mean activation per Yeo 7 functional network.

    Args:
        activation: Shape (20484,) activation vector from TRIBE v2.
        network_ids: Shape (20484,) network IDs from Yeo parcellation.

    Returns:
        Dict mapping network name to mean activation score.
    """
    scores = {}
    for net_id, net_name in YEO7_NETWORKS.items():
        mask = network_ids == net_id
        if mask.sum() > 0:
            scores[net_name] = float(np.mean(activation[mask]))
        else:
            scores[net_name] = 0.0

    return scores


def compute_manipulation_ratio(network_scores: dict[str, float]) -> float:
    """Compute the manipulation ratio from network activation scores.

    manipulation_ratio = emotional_networks / rational_networks

    High ratio = content activates attention-capture and emotional networks
    while suppressing rational evaluation.
    """
    emotional_sum = sum(
        abs(network_scores.get(net, 0.0)) for net in EMOTIONAL_NETWORKS
    )
    rational_sum = sum(
        abs(network_scores.get(net, 0.0)) for net in RATIONAL_NETWORKS
    )

    if rational_sum < 0.001:
        # Avoid division by zero; maximum manipulation signal
        return 10.0

    return round(emotional_sum / rational_sum, 2)


def interpret_activation(
    activation: np.ndarray,
    network_ids: np.ndarray | None = None,
) -> NeuralAnalysis:
    """Full neural interpretation pipeline.

    Args:
        activation: Shape (20484,) or (N, 20484) activation from TRIBE v2.
                   If 2D, averages across time dimension.
        network_ids: Pre-loaded Yeo network IDs. Loaded if None.

    Returns:
        NeuralAnalysis with network scores, manipulation ratio, and interpretation.
    """
    # Handle multi-timepoint activations by averaging
    if activation.ndim == 2:
        activation = np.mean(activation, axis=0)

    assert activation.shape == (20484,), (
        f"Expected 20484-element vector, got shape {activation.shape}"
    )

    if network_ids is None:
        network_ids = load_yeo7_network_ids()

    network_scores = compute_network_scores(activation, network_ids)
    manipulation_ratio = compute_manipulation_ratio(network_scores)

    # Find dominant network for manipulation detection:
    # Emotional networks take priority (they drive the manipulation trigger).
    # Among emotional networks, pick the highest activation.
    # If no emotional network is active, fall back to cognitive networks.
    emotional_networks = {
        k: v
        for k, v in network_scores.items()
        if k in EMOTIONAL_NETWORKS and v > 0
    }
    cognitive_networks = {
        k: v
        for k, v in network_scores.items()
        if k not in EMOTIONAL_NETWORKS
        and k not in ("Visual", "Somatomotor")
        and v > 0
    }
    if emotional_networks:
        dominant_network = max(emotional_networks, key=emotional_networks.get)  # type: ignore[arg-type]
    elif cognitive_networks:
        dominant_network = max(cognitive_networks, key=cognitive_networks.get)  # type: ignore[arg-type]
    else:
        dominant_network = max(network_scores, key=network_scores.get)  # type: ignore[arg-type]

    # Identify dominant regions (would need DK atlas for full detail)
    dominant_regions = _infer_dominant_regions(dominant_network)

    # Generate human-readable interpretation
    interpretation = _generate_interpretation(
        network_scores, manipulation_ratio, dominant_network
    )

    return NeuralAnalysis(
        network_scores=network_scores,
        manipulation_ratio=manipulation_ratio,
        dominant_network=dominant_network,
        dominant_regions=dominant_regions,
        interpretation=interpretation,
    )


def _infer_dominant_regions(dominant_network: str) -> list[str]:
    """Infer likely dominant brain regions from the dominant network."""
    network_regions: dict[str, list[str]] = {
        "Salience": ["insula", "rostralanteriorcingulate"],
        "Default_Mode": ["medialorbitofrontal", "posteriorcingulate", "inferiorparietal"],
        "Limbic": ["temporalpole", "lateralorbitofrontal", "medialorbitofrontal"],
        "Executive_Control": ["rostralmiddlefrontal", "superiorparietal"],
        "Dorsal_Attention": ["superiorparietal", "precentral"],
    }
    return network_regions.get(dominant_network, [])


def _generate_interpretation(
    network_scores: dict[str, float],
    manipulation_ratio: float,
    dominant_network: str,
) -> str:
    """Generate a human-readable interpretation of neural activation patterns."""
    if manipulation_ratio >= 3.0:
        intensity = "strongly"
    elif manipulation_ratio >= 2.0:
        intensity = "moderately"
    elif manipulation_ratio >= 1.5:
        intensity = "somewhat"
    else:
        return (
            "Content engages rational processing alongside emotional responses. "
            "Neural activation pattern is balanced."
        )

    network_descriptions = {
        "Salience": "threat detection and emotional salience",
        "Default_Mode": "self-referential and personal relevance processing",
        "Limbic": "emotional memory and social emotion processing",
        "Executive_Control": "analytical and rational processing",
        "Dorsal_Attention": "focused voluntary attention",
    }

    dominant_desc = network_descriptions.get(
        dominant_network, dominant_network.lower()
    )

    return (
        f"Content {intensity} activates {dominant_desc} "
        f"while suppressing analytical processing. "
        f"Emotional networks activate {manipulation_ratio}x "
        f"more than rational evaluation networks."
    )
