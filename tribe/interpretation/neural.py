"""Neural interpretation - map TRIBE v2 cortical activations to persuasion signals.

Two-level analysis:
1. Yeo 7-network scores (for visualization - which networks are active)
2. Region-level persuasion analysis (for scoring - based on neuroscience of persuasion)

The persuasion signal is based on Falk et al. (2010, J Neurosci) and Ntoumanis et al. (2024, PNAS):
- vmPFC activation = value integration (person adopts message)
- dlPFC activation = critical evaluation (person counterargues)
- TPJ activation = motive analysis (person questions persuader intent)
- High vmPFC + low dlPFC + low TPJ = persuasion/manipulation signal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tribe.schema import NeuralAnalysis

# ---------------------------------------------------------------------------
# Yeo 7-network parcellation (kept for visualization)
# ---------------------------------------------------------------------------

YEO7_NETWORKS: dict[int, str] = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal_Attention",
    4: "Salience",
    5: "Limbic",
    6: "Executive_Control",
    7: "Default_Mode",
}

# ---------------------------------------------------------------------------
# Destrieux atlas regions mapped to persuasion-relevant circuits
# Based on Falk et al. 2010 (J Neurosci), Ntoumanis et al. 2024 (PNAS)
# ---------------------------------------------------------------------------

# vmPFC - ventromedial prefrontal cortex
# Role: value integration, self-relevance judgment
# High activation = person is adopting/internalizing the message
VMPC_REGIONS = [
    "G_rectus",
    "G_orbital",
    "S_orbital_med-olfact",
    "G_and_S_cingul-Ant",
    "S_suborbital",
]

# dlPFC - dorsolateral prefrontal cortex
# Role: critical evaluation, counterarguing, cognitive control
# Low activation = reduced counterarguing (vulnerability to persuasion)
DLPFC_REGIONS = [
    "G_front_middle",
    "G_front_sup",
    "S_front_middle",
    "S_front_sup",
]

# TPJ - temporoparietal junction
# Role: mentalizing, perspective-taking, questioning persuader intent
# Low activation = not questioning the source's motives
TPJ_REGIONS = [
    "G_pariet_inf-Supramar",
    "G_pariet_inf-Angular",
    "G_temp_sup-Lateral",
]

# Precuneus - self-referential processing
# Role: relating content to personal experience
# High activation = content feels personally relevant
PRECUNEUS_REGIONS = [
    "G_precuneus",
]

# Temporal poles - social/emotional semantic processing
# Role: social emotion, tribal/group identity processing
TEMPORAL_POLE_REGIONS = [
    "Pole_temporal",
]

# Insula - interoception and emotional salience
# Role: gut feeling, emotional awareness
INSULA_REGIONS = [
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
]

# Atlas directory (bundled with package)
ATLAS_DIR = Path(__file__).parent / "atlas"


# ---------------------------------------------------------------------------
# Atlas loading
# ---------------------------------------------------------------------------


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
        raise FileNotFoundError(f"Yeo 7-network atlas files not found in {atlas_dir}.")

    lh_labels, _, lh_names = nib.freesurfer.read_annot(str(lh_path))
    rh_labels, _, rh_names = nib.freesurfer.read_annot(str(rh_path))

    lh_network_ids = _annot_labels_to_network_ids(lh_labels, lh_names)
    rh_network_ids = _annot_labels_to_network_ids(rh_labels, rh_names)

    return np.concatenate([lh_network_ids, rh_network_ids])


def load_destrieux_region_ids(atlas_dir: Path | None = None) -> tuple[np.ndarray, list[str]]:
    """Load Destrieux cortical parcellation for fsaverage5.

    Returns:
        Tuple of (region_ids array shape (20484,), region_names list).
    """
    if atlas_dir is None:
        atlas_dir = ATLAS_DIR

    import nibabel as nib

    lh_path = atlas_dir / "lh.aparc.a2009s.annot"
    rh_path = atlas_dir / "rh.aparc.a2009s.annot"

    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(f"Destrieux atlas files not found in {atlas_dir}.")

    lh_labels, _, lh_names = nib.freesurfer.read_annot(str(lh_path))
    rh_labels, _, rh_names = nib.freesurfer.read_annot(str(rh_path))

    names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in lh_names]
    return np.concatenate([lh_labels, rh_labels]), names


# ---------------------------------------------------------------------------
# Network-level analysis (for visualization)
# ---------------------------------------------------------------------------


def _annot_labels_to_network_ids(labels: np.ndarray, names: list[bytes]) -> np.ndarray:
    """Convert annotation label indices to Yeo network IDs (1-7)."""
    name_to_netid: dict[int, int] = {}
    for idx, name in enumerate(names):
        n = name.decode("utf-8") if isinstance(name, bytes) else name
        if "7Networks_" in n:
            net_num = int(n.split("7Networks_")[1])
            name_to_netid[idx] = net_num
        else:
            name_to_netid[idx] = 0

    return np.array([name_to_netid.get(int(label), 0) for label in labels])


def compute_network_scores(
    activation: np.ndarray,
    network_ids: np.ndarray,
) -> dict[str, float]:
    """Compute mean activation per Yeo 7 functional network."""
    scores = {}
    for net_id, net_name in YEO7_NETWORKS.items():
        mask = network_ids == net_id
        if mask.sum() > 0:
            scores[net_name] = float(np.mean(activation[mask]))
        else:
            scores[net_name] = 0.0
    return scores


# ---------------------------------------------------------------------------
# Region-level persuasion analysis (for scoring)
# ---------------------------------------------------------------------------


def compute_region_group_score(
    activation: np.ndarray,
    region_ids: np.ndarray,
    region_names: list[str],
    target_regions: list[str],
) -> float:
    """Compute mean activation for a group of named brain regions.

    Args:
        activation: Shape (20484,) activation vector.
        region_ids: Shape (20484,) region label indices.
        region_names: List of region names (index = label ID).
        target_regions: Region names to include.

    Returns:
        Mean activation across all vertices in the target regions.
    """
    target_indices = [i for i, name in enumerate(region_names) if name in target_regions]
    if not target_indices:
        return 0.0

    mask = np.isin(region_ids, target_indices)
    if mask.sum() == 0:
        return 0.0

    return float(np.mean(activation[mask]))


def compute_persuasion_scores(
    activation: np.ndarray,
    region_ids: np.ndarray,
    region_names: list[str],
) -> dict[str, float]:
    """Compute activation scores for persuasion-relevant brain regions.

    Returns dict with keys: vmPFC, dlPFC, TPJ, precuneus, temporal_pole, insula.
    """
    groups = {
        "vmPFC": VMPC_REGIONS,
        "dlPFC": DLPFC_REGIONS,
        "TPJ": TPJ_REGIONS,
        "precuneus": PRECUNEUS_REGIONS,
        "temporal_pole": TEMPORAL_POLE_REGIONS,
        "insula": INSULA_REGIONS,
    }
    return {
        name: compute_region_group_score(activation, region_ids, region_names, regions)
        for name, regions in groups.items()
    }


def compute_persuasion_signal(persuasion_scores: dict[str, float]) -> float:
    """Compute the persuasion signal from region-level scores.

    Based on Falk et al. (2010) and Ntoumanis et al. (2024):
    - High vmPFC = value adoption (person internalizes message)
    - Low dlPFC = suppressed critical evaluation
    - Low TPJ = not questioning persuader's motives
    - High insula = emotional arousal
    - High precuneus = self-relevance engaged

    The signal captures: content that triggers value integration
    while suppressing analytical counterarguing.

    Returns:
        Persuasion signal from 0.0 (no persuasion pattern) to 1.0 (strong pattern).
    """
    vmPFC = persuasion_scores.get("vmPFC", 0.0)
    dlPFC = persuasion_scores.get("dlPFC", 0.0)
    tpj = persuasion_scores.get("TPJ", 0.0)
    insula = persuasion_scores.get("insula", 0.0)
    precuneus = persuasion_scores.get("precuneus", 0.0)

    # Normalize all scores to a common scale using the absolute values
    all_scores = [abs(vmPFC), abs(dlPFC), abs(tpj), abs(insula), abs(precuneus)]
    max_score = max(all_scores) if max(all_scores) > 0.001 else 1.0

    vmPFC_n = abs(vmPFC) / max_score
    dlPFC_n = abs(dlPFC) / max_score
    tpj_n = abs(tpj) / max_score
    insula_n = abs(insula) / max_score
    precuneus_n = abs(precuneus) / max_score

    # Persuasion signal components:
    # 1. Value adoption: high vmPFC (weight: 0.30)
    value_adoption = vmPFC_n * 0.30

    # 2. Suppressed critical evaluation: inverse of dlPFC (weight: 0.25)
    critical_suppression = (1.0 - dlPFC_n) * 0.25

    # 3. Suppressed motive analysis: inverse of TPJ (weight: 0.20)
    motive_suppression = (1.0 - tpj_n) * 0.20

    # 4. Emotional arousal: high insula (weight: 0.15)
    emotional_arousal = insula_n * 0.15

    # 5. Self-relevance: high precuneus (weight: 0.10)
    self_relevance = precuneus_n * 0.10

    signal = (
        value_adoption
        + critical_suppression
        + motive_suppression
        + emotional_arousal
        + self_relevance
    )

    return round(min(max(signal, 0.0), 1.0), 4)


def persuasion_signal_to_score(signal: float) -> float:
    """Convert persuasion signal (0-1) to manipulation score (0-10).

    The mapping is calibrated so that:
    - 0.0-0.3 signal = 0-2 score (low - balanced processing)
    - 0.3-0.5 signal = 2-4 score (mild persuasion pattern)
    - 0.5-0.7 signal = 4-7 score (moderate persuasion pattern)
    - 0.7-1.0 signal = 7-10 score (strong persuasion pattern)
    """
    if signal <= 0.3:
        return round(signal / 0.3 * 2.0, 1)
    if signal <= 0.5:
        return round(2.0 + (signal - 0.3) / 0.2 * 2.0, 1)
    if signal <= 0.7:
        return round(4.0 + (signal - 0.5) / 0.2 * 3.0, 1)
    return round(min(7.0 + (signal - 0.7) / 0.3 * 3.0, 10.0), 1)


# ---------------------------------------------------------------------------
# Backward compatibility - old ratio (kept for Yeo network visualization)
# ---------------------------------------------------------------------------

EMOTIONAL_NETWORKS = {"Salience", "Default_Mode", "Limbic"}
RATIONAL_NETWORKS = {"Executive_Control", "Dorsal_Attention"}


def compute_manipulation_ratio(network_scores: dict[str, float]) -> float:
    """Compute emotional/rational network ratio (for visualization only).

    NOTE: This ratio is NOT used for scoring. It is kept for the Yeo 7-network
    bar chart visualization. The actual manipulation score comes from
    compute_persuasion_signal() which uses region-level analysis.
    """
    emotional_sum = sum(abs(network_scores.get(net, 0.0)) for net in EMOTIONAL_NETWORKS)
    rational_sum = sum(abs(network_scores.get(net, 0.0)) for net in RATIONAL_NETWORKS)

    if rational_sum < 0.001:
        return 10.0

    return round(emotional_sum / rational_sum, 2)


# ---------------------------------------------------------------------------
# Main interpretation pipeline
# ---------------------------------------------------------------------------


def interpret_activation(
    activation: np.ndarray,
    network_ids: np.ndarray | None = None,
) -> NeuralAnalysis:
    """Full neural interpretation pipeline.

    Uses two levels of analysis:
    1. Yeo 7-network scores (for visualization)
    2. Destrieux region-level persuasion analysis (for scoring)
    """
    if activation.ndim == 2:
        activation = np.mean(activation, axis=0)

    assert activation.shape == (20484,), (
        f"Expected 20484-element vector, got shape {activation.shape}"
    )

    if network_ids is None:
        network_ids = load_yeo7_network_ids()

    # Level 1: Yeo 7-network scores (visualization)
    network_scores = compute_network_scores(activation, network_ids)
    manipulation_ratio = compute_manipulation_ratio(network_scores)

    # Level 2: Region-level persuasion analysis (scoring)
    region_ids, region_names = load_destrieux_region_ids()
    persuasion_scores = compute_persuasion_scores(activation, region_ids, region_names)
    persuasion_signal = compute_persuasion_signal(persuasion_scores)

    # Dominant network (for display - from Yeo)
    dominant_network = max(network_scores, key=network_scores.get)  # type: ignore[arg-type]

    # Dominant regions (from persuasion analysis)
    dominant_regions = _identify_persuasion_regions(persuasion_scores)

    # Interpretation text
    interpretation = _generate_interpretation(persuasion_scores, persuasion_signal)

    return NeuralAnalysis(
        network_scores=network_scores,
        manipulation_ratio=manipulation_ratio,
        dominant_network=dominant_network,
        dominant_regions=dominant_regions,
        interpretation=interpretation,
        persuasion_scores=persuasion_scores,
        persuasion_signal=persuasion_signal,
    )


def _identify_persuasion_regions(persuasion_scores: dict[str, float]) -> list[str]:
    """Identify the most active persuasion-relevant regions."""
    if not persuasion_scores:
        return []
    sorted_regions = sorted(persuasion_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return [name for name, _ in sorted_regions[:3]]


def _generate_interpretation(
    persuasion_scores: dict[str, float],
    persuasion_signal: float,
) -> str:
    """Generate a human-readable interpretation of the persuasion analysis."""
    vmPFC = persuasion_scores.get("vmPFC", 0.0)
    dlPFC = persuasion_scores.get("dlPFC", 0.0)
    tpj = persuasion_scores.get("TPJ", 0.0)

    parts = []

    if persuasion_signal >= 0.6:
        parts.append("Strong persuasion pattern detected.")
    elif persuasion_signal >= 0.4:
        parts.append("Moderate persuasion pattern detected.")
    else:
        parts.append("Weak or no persuasion pattern.")

    # Describe the specific pattern
    all_vals = [abs(v) for v in persuasion_scores.values()]
    max_val = max(all_vals) if all_vals else 1.0
    if max_val < 0.001:
        return "Minimal neural activation predicted."

    vmPFC_rel = abs(vmPFC) / max_val
    dlPFC_rel = abs(dlPFC) / max_val
    tpj_rel = abs(tpj) / max_val

    if vmPFC_rel > 0.6:
        parts.append("High value integration (vmPFC) - message is being internalized.")
    if dlPFC_rel < 0.4:
        parts.append("Low critical evaluation (dlPFC) - counterarguing suppressed.")
    if tpj_rel < 0.4:
        parts.append("Low motive analysis (TPJ) - source intent not questioned.")

    return " ".join(parts)
