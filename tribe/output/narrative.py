"""Human-first narrative output renderer."""

from __future__ import annotations

from tribe.schema import ContentAnalysis


def render_narrative(analysis: ContentAnalysis, verbose: bool = False) -> str:
    """Render ContentAnalysis as a human-readable narrative.

    Args:
        analysis: The analysis result to render.
        verbose: If True, show per-technique details and neural data.

    Returns:
        Formatted string for terminal output.
    """
    lines: list[str] = []

    # Header with primary trigger
    score = analysis.manipulation_score
    trigger = analysis.primary_trigger

    if score >= 6.0:
        icon = "\u26a0\ufe0f "  # warning sign
        verb = "engineered"
    elif score >= 3.0:
        icon = "\u26a0 "
        verb = "designed"
    else:
        icon = "\u2139\ufe0f "  # info
        verb = "tends"

    if score >= 3.0:
        lines.append(f"{icon} This content is {verb} to trigger " f"a {trigger.upper()} response.")
    else:
        lines.append(
            f"{icon} This content {verb} toward a {trigger} tone. " f"Low manipulation signal."
        )

    lines.append("")

    # Technique summary
    if analysis.techniques:
        technique_strs = []
        for t in analysis.techniques[:5]:  # top 5
            level = _confidence_level(t.confidence)
            technique_strs.append(f"{t.name.lower()} ({level})")

        if technique_strs:
            # Short narrative about what the content does
            top_technique = analysis.techniques[0]
            lines.append(
                f"   It uses {top_technique.name.lower()}: "
                f"{top_technique.description.lower().rstrip('.')}."
            )
            lines.append("")

    # Score line
    lines.append(f"   Manipulation score: {score:.1f}/10")

    # Trigger details
    lines.append(f"   Primary emotion targeted: {trigger}")
    secondary = _get_secondary_triggers(analysis)
    if secondary:
        lines.append(f"   Secondary: {', '.join(secondary)}")

    lines.append("")

    # Techniques line
    if analysis.techniques:
        tech_parts = []
        for t in analysis.techniques[:4]:
            level = _confidence_level(t.confidence)
            tech_parts.append(f"{t.name.lower()} ({level})")
        lines.append(f"   Techniques: {', '.join(tech_parts)}")

    # Neural analysis summary
    if analysis.neural:
        ratio = analysis.neural.manipulation_ratio
        dominant = analysis.neural.dominant_network.replace("_", " ").lower()
        lines.append("")
        lines.append(
            f"   Neural analysis: {dominant} network activates "
            f"{ratio}x above executive control network "
            f"({analysis.backend} backend)"
        )

    # Verbose: detailed technique breakdown
    if verbose and analysis.techniques:
        lines.append("")
        lines.append(
            "   \u2500\u2500\u2500 Techniques Detected "
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        )
        for t in analysis.techniques:
            pct = int(t.confidence * 100)
            lines.append("")
            lines.append(f"   {t.name} ({pct}% confidence)")
            lines.append(f"     Target emotion: {t.emotion_target.title()}")
            lines.append(f"     {t.description}")

    # Verbose: neural network breakdown
    if verbose and analysis.neural:
        lines.append("")
        lines.append(
            "   \u2500\u2500\u2500 Neural Analysis (TRIBE v2) "
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            "\u2500\u2500"
        )
        lines.append("")
        lines.append("   Network Activation:")

        for net_name, score_val in sorted(
            analysis.neural.network_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if net_name in ("Visual", "Somatomotor"):
                continue
            bar = _make_bar(score_val, max_val=1.0)
            label = _network_label(net_name)
            lines.append(f"     {label:35s} {bar} {score_val:.2f}")

        lines.append("")
        ratio = analysis.neural.manipulation_ratio
        lines.append(f"   Manipulation ratio: {ratio}x")
        lines.append(f"   (emotional networks activate {ratio}x more than " f"rational networks)")

        if analysis.neural.dominant_regions:
            regions = ", ".join(analysis.neural.dominant_regions)
            lines.append(f"   Dominant regions: {regions}")

        lines.append(f"   Interpretation: {analysis.neural.interpretation}")

    # Footer metadata
    lines.append("")
    time_str = f"{analysis.processing_time_ms}ms"
    if analysis.processing_time_ms >= 1000:
        time_str = f"{analysis.processing_time_ms / 1000:.1f}s"
    lines.append(f"   Backend: {analysis.backend} | Time: {time_str}")

    return "\n".join(lines)


def render_quiet(analysis: ContentAnalysis) -> str:
    """Render a single-line score summary."""
    score = analysis.manipulation_score
    trigger = analysis.primary_trigger
    time_str = f"{analysis.processing_time_ms}ms"
    if analysis.processing_time_ms >= 1000:
        time_str = f"{analysis.processing_time_ms / 1000:.1f}s"
    return f"{score:.1f}/10 \u2014 {trigger} ({analysis.backend}, {time_str})"


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _get_secondary_triggers(analysis: ContentAnalysis) -> list[str]:
    """Get secondary emotion triggers beyond the primary."""
    triggers: set[str] = set()

    for t in analysis.techniques[1:4]:
        if t.emotion_target != analysis.primary_trigger.lower():
            triggers.add(t.emotion_target.title())

    for e in analysis.emotions[:3]:
        if (
            e.name.lower() != analysis.primary_trigger.lower()
            and e.name.lower() != "neutral"
            and e.confidence > 0.15
        ):
            triggers.add(e.name.title())

    return list(triggers)[:3]


def _make_bar(value: float, max_val: float = 1.0, width: int = 10) -> str:
    """Create a simple bar chart character."""
    if max_val <= 0:
        return "\u2591" * width
    filled = int((abs(value) / max_val) * width)
    filled = min(filled, width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _network_label(network_name: str) -> str:
    """Get a human-readable label for a brain network."""
    labels = {
        "Salience": "Salience (attention capture):",
        "Default_Mode": "Default Mode (self-referential):",
        "Limbic": "Limbic (emotional):",
        "Executive_Control": "Executive Control (rational):",
        "Dorsal_Attention": "Dorsal Attention (focused):",
        "Visual": "Visual (sensory):",
        "Somatomotor": "Somatomotor (sensory):",
    }
    return labels.get(network_name, f"{network_name}:")
