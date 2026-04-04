"""JSON output renderer."""

from tribe.schema import ContentAnalysis


def render_json(analysis: ContentAnalysis, indent: int = 2) -> str:
    """Render ContentAnalysis as formatted JSON."""
    return analysis.to_json(indent=indent)
