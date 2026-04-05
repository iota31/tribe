"""Output renderers — human narrative and JSON."""

from tribe.output.json_output import render_json
from tribe.output.narrative import render_narrative

__all__ = ["render_narrative", "render_json"]
