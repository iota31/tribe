"""SVG visualization generator for TRIBE benchmark results.

Generates publication-ready SVG charts using string formatting (no external
dependencies). Visual style matches images/generate_diagrams.py -- dark
background, monospace typography, accent colors.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared SVG helpers (matching images/generate_diagrams.py style)
# ---------------------------------------------------------------------------

SVG_NS = "http://www.w3.org/2000/svg"

# Color palette
BG_DARK = "#0d1117"
BG_CARD = "#131920"
BORDER = "#1e2832"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#7d8590"
TEXT_DIM = "#4b5563"
COLOR_MANIPULATIVE = "#ef4444"
COLOR_NEUTRAL = "#22c55e"
COLOR_ACCENT = "#58a6ff"
FONT = "ui-monospace,SFMono,Menlo,monospace"


def _el(
    tag: str,
    attrs: dict[str, str] | None = None,
    text: str | None = None,
    parent: ET.Element | None = None,
) -> ET.Element:
    """Create an SVG element."""
    e = ET.Element(tag, attrs or {})
    if text:
        e.text = text
    if parent is not None:
        parent.append(e)
    return e


def _rect(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str | None = None,
    rx: float = 6,
    stroke_width: float = 1.5,
) -> ET.Element:
    """Draw a rectangle."""
    attrs = {
        "x": str(x),
        "y": str(y),
        "width": str(w),
        "height": str(h),
        "fill": fill,
        "rx": str(rx),
        "stroke": stroke or fill,
        "stroke-width": str(stroke_width),
    }
    return _el("rect", attrs, parent=parent)


def _text(
    parent: ET.Element,
    x: float,
    y: float,
    content: str,
    font_size: int = 13,
    fill: str = TEXT_PRIMARY,
    text_anchor: str = "middle",
    bold: bool = False,
) -> ET.Element:
    """Draw a text element."""
    attrs = {
        "x": str(x),
        "y": str(y),
        "font-size": str(font_size),
        "fill": fill,
        "font-family": FONT,
        "text-anchor": text_anchor,
        "dominant-baseline": "middle",
    }
    if bold:
        attrs["font-weight"] = "600"
    return _el("text", attrs, text=content, parent=parent)


def _line(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = BORDER,
    stroke_width: float = 1,
    dash: str | None = None,
) -> ET.Element:
    """Draw a line."""
    attrs = {
        "x1": str(x1),
        "y1": str(y1),
        "x2": str(x2),
        "y2": str(y2),
        "stroke": stroke,
        "stroke-width": str(stroke_width),
    }
    if dash:
        attrs["stroke-dasharray"] = dash
    return _el("line", attrs, parent=parent)


def _make_svg(width: int, height: int) -> ET.Element:
    """Create a root SVG element with dark background."""
    return ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "viewBox": f"0 0 {width} {height}",
            "width": str(width),
            "height": str(height),
            "style": f"background:{BG_DARK}; border-radius:12px;",
        },
    )


def _write_svg(root: ET.Element, output_path: Path) -> None:
    """Write an SVG element tree to file."""
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        str(output_path),
        encoding="unicode",
        xml_declaration=True,
    )


# ---------------------------------------------------------------------------
# Helpers for extracting stats from results dicts
# ---------------------------------------------------------------------------


def _mean_score_by_label(scores: list[dict[str, Any]], label: int) -> float:
    """Compute mean manipulation_score for items matching a label."""
    vals = [s["manipulation_score"] for s in scores if s["label"] == label]
    return sum(vals) / len(vals) if vals else 0.0


def _scores_by_label(scores: list[dict[str, Any]], label: int) -> list[float]:
    """Extract manipulation_score values for a given label."""
    return [s["manipulation_score"] for s in scores if s["label"] == label]


def _quartiles(values: list[float]) -> tuple[float, float, float, float, float]:
    """Compute min, Q1, median, Q3, max for a list of values."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def _percentile(data: list[float], p: float) -> float:
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    return (s[0], _percentile(s, 0.25), _percentile(s, 0.5), _percentile(s, 0.75), s[-1])


# ---------------------------------------------------------------------------
# Chart 1: Summary bar chart
# ---------------------------------------------------------------------------


def generate_summary_bar_chart(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a horizontal bar chart comparing manipulative vs neutral scores.

    Args:
        results: Mapping of dataset name to results dict.
        output_path: Where to write the SVG file.
    """
    W, H = 800, 400
    root = _make_svg(W, H)

    # Title
    _text(
        root, W / 2, 30, "Manipulation Score by Dataset", font_size=17, fill=TEXT_PRIMARY, bold=True
    )
    _text(
        root,
        W / 2,
        52,
        "Mean manipulation_score for manipulative vs non-manipulative samples",
        font_size=11,
        fill=TEXT_SECONDARY,
    )

    datasets = list(results.keys())
    n = len(datasets)
    if n == 0:
        _write_svg(root, output_path)
        return

    # Layout
    margin_left = 160
    margin_right = 80
    bar_area_w = W - margin_left - margin_right
    top_y = 90
    row_h = 70
    bar_h = 18

    # Find max score for scaling
    max_score = 0.1
    for ds_results in results.values():
        for s in ds_results.get("scores", []):
            max_score = max(max_score, s["manipulation_score"])
    max_score = max_score * 1.15  # headroom

    # Legend
    _rect(root, W - 220, 70, 12, 12, COLOR_MANIPULATIVE, stroke="none", rx=2, stroke_width=0)
    _text(root, W - 200, 76, "Manipulative", font_size=10, fill=TEXT_SECONDARY, text_anchor="start")
    _rect(root, W - 120, 70, 12, 12, COLOR_NEUTRAL, stroke="none", rx=2, stroke_width=0)
    _text(root, W - 100, 76, "Neutral", font_size=10, fill=TEXT_SECONDARY, text_anchor="start")

    for i, ds_name in enumerate(datasets):
        ds = results[ds_name]
        scores = ds.get("scores", [])
        mean_manip = _mean_score_by_label(scores, 1)
        mean_neutral = _mean_score_by_label(scores, 0)

        y_center = top_y + i * row_h + row_h / 2

        # Dataset label
        _text(
            root,
            margin_left - 15,
            y_center,
            ds_name.capitalize(),
            font_size=13,
            fill=TEXT_PRIMARY,
            text_anchor="end",
            bold=True,
        )

        # Background track
        _rect(
            root,
            margin_left,
            y_center - bar_h - 2,
            bar_area_w,
            bar_h,
            fill=BG_CARD,
            stroke=BORDER,
            rx=3,
            stroke_width=1,
        )
        _rect(
            root,
            margin_left,
            y_center + 2,
            bar_area_w,
            bar_h,
            fill=BG_CARD,
            stroke=BORDER,
            rx=3,
            stroke_width=1,
        )

        # Manipulative bar (top)
        manip_w = max(2, (mean_manip / max_score) * bar_area_w)
        _rect(
            root,
            margin_left,
            y_center - bar_h - 2,
            manip_w,
            bar_h,
            fill=COLOR_MANIPULATIVE,
            stroke="none",
            rx=3,
            stroke_width=0,
        )
        _text(
            root,
            margin_left + manip_w + 8,
            y_center - bar_h / 2 - 2,
            f"{mean_manip:.2f}",
            font_size=10,
            fill=COLOR_MANIPULATIVE,
            text_anchor="start",
        )

        # Neutral bar (bottom)
        neutral_w = max(2, (mean_neutral / max_score) * bar_area_w)
        _rect(
            root,
            margin_left,
            y_center + 2,
            neutral_w,
            bar_h,
            fill=COLOR_NEUTRAL,
            stroke="none",
            rx=3,
            stroke_width=0,
        )
        _text(
            root,
            margin_left + neutral_w + 8,
            y_center + bar_h / 2 + 2,
            f"{mean_neutral:.2f}",
            font_size=10,
            fill=COLOR_NEUTRAL,
            text_anchor="start",
        )

        # Separator line
        if i < n - 1:
            sep_y = top_y + (i + 1) * row_h
            _line(root, margin_left, sep_y, W - margin_right, sep_y, stroke=BORDER, dash="4 4")

    _write_svg(root, output_path)


# ---------------------------------------------------------------------------
# Chart 2: Separation plot (box plots)
# ---------------------------------------------------------------------------


def generate_separation_plot(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a box-plot style SVG showing score distributions.

    For each dataset, draws two side-by-side simplified box plots
    (median line, Q1-Q3 box, whiskers) for manipulative vs non-manipulative.

    Args:
        results: Mapping of dataset name to results dict.
        output_path: Where to write the SVG file.
    """
    W, H = 800, 500
    root = _make_svg(W, H)

    # Title
    _text(
        root, W / 2, 30, "Score Distribution by Dataset", font_size=17, fill=TEXT_PRIMARY, bold=True
    )
    _text(
        root,
        W / 2,
        52,
        "Box plots: manipulative (red) vs non-manipulative (green)",
        font_size=11,
        fill=TEXT_SECONDARY,
    )

    datasets = list(results.keys())
    n = len(datasets)
    if n == 0:
        _write_svg(root, output_path)
        return

    # Layout
    margin_left = 80
    margin_right = 40
    plot_top = 80
    plot_bottom = H - 60
    plot_h = plot_bottom - plot_top

    col_w = (W - margin_left - margin_right) / max(n, 1)

    # Find global min/max for y-axis
    all_scores: list[float] = []
    for ds_results in results.values():
        for s in ds_results.get("scores", []):
            all_scores.append(s["manipulation_score"])

    if not all_scores:
        _write_svg(root, output_path)
        return

    y_min = min(all_scores) - 0.5
    y_max = max(all_scores) + 0.5
    y_range = y_max - y_min if y_max > y_min else 1.0

    def _y(val: float) -> float:
        """Map a score value to SVG y coordinate (inverted)."""
        return plot_bottom - ((val - y_min) / y_range) * plot_h

    # Y-axis grid lines
    n_grid = 5
    for i in range(n_grid + 1):
        val = y_min + (y_max - y_min) * i / n_grid
        y_pos = _y(val)
        _line(
            root,
            margin_left,
            y_pos,
            W - margin_right,
            y_pos,
            stroke=BORDER,
            dash="2 4",
            stroke_width=0.5,
        )
        _text(
            root,
            margin_left - 10,
            y_pos,
            f"{val:.1f}",
            font_size=9,
            fill=TEXT_DIM,
            text_anchor="end",
        )

    # Y-axis label
    _text(root, 20, (plot_top + plot_bottom) / 2, "Score", font_size=11, fill=TEXT_SECONDARY)

    # Draw box plots for each dataset
    box_w = min(40, col_w * 0.3)
    gap = 8

    for i, ds_name in enumerate(datasets):
        ds = results[ds_name]
        scores = ds.get("scores", [])
        manip_scores = _scores_by_label(scores, 1)
        neutral_scores = _scores_by_label(scores, 0)

        cx = margin_left + col_w * i + col_w / 2

        # Dataset label
        _text(root, cx, H - 25, ds_name.capitalize(), font_size=12, fill=TEXT_PRIMARY, bold=True)

        # Draw manipulative box (left)
        if manip_scores:
            _draw_box(root, cx - gap / 2 - box_w, box_w, manip_scores, _y, COLOR_MANIPULATIVE)

        # Draw neutral box (right)
        if neutral_scores:
            _draw_box(root, cx + gap / 2, box_w, neutral_scores, _y, COLOR_NEUTRAL)

    # Legend
    _rect(root, W - 200, plot_top, 12, 12, COLOR_MANIPULATIVE, stroke="none", rx=2, stroke_width=0)
    _text(
        root,
        W - 182,
        plot_top + 6,
        "Manipulative",
        font_size=10,
        fill=TEXT_SECONDARY,
        text_anchor="start",
    )
    _rect(root, W - 200, plot_top + 20, 12, 12, COLOR_NEUTRAL, stroke="none", rx=2, stroke_width=0)
    _text(
        root,
        W - 182,
        plot_top + 26,
        "Neutral",
        font_size=10,
        fill=TEXT_SECONDARY,
        text_anchor="start",
    )

    _write_svg(root, output_path)


def _draw_box(
    parent: ET.Element,
    x: float,
    w: float,
    values: list[float],
    y_fn: Any,
    color: str,
) -> None:
    """Draw a single box plot (whiskers + IQR box + median line).

    Args:
        parent: SVG parent element.
        x: Left x coordinate of the box.
        w: Width of the box.
        values: Data values.
        y_fn: Function mapping data value to SVG y coordinate.
        color: Fill color.
    """
    mn, q1, med, q3, mx = _quartiles(values)

    y_min = y_fn(mn)
    y_q1 = y_fn(q1)
    y_med = y_fn(med)
    y_q3 = y_fn(q3)
    y_max = y_fn(mx)

    cx = x + w / 2

    # Whisker line (min to max)
    _line(parent, cx, y_min, cx, y_max, stroke=color, stroke_width=1)

    # Min cap
    _line(parent, x + w * 0.25, y_min, x + w * 0.75, y_min, stroke=color, stroke_width=1)

    # Max cap
    _line(parent, x + w * 0.25, y_max, x + w * 0.75, y_max, stroke=color, stroke_width=1)

    # IQR box (Q1 to Q3)
    box_y = min(y_q1, y_q3)
    box_h = abs(y_q3 - y_q1)
    _rect(
        parent, x, box_y, w, max(box_h, 1), fill=color + "33", stroke=color, rx=2, stroke_width=1.5
    )

    # Median line
    _line(parent, x, y_med, x + w, y_med, stroke=color, stroke_width=2)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_all(results_dir: Path, output_dir: Path) -> None:
    """Read all *_results.json files and generate both SVG visualizations.

    Args:
        results_dir: Directory containing *_results.json files.
        output_dir: Directory to write SVG files to.
    """
    results: dict[str, dict[str, Any]] = {}

    for path in sorted(results_dir.glob("*_results.json")):
        with open(path) as f:
            data = json.load(f)
        name = data.get("dataset", path.stem.replace("_results", ""))
        results[name] = data

    if not results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_summary_bar_chart(results, output_dir / "summary_bar_chart.svg")
    generate_separation_plot(results, output_dir / "separation_plot.svg")
