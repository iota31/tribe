"""Generate architecture diagram SVGs for the README."""

import xml.etree.ElementTree as ET
import math

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

SVG_NS = "http://www.w3.org/2000/svg"


def el(tag, attrs=None, text=None, parent=None):
    """Create an SVG element."""
    e = ET.Element(tag, attrs or {})
    if text:
        e.text = text
    if parent is not None:
        parent.append(e)
    return e


def rect(parent, x, y, w, h, fill, stroke=None, rx=6, stroke_width=1.5):
    attrs = {
        "x": str(x), "y": str(y),
        "width": str(w), "height": str(h),
        "fill": fill, "rx": str(rx),
        "stroke": stroke or fill, "stroke-width": str(stroke_width),
    }
    return el("rect", attrs, parent=parent)


def text(parent, x, y, content, font_size=13, fill="#e6edf3",
         font_family="ui-monospace,SFMono,Menlo,monospace", text_anchor="middle", bold=False):
    attrs = {
        "x": str(x), "y": str(y),
        "font-size": str(font_size),
        "fill": fill,
        "font-family": font_family,
        "text-anchor": text_anchor,
        "dominant-baseline": "middle",
    }
    if bold:
        attrs["font-weight"] = "600"
    t = el("text", attrs, text=content, parent=parent)
    return t


def tspan(parent, content, font_size=12, fill="#7d8590", bold=False):
    attrs = {
        "font-size": str(font_size),
        "fill": fill,
        "x": parent.get("x") if parent.get else "0",
        "dy": "1.2em",
    }
    if bold:
        attrs["font-weight"] = "600"
        attrs["fill"] = "#e6edf3"
    ts = el("tspan", attrs, text=content, parent=parent)
    return ts


def arrow(parent, x1, y1, x2, y2, color="#1e2832", stroke_width=1.5):
    attrs = {
        "x1": str(x1), "y1": str(y1),
        "x2": str(x2), "y2": str(y2),
        "stroke": color, "stroke-width": str(stroke_width),
        "marker-end": "url(#arrowhead)",
    }
    return el("line", attrs, parent=parent)


def chip(parent, x, y, label, color):
    """Small colored label chip."""
    rect(parent, x, y, 70, 20, color, stroke="none", rx=4, stroke_width=0)
    text(parent, x + 35, y + 10, label, font_size=10, fill="#fff", bold=True)


# ─────────────────────────────────────────────────────────────────────────────
# SVG 1: Pipeline Architecture
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline_svg():
    W, H = 920, 520
    root = ET.Element("svg", {
        "xmlns": SVG_NS, "viewBox": f"0 0 {W} {H}",
        "width": str(W), "height": str(H),
        "style": "background:#0a0e14; border-radius:12px;",
    })

    # Defs
    defs = el("defs", parent=root)
    marker = el("marker", {
        "id": "arrowhead", "markerWidth": "8", "markerHeight": "6",
        "refX": "7", "refY": "3", "orient": "auto",
    }, parent=defs)
    el("polygon", {
        "points": "0 0, 8 3, 0 6", "fill": "#30363d",
    }, parent=marker)

    # ── Title ──
    text(root, W / 2, 30, "How Tribe Works — Multimodal Pipeline",
         font_size=17, fill="#e6edf3", bold=True)

    # ── Row 1: Three input modality boxes ──
    box_w = 260
    box_h = 80
    gap = 35
    total_w = 3 * box_w + 2 * gap
    start_x = (W - total_w) / 2
    row1_y = 60

    inputs = [
        ("Text", "Articles, URLs, stdin", "LLaMA 3.2", "#6366f1"),
        ("Audio", "Podcasts, speeches, WAV", "Wav2Vec-BERT", "#ec4899"),
        ("Video", "News clips, ads, MP4", "V-JEPA2", "#f59e0b"),
    ]

    for i, (title, subtitle, model, color) in enumerate(inputs):
        bx = start_x + i * (box_w + gap)
        rect(root, bx, row1_y, box_w, box_h,
             fill="#131920", stroke=color, rx=8, stroke_width=1.5)
        text(root, bx + box_w / 2, row1_y + 22, title,
             font_size=14, fill="#e6edf3", bold=True)
        text(root, bx + box_w / 2, row1_y + 40, subtitle,
             font_size=11, fill="#7d8590")
        chip(root, bx + box_w / 2 - 35, row1_y + 52, model, color)

    # ── Converging arrows from 3 boxes down to fusion ──
    arrow_start_y = row1_y + box_h + 2
    arrow_end_y = row1_y + box_h + 38
    center_x = W / 2

    for i in range(3):
        bx = start_x + i * (box_w + gap) + box_w / 2
        arrow(root, bx, arrow_start_y, center_x, arrow_end_y, color="#30363d")

    # ── Row 2: Fusion Transformer ──
    row2_y = row1_y + box_h + 40
    fusion_w = 580
    fusion_h = 70
    fusion_x = (W - fusion_w) / 2

    rect(root, fusion_x, row2_y, fusion_w, fusion_h,
         fill="#131920", stroke="#8b5cf6", rx=8, stroke_width=1.5)
    text(root, W / 2, row2_y + 20,
         "Fusion Transformer  —  eugenehp/tribev2",
         font_size=14, fill="#e6edf3", bold=True)
    text(root, W / 2, row2_y + 38,
         "Predicts fMRI brain activation at 20,484 cortical vertices",
         font_size=11, fill="#7d8590")
    chip(root, W / 2 - 50, row2_y + 48, "Brain Encoding", "#8b5cf6")

    # ── Arrow down ──
    arrow(root, center_x, row2_y + fusion_h + 2,
          center_x, row2_y + fusion_h + 28, color="#30363d")

    # ── Row 3: Yeo Parcellation ──
    row3_y = row2_y + fusion_h + 30
    parc_w = 460
    parc_h = 55
    parc_x = (W - parc_w) / 2

    rect(root, parc_x, row3_y, parc_w, parc_h,
         fill="#131920", stroke="#06b6d4", rx=8, stroke_width=1.5)
    text(root, W / 2, row3_y + 20,
         "Yeo 2011 Parcellation",
         font_size=14, fill="#e6edf3", bold=True)
    text(root, W / 2, row3_y + 38,
         "Maps activation to 7 functional networks",
         font_size=11, fill="#7d8590")

    # ── Arrow down ──
    arrow(root, center_x, row3_y + parc_h + 2,
          center_x, row3_y + parc_h + 28, color="#30363d")

    # ── Row 4: Interpretation ──
    row4_y = row3_y + parc_h + 30
    interp_w = 460
    interp_h = 55
    interp_x = (W - interp_w) / 2

    rect(root, interp_x, row4_y, interp_w, interp_h,
         fill="#131920", stroke="#ef4444", rx=8, stroke_width=1.5)
    text(root, W / 2, row4_y + 20,
         "Neural Interpretation",
         font_size=14, fill="#e6edf3", bold=True)
    text(root, W / 2, row4_y + 38,
         "Emotional / rational ratio -> manipulation score",
         font_size=11, fill="#7d8590")

    # ── Arrow down ──
    arrow(root, center_x, row4_y + interp_h + 2,
          center_x, row4_y + interp_h + 28, color="#30363d")

    # ── Row 5: Result (green border) ──
    row5_y = row4_y + interp_h + 30
    result_w = 520
    result_h = 55
    result_x = (W - result_w) / 2

    rect(root, result_x, row5_y, result_w, result_h,
         fill="#0d1117", stroke="#3fb950", rx=8, stroke_width=1.5)
    text(root, W / 2, row5_y + 20,
         "1.3/10  —  Fear  —  Salience dominant",
         font_size=14, fill="#3fb950", bold=True)
    text(root, W / 2, row5_y + 38,
         "Manipulation ratio: 0.63  |  7 network scores",
         font_size=11, fill="#7d8590")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        "/Users/tushars/PycharmProjects/tribe/images/architecture.svg",
        encoding="unicode",
        xml_declaration=True,
    )
    print("Saved architecture.svg")


# ─────────────────────────────────────────────────────────────────────────────
# SVG 2: Yeo 7 Brain Networks
# ─────────────────────────────────────────────────────────────────────────────

def make_brain_svg():
    W, H = 920, 360
    root = ET.Element("svg", {
        "xmlns": SVG_NS, "viewBox": f"0 0 {W} {H}",
        "width": str(W), "height": str(H),
        "style": "background:#0a0e14; border-radius:12px;",
    })

    # Title
    text(root, W / 2, 28, "Yeo 2011 — 7 Functional Brain Networks",
         font_size=17, fill="#e6edf3", bold=True)
    text(root, W / 2, 50,
         "TRIBE v2 maps 20,484 cortical vertices into 7 networks — emotional vs rational drives manipulation",
         font_size=11, fill="#7d8590")

    # ── Left: Network bars ──
    bar_x = 30
    bar_y = 80
    bar_w = 320
    bar_h = 28
    gap = 6

    networks = [
        ("Salience",           0.011,  "#ef4444", "⚠ Threat detection · Rapid emotional response"),
        ("Default Mode",       0.009,  "#f97316", "💭 Self-referential · Social cognition"),
        ("Limbic",             0.004,  "#fbbf24", "❤️ Emotional memory · Goal-driven"),
        ("Dorsal Attention",    0.024,  "#06b6d4", "👁️ Top-down focus · Voluntary attention"),
        ("Executive Control",   0.005,  "#3b82f6", "🧠 Problem solving · Rational evaluation"),
        ("Somatomotor",        0.001,  "#6b7280", "🤲 Motor control · Interoception"),
        ("Visual",             0.020,  "#9ca3af", "👁️ Visual processing"),
    ]

    max_val = max(s for _, s, _, _ in networks)

    for i, (name, val, color, desc) in enumerate(networks):
        y = bar_y + i * (bar_h + gap)
        # Label
        bold_label = name in ("Salience", "Default Mode", "Limbic")
        label_color = "#fca5a5" if bold_label else "#9ca3af"
        text(root, bar_x, y + bar_h / 2, name,
             font_size=11, fill=label_color,
             text_anchor="start", bold=bold_label)
        # Bar background
        rect(root, bar_x + 110, y, bar_w, bar_h,
             fill="#131920", stroke="#1e2832", rx=4, stroke_width=1)
        # Bar fill
        pct = val / max_val
        bar_fill_w = int(bar_w * pct)
        if bar_fill_w > 0:
            rect(root, bar_x + 110, y, bar_fill_w, bar_h,
                 fill=color, stroke="none", rx=4, stroke_width=0)
        # Value
        text(root, bar_x + 110 + bar_w + 8, y + bar_h / 2,
             f"{val:.3f}", font_size=10, fill="#7d8590",
             text_anchor="start")
        # Description
        text(root, bar_x + 110, y + bar_h + 4,
             desc, font_size=9, fill="#4b5563",
             text_anchor="start")

    # ── Middle: Legend ──
    leg_x = 510
    leg_y = 90

    # Emotional box
    rect(root, leg_x, leg_y, 180, 110,
         fill="#1f0a0a", stroke="#ef4444", rx=8, stroke_width=1.5)
    text(root, leg_x + 90, leg_y + 20, "EMOTIONAL NETWORKS",
         font_size=11, fill="#ef4444", bold=True)
    text(root, leg_x + 90, leg_y + 42,
         "Activated during emotional",
         font_size=10, fill="#9ca3af")
    text(root, leg_x + 90, leg_y + 57,
         "processing and threat detection.",
         font_size=10, fill="#9ca3af")
    text(root, leg_x + 90, leg_y + 80,
         "High activation = manipulation",
         font_size=10, fill="#fca5a5")
    text(root, leg_x + 90, leg_y + 95,
         "signal (ratio goes up)",
         font_size=10, fill="#fca5a5")

    # Rational box
    rect(root, leg_x, leg_y + 125, 180, 110,
         fill="#0a1a1f", stroke="#3b82f6", rx=8, stroke_width=1.5)
    text(root, leg_x + 90, leg_y + 145, "RATIONAL NETWORKS",
         font_size=11, fill="#3b82f6", bold=True)
    text(root, leg_x + 90, leg_y + 167,
         "Activated during analytical",
         font_size=10, fill="#9ca3af")
    text(root, leg_x + 90, leg_y + 182,
         "thinking and evaluation.",
         font_size=10, fill="#9ca3af")
    text(root, leg_x + 90, leg_y + 205,
         "High activation = lower",
         font_size=10, fill="#93c5fd")
    text(root, leg_x + 90, leg_y + 220,
         "manipulation ratio",
         font_size=10, fill="#93c5fd")

    # ── Right: Formula ──
    form_x = 730
    form_y = 110

    rect(root, form_x, form_y, 170, 200,
         fill="#0d1117", stroke="#30363d", rx=8, stroke_width=1)
    text(root, form_x + 85, form_y + 25,
         "Manipulation Ratio",
         font_size=13, fill="#e6edf3", bold=True)
    text(root, form_x + 85, form_y + 55,
         "Salience + Limbic + Default",
         font_size=10, fill="#9ca3af")
    text(root, form_x + 85, form_y + 70,
         "─────────────────",
         font_size=10, fill="#30363d")
    text(root, form_x + 85, form_y + 85,
         "Exec + Dorsal Attention",
         font_size=10, fill="#9ca3af")
    text(root, form_x + 85, form_y + 108,
         "= 0.63",
         font_size=20, fill="#f59e0b", bold=True)
    text(root, form_x + 85, form_y + 130,
         "Ratio > 1.0 = emotional dominant",
         font_size=9, fill="#6b7280")
    text(root, form_x + 85, form_y + 145,
         "Ratio < 1.0 = rational dominant",
         font_size=9, fill="#6b7280")
    text(root, form_x + 85, form_y + 165,
         "Fear appeal example: 0.63",
         font_size=10, fill="#f59e0b")
    text(root, form_x + 85, form_y + 182,
         "(Salience barely above rational)",
         font_size=9, fill="#6b7280")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        "/Users/tushars/PycharmProjects/tribe/images/brain-networks.svg",
        encoding="unicode",
        xml_declaration=True,
    )
    print("Saved brain-networks.svg")


if __name__ == "__main__":
    make_pipeline_svg()
    make_brain_svg()
