"""Tribe realtime demo server — FastAPI + HTML demo page."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

_start_time = time.time()

# ---------------------------------------------------------------------------
# HTML Demo Page
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tribe — Realtime Neural Analysis</title>
  <style>
    :root {
      --bg: #0a0e14;
      --surface: #131920;
      --surface2: #1a2230;
      --border: #1e2832;
      --text: #e6edf3;
      --muted: #7d8590;
      --accent: #58a6ff;
      --score-low: #3fb950;
      --score-mid: #d29922;
      --score-high: #f85149;
    }
    * { box-sizing: border-box; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: ui-monospace, 'SF Mono', 'Fira Code', monospace;
      margin: 0;
      padding: 0;
      min-height: 100vh;
    }
    .container { max-width: 760px; margin: 0 auto; padding: 2rem 1.5rem; }

    /* Header */
    header {
      text-align: center;
      padding: 1.5rem 0 2rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }
    .logo { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
    .logo-slash { color: var(--accent); }
    .tagline { color: var(--muted); font-size: 0.8rem; margin-top: 0.25rem; }
    .brain-icon { font-size: 1.25rem; }

    /* Input section */
    .input-section { margin-bottom: 1rem; }
    textarea {
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 10px;
      padding: 1rem;
      font-family: inherit;
      font-size: 0.85rem;
      min-height: 140px;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
      line-height: 1.6;
    }
    textarea:focus { border-color: var(--accent); }
    textarea::placeholder { color: var(--muted); }

    .controls {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }
    select {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.5rem 0.875rem;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.8rem;
      cursor: pointer;
      outline: none;
    }
    select:focus { border-color: var(--accent); }
    button {
      background: var(--accent);
      color: #000;
      border: none;
      padding: 0.5rem 1.25rem;
      border-radius: 8px;
      font-family: inherit;
      font-weight: 700;
      cursor: pointer;
      font-size: 0.8rem;
      transition: opacity 0.15s, transform 0.1s;
    }
    button:hover { opacity: 0.88; }
    button:active { transform: scale(0.97); }
    button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    .examples { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .examples button {
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.3rem 0.875rem;
      font-size: 0.75rem;
      font-weight: 400;
    }
    .examples button:hover { background: var(--surface2); opacity: 1; }

    /* Loading */
    #loading { display: none; text-align: center; padding: 2.5rem; }
    #loading.visible { display: block; }
    .pulse { animation: pulse 1.4s ease-in-out infinite; font-size: 2.5rem; display: block; margin-bottom: 0.75rem; }
    @keyframes pulse { 0%,100% { opacity: 0.2; transform: scale(1); } 50% { opacity: 1; transform: scale(1.05); } }
    .loading-text { color: var(--muted); font-size: 0.8rem; }
    .loading-sub { color: var(--accent); font-size: 0.75rem; margin-top: 0.25rem; }

    /* Results */
    #results { display: none; }
    #results.visible { display: block; animation: fadeIn 0.3s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

    .result-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 0.875rem;
    }
    .result-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .score-display { text-align: left; }
    .score-num {
      font-size: 3.5rem;
      font-weight: 800;
      line-height: 1;
      transition: color 0.6s ease;
      letter-spacing: -0.03em;
    }
    .score-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.25rem; }
    .score-details { text-align: right; }
    .trigger-badge {
      display: inline-block;
      padding: 0.4rem 1rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      border: 2px solid;
      letter-spacing: 0.02em;
    }
    .confidence-line { font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }
    .confidence-line span { color: var(--text); }

    .meta-row { display: flex; gap: 0.875rem; font-size: 0.72rem; color: var(--muted); flex-wrap: wrap; }
    .badge { padding: 0.15rem 0.5rem; border-radius: 5px; font-weight: 600; }
    .badge-rust { background: #0f2a1a; color: #3fb950; }
    .badge-cls { background: #1a1f2e; color: #58a6ff; }
    .badge-tribe { background: #1a2a0f; color: #85d45b; }

    /* Section title */
    .section-title {
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      color: var(--muted);
      margin: 0 0 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

    /* Network bars */
    .network-bars { display: flex; flex-direction: column; gap: 0.6rem; }
    .network-row { display: grid; grid-template-columns: 150px 1fr 46px; gap: 0.75rem; align-items: center; font-size: 0.75rem; }
    .network-name { color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .network-bar-bg { background: var(--surface2); border-radius: 5px; height: 9px; overflow: hidden; }
    .network-bar-fill { height: 100%; border-radius: 5px; transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1); }
    .network-score { color: var(--muted); text-align: right; font-size: 0.7rem; }
    .ratio-line { font-size: 0.78rem; color: var(--muted); margin-top: 0.875rem; padding-top: 0.875rem; border-top: 1px solid var(--border); }
    .ratio-line strong { color: var(--text); }

    /* Techniques */
    .techniques { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .tech-pill {
      background: var(--surface2);
      border: 1px solid var(--border);
      padding: 0.3rem 0.75rem;
      border-radius: 999px;
      font-size: 0.73rem;
    }
    .tech-pill span { color: var(--muted); }
    .no-techniques { color: var(--muted); font-size: 0.8rem; font-style: italic; }

    /* Error */
    #error-box {
      background: #2d1115;
      border: 1px solid #5c1f26;
      color: #f85149;
      padding: 1rem;
      border-radius: 10px;
      margin-top: 1rem;
      font-size: 0.8rem;
      display: none;
    }
    #error-box.visible { display: block; }

    /* Footer */
    .footer {
      text-align: center;
      color: var(--muted);
      font-size: 0.7rem;
      padding: 2rem 0 1rem;
      margin-top: 1rem;
      border-top: 1px solid var(--border);
    }
    .footer a { color: var(--accent); text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    .keyboard-hint { color: var(--muted); font-size: 0.7rem; margin-top: 0.4rem; }
    .keyboard-hint kbd { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 0.1rem 0.35rem; font-family: inherit; font-size: 0.68rem; }
  </style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo">Tribe <span class="logo-slash">///</span></div>
    <div class="tagline">
      <span class="brain-icon">🧠</span>
      Content Manipulation Awareness Engine — realtime neural analysis
    </div>
  </header>

  <div class="input-section">
    <textarea id="input-text" placeholder="Paste any text to analyze for emotional manipulation triggers..."></textarea>

    <div class="controls">
      <select id="backend-select">
        <option value="auto">Auto-detect backend</option>
        <option value="rust">TRIBE v2 Rust (MacBook Metal)</option>
        <option value="cls">Classifier (CPU, fast)</option>
      </select>
      <button id="analyze-btn" onclick="doAnalyze()">Analyze</button>
      <div class="examples">
        <button onclick="loadExample('fear')">Fear appeal</button>
        <button onclick="loadExample('neutral')">Neutral</button>
        <button onclick="loadExample('outrage')">Outrage</button>
      </div>
    </div>
    <div class="keyboard-hint"><kbd>Ctrl</kbd>+<kbd>Enter</kbd> to analyze</div>
  </div>

  <!-- Loading state -->
  <div id="loading">
    <span class="pulse">🧠</span>
    <div class="loading-text">Running TRIBE v2 neural analysis...</div>
    <div class="loading-sub" id="loading-sub">Loading Metal GPU...</div>
  </div>

  <!-- Error -->
  <div id="error-box"></div>

  <!-- Results -->
  <div id="results">

    <!-- Score + Trigger -->
    <div class="result-card">
      <div class="result-card-header">
        <div class="score-display">
          <div class="score-num" id="score-num">—</div>
          <div class="score-label">Manipulation Score</div>
        </div>
        <div class="score-details">
          <div class="trigger-badge" id="trigger-badge">—</div>
          <div class="confidence-line">Confidence: <span id="confidence">—</span></div>
        </div>
      </div>
      <div class="meta-row">
        <span class="badge badge-rust" id="backend-badge">—</span>
        <span id="time-badge">—</span>
        <span id="words-badge">— words</span>
      </div>
    </div>

    <!-- Neural analysis -->
    <div class="result-card" id="neural-card" style="display:none;">
      <div class="section-title">Predicted Brain Activation — Yeo 2011 7 Networks</div>
      <div class="network-bars" id="network-bars"></div>
      <div class="ratio-line" id="ratio-line"></div>
    </div>

    <!-- Techniques -->
    <div class="result-card" id="tech-card" style="display:none;">
      <div class="section-title">Detected Propaganda Techniques</div>
      <div class="techniques" id="tech-list"></div>
    </div>

  </div>

  <div class="footer">
    Tribe v0.1.0 &nbsp;|&nbsp;
    TRIBE v2 via Metal GPU &nbsp;|&nbsp;
    Yeo 2011 7-Network Parcellation &nbsp;|&nbsp;
    <a href="https://github.com/iota31/tribe">GitHub</a>
  </div>
</div>

<script>
// Example texts
const EXAMPLES = {
  fear: "URGENT: The government has FAILED to protect our children from this deadly threat. Every day you wait is another day of danger. Act NOW before it's too late! Share this with everyone you know — they deserve to know the TRUTH that the mainstream media is hiding from you.",
  neutral: "A recent randomized controlled trial involving 240 participants found that adults who slept between 7 and 9 hours per night performed significantly better on cognitive assessments than those who slept fewer than 6 hours. The study controlled for age, education level, and baseline health status.",
  outrage: "BREAKING: Senator caught taking BRIBES from Big Pharma — proof inside! These corrupt politicians don't care about you. They're laughing all the way to the bank while YOU pay the price. The evidence is DAMNING. Share this to EXPOSE them!"
};

const BACKEND_NAMES = {
  tribe_v2_rust: 'TRIBE v2 Rust (Metal)',
  tribe_v2: 'TRIBE v2',
  classifier: 'Classifier (CPU)'
};

const NETWORK_COLORS = {
  Visual: '#6366f1',
  Somatomotor: '#ec4899',
  Dorsal_Attention: '#f59e0b',
  Ventral_Attention: '#10b981',
  Limbic: '#ef4444',
  Default_Mode: '#8b5cf6',
  Frontoparietal: '#06b6d4',
};

const TRIGGER_COLORS = {
  'Fear': '#ef4444',
  'Self-Referential Anxiety': '#8b5cf6',
  'Outrage': '#f97316',
  'Analytical Engagement': '#06b6d4',
  'Focused Attention': '#f59e0b',
  'Manipulation': '#ec4899',
  'Resonance': '#a78bfa',
  'Distrust': '#f472b6',
  'Contempt': '#fb923c',
};

const BACKEND_BADGE_CLASS = {
  tribe_v2_rust: 'badge-rust',
  tribe_v2: 'badge-tribe',
  classifier: 'badge-cls',
};

function loadExample(key) {
  document.getElementById('input-text').value = EXAMPLES[key];
}

async function doAnalyze() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) return;
  const backend = document.getElementById('backend-select').value;

  document.getElementById('results').classList.remove('visible');
  document.getElementById('error-box').classList.remove('visible');
  document.getElementById('loading').classList.add('visible');
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('loading-sub').textContent = 'Loading Metal GPU...';

  // Simulate loading text update after delay
  const loadingTimer = setTimeout(() => {
    document.getElementById('loading-sub').textContent = 'Running fusion transformer...';
  }, 800);

  try {
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, backend })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Analysis failed');
    renderResults(data);
  } catch (e) {
    document.getElementById('error-box').textContent = '⚠ ' + e.message;
    document.getElementById('error-box').classList.add('visible');
  } finally {
    clearTimeout(loadingTimer);
    document.getElementById('loading').classList.remove('visible');
    document.getElementById('analyze-btn').disabled = false;
  }
}

function scoreColor(score) {
  if (score < 3) return 'var(--score-low)';
  if (score < 6) return 'var(--score-mid)';
  return 'var(--score-high)';
}

function renderResults(data) {
  // Score
  const scoreEl = document.getElementById('score-num');
  scoreEl.textContent = data.manipulation_score.toFixed(1) + '/10';
  scoreEl.style.color = scoreColor(data.manipulation_score);

  // Trigger badge
  const badge = document.getElementById('trigger-badge');
  badge.textContent = data.primary_trigger;
  const triggerColor = TRIGGER_COLORS[data.primary_trigger] || 'var(--accent)';
  badge.style.borderColor = triggerColor;
  badge.style.color = triggerColor;

  // Confidence
  document.getElementById('confidence').textContent = (data.trigger_confidence * 100).toFixed(0) + '%';

  // Meta badges
  const badgeClass = BACKEND_BADGE_CLASS[data.backend] || 'badge-cls';
  document.getElementById('backend-badge').textContent = BACKEND_NAMES[data.backend] || data.backend;
  document.getElementById('backend-badge').className = 'badge ' + badgeClass;
  document.getElementById('time-badge').textContent = (data.processing_time_ms / 1000).toFixed(1) + 's';
  document.getElementById('words-badge').textContent = data.content_length + ' words';

  // Neural analysis
  const neural = data.neural;
  if (neural && neural.network_scores) {
    document.getElementById('neural-card').style.display = 'block';
    const barsEl = document.getElementById('network-bars');
    barsEl.innerHTML = '';
    const maxScore = Math.max(...Object.values(neural.network_scores), 0.001);
    for (const [network, score] of Object.entries(neural.network_scores)) {
      const pct = (score / maxScore * 100).toFixed(1);
      const color = NETWORK_COLORS[network] || '#6366f1';
      barsEl.innerHTML += `
        <div class="network-row">
          <div class="network-name" title="${network}">${network.replace(/_/g, ' ')}</div>
          <div class="network-bar-bg">
            <div class="network-bar-fill" style="width:${pct}%;background:${color};"></div>
          </div>
          <div class="network-score">${score.toFixed(3)}</div>
        </div>`;
    }
    document.getElementById('ratio-line').innerHTML =
      `<strong>Manipulation ratio:</strong> emotional ÷ rational = ${neural.manipulation_ratio.toFixed(3)}` +
      ` &nbsp;|&nbsp; Dominant: <strong>${neural.dominant_network.replace(/_/g, ' ')}</strong>`;
  } else {
    document.getElementById('neural-card').style.display = 'none';
  }

  // Techniques
  if (data.techniques && data.techniques.length > 0) {
    document.getElementById('tech-card').style.display = 'block';
    const techList = document.getElementById('tech-list');
    techList.innerHTML = data.techniques.map(t =>
      `<div class="tech-pill">${t.name} <span>(${(t.confidence * 100).toFixed(0)}%)</span></div>`
    ).join('');
  } else {
    document.getElementById('tech-card').style.display = 'none';
  }

  document.getElementById('results').classList.add('visible');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Ctrl+Enter to analyze
document.getElementById('input-text').addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    doAnalyze();
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    backend: str = "auto"


class HealthResponse(BaseModel):
    status: str
    backend: str
    uptime_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()
    yield


app = FastAPI(
    title="Tribe Demo",
    description="Realtime content manipulation awareness engine — TRIBE v2 on your MacBook",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the HTML demo page."""
    return HTML_TEMPLATE


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    """Analyze text for emotional manipulation.

    Args:
        req.text: The text content to analyze.
        req.backend: "auto" (default), "rust" (TRIBE v2 Rust/Metal), or "cls" (classifier/CPU).

    Returns:
        ContentAnalysis as JSON.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    allowed = {"auto", "tribe", "rust", "cls"}
    if req.backend not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend '{req.backend}'. Must be one of: {', '.join(sorted(allowed))}",
        )

    force = None if req.backend == "auto" else req.backend
    try:
        from tribe.backends.router import get_backend

        backend = get_backend(force_backend=force)
        result = backend.analyze_text(req.text)
        return JSONResponse(content=result.to_dict())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    from tribe.backends.router import get_backend

    return HealthResponse(
        status="ok",
        backend=get_backend().name,
        uptime_seconds=time.time() - _start_time,
    )
