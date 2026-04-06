"""Tribe realtime demo server — FastAPI + HTML demo page."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
    .container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
    .results-grid { display: grid; grid-template-columns: 500px 1fr; gap: 1.25rem; align-items: start; }
    @media (max-width: 960px) { .results-grid { grid-template-columns: 1fr; } }

    /* Brain canvas */
    #brain-container {
      position: relative;
      width: 500px;
      height: 500px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }
    #brain-container canvas { display: block; width: 100% !important; height: 100% !important; }
    #brain-loading {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 0.8rem;
      z-index: 2;
    }
    #brain-loading .pulse { font-size: 2rem; }
    .brain-label {
      position: absolute;
      color: var(--text);
      font-size: 0.65rem;
      pointer-events: none;
      white-space: nowrap;
      text-shadow: 0 0 6px rgba(0,0,0,0.9);
      z-index: 3;
      opacity: 0;
      transition: opacity 0.4s;
    }
    .brain-label.visible { opacity: 1; }

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
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.min.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"
    }
  }
  </script>
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
      <select id="backend-select" disabled>
        <option value="rust">TRIBE v2 Rust (MacBook Metal)</option>
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
    <div class="results-grid">

      <!-- Left column: 3D brain -->
      <div id="brain-container">
        <div id="brain-loading">
          <span class="pulse">🧠</span>
          <div>Loading brain model...</div>
        </div>
      </div>

      <!-- Right column: scores -->
      <div class="results-scores">

        <!-- Score + Trigger -->
        <div class="result-card">
          <div class="result-card-header">
            <div class="score-display">
              <div class="score-num" id="score-num">-</div>
              <div class="score-label">Manipulation Score</div>
            </div>
            <div class="score-details">
              <div class="trigger-badge" id="trigger-badge">-</div>
              <div class="confidence-line">Confidence: <span id="confidence">-</span></div>
            </div>
          </div>
          <div class="meta-row">
            <span class="badge badge-rust" id="backend-badge">-</span>
            <span id="time-badge">-</span>
            <span id="words-badge">- words</span>
          </div>
        </div>

        <!-- Neural analysis -->
        <div class="result-card" id="neural-card" style="display:none;">
          <div class="section-title">Predicted Brain Activation - Yeo 2011 7 Networks</div>
          <div class="network-bars" id="network-bars"></div>
          <div class="ratio-line" id="ratio-line"></div>
        </div>

        <!-- Techniques -->
        <div class="result-card" id="tech-card" style="display:none;">
          <div class="section-title">Detected Propaganda Techniques</div>
          <div class="techniques" id="tech-list"></div>
        </div>

      </div>
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
};

function loadExample(key) {
  document.getElementById('input-text').value = EXAMPLES[key];
}

async function doAnalyze() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) return;
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
      body: JSON.stringify({ text })
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
  const badgeClass = BACKEND_BADGE_CLASS[data.backend] || 'badge-rust';
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

  // Update 3D brain if neural data available
  if (data.neural && window.updateBrain) {
    window.updateBrain(data.neural);
  }
}

// Ctrl+Enter to analyze
document.getElementById('input-text').addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    doAnalyze();
  }
});
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// -- Scene setup --
const container = document.getElementById('brain-container');
const width = 500, height = 500;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
container.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 2000);
camera.position.set(0, 20, 280);

const scene = new THREE.Scene();

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);
const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.7);
dirLight1.position.set(100, 150, 100);
scene.add(dirLight1);
const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.35);
dirLight2.position.set(-80, -60, -100);
scene.add(dirLight2);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3 * 6; // 0.3 RPM (speed is in deg/s, 1 RPM = 6 deg/s)
controls.minDistance = 120;
controls.maxDistance = 500;

// -- Region meshes --
const regionMeshes = new Map();
const brainGroup = new THREE.Group();
scene.add(brainGroup);

// Target colors for lerp animation
const targetColors = new Map();
const targetEmissive = new Map();

// Persuasion region name mapping (lowercase keys for matching)
const PERSUASION_REGIONS = {
  'vmPFC': ['medialorbitofrontal', 'rostralanteriorcingulate'],
  'dlPFC': ['rostralmiddlefrontal', 'caudalmiddlefrontal'],
  'TPJ': ['supramarginal', 'inferiorparietal', 'bankssts'],
  'precuneus': ['precuneus'],
  'temporal_pole': ['temporalpole'],
  'insula': ['insula']
};

// Build reverse map: mesh-name -> persuasion region
const meshToPersuasion = {};
for (const [region, meshNames] of Object.entries(PERSUASION_REGIONS)) {
  for (const m of meshNames) {
    meshToPersuasion[m.toLowerCase()] = region;
  }
}

// Yeo network to region mapping (approximate DK atlas assignment)
const REGION_TO_NETWORK = {
  'pericalcarine': 'Visual', 'cuneus': 'Visual', 'lingual': 'Visual',
  'lateraloccipital': 'Visual', 'fusiform': 'Visual',
  'precentral': 'Somatomotor', 'postcentral': 'Somatomotor',
  'paracentral': 'Somatomotor',
  'superiorfrontal': 'Frontoparietal', 'parstriangularis': 'Frontoparietal',
  'parsopercularis': 'Frontoparietal', 'parsorbitalis': 'Frontoparietal',
  'frontalpole': 'Frontoparietal',
  'superiorparietal': 'Dorsal_Attention', 'inferiorparietal': 'Dorsal_Attention',
  'supramarginal': 'Ventral_Attention', 'bankssts': 'Ventral_Attention',
  'isthmuscingulate': 'Default_Mode', 'posteriorcingulate': 'Default_Mode',
  'precuneus': 'Default_Mode', 'medialorbitofrontal': 'Default_Mode',
  'superiortemporal': 'Default_Mode', 'middletemporal': 'Default_Mode',
  'inferiortemporal': 'Default_Mode',
  'rostralanteriorcingulate': 'Limbic', 'caudalanteriorcingulate': 'Limbic',
  'entorhinal': 'Limbic', 'parahippocampal': 'Limbic',
  'temporalpole': 'Limbic', 'insula': 'Limbic',
  'transversetemporal': 'Somatomotor',
  'rostralmiddlefrontal': 'Frontoparietal',
  'caudalmiddlefrontal': 'Frontoparietal',
  'lateralorbitofrontal': 'Limbic',
};

// Color ramp: cold blue -> neutral gray -> hot red
function activationColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min + 0.0001)));
  const r = t < 0.5 ? t * 2 * 0.3 : 0.3 + (t - 0.5) * 2 * 0.7;
  const g = t < 0.5 ? 0.1 + t * 0.3 : 0.25 - (t - 0.5) * 0.3;
  const b = t < 0.5 ? 0.35 - t * 0.3 : 0.2 - (t - 0.5) * 0.3;
  return new THREE.Color(r, g, b);
}

// -- Load brain meshes --
const loader = new OBJLoader();

async function loadBrain() {
  try {
    const resp = await fetch('/static/brain/regions.json');
    if (!resp.ok) {
      console.warn('No brain regions.json found at /static/brain/regions.json');
      document.getElementById('brain-loading').innerHTML =
        '<div style="color:#7d8590;font-size:0.75rem;">Brain model not found.<br>Place OBJ files in tribe/static/brain/</div>';
      return;
    }
    const manifest = await resp.json();
    const files = manifest.regions || manifest.files || manifest;

    // Load all OBJ files in parallel
    const promises = (Array.isArray(files) ? files : Object.keys(files)).map(async (entry) => {
      const filename = typeof entry === 'string' ? entry : entry.file;
      const name = filename.replace('.obj', '').replace(/^(lh_|rh_)/, '');
      const hemi = filename.startsWith('rh_') ? 'rh' : 'lh';
      const url = '/static/brain/' + filename;
      try {
        const obj = await new Promise((resolve, reject) => {
          loader.load(url, resolve, undefined, reject);
        });
        obj.traverse((child) => {
          if (child.isMesh) {
            const mat = new THREE.MeshPhongMaterial({
              color: 0x1a2a3a,
              shininess: 30,
              transparent: true,
              opacity: 0.85,
              side: THREE.DoubleSide,
            });
            child.material = mat;
            const key = hemi + '_' + name.toLowerCase();
            regionMeshes.set(key, child);
            targetColors.set(key, new THREE.Color(0x1a2a3a));
            targetEmissive.set(key, new THREE.Color(0x000000));
          }
        });
        brainGroup.add(obj);
      } catch (e) {
        console.warn('Failed to load', url, e);
      }
    });

    await Promise.all(promises);

    // Center the brain group
    const box = new THREE.Box3().setFromObject(brainGroup);
    const center = box.getCenter(new THREE.Vector3());
    brainGroup.position.sub(center);

    // FreeSurfer RAS -> good default view: rotate so superior is up
    brainGroup.rotation.x = -Math.PI / 2;

    // Adjust camera to fit
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.set(0, 0, size * 1.1);
    controls.target.set(0, 0, 0);
    controls.update();

    // Hide loading
    document.getElementById('brain-loading').style.display = 'none';
    console.log('Brain loaded:', regionMeshes.size, 'region meshes');
  } catch (e) {
    console.error('Brain load error:', e);
    document.getElementById('brain-loading').innerHTML =
      '<div style="color:#f85149;font-size:0.75rem;">Error loading brain model</div>';
  }
}

loadBrain();

// -- Persuasion labels --
const labelEls = {};
const PERSUASION_DISPLAY = {
  'vmPFC': 'vmPFC', 'dlPFC': 'dlPFC', 'TPJ': 'TPJ',
  'precuneus': 'Precuneus', 'temporal_pole': 'Temporal Pole', 'insula': 'Insula'
};
for (const region of Object.keys(PERSUASION_DISPLAY)) {
  const el = document.createElement('div');
  el.className = 'brain-label';
  el.textContent = PERSUASION_DISPLAY[region] + ': -';
  container.appendChild(el);
  labelEls[region] = el;
}

// Project 3D centroid to 2D screen
function projectToScreen(mesh) {
  if (!mesh || !mesh.geometry) return null;
  if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
  const pos = new THREE.Vector3();
  mesh.localToWorld(pos.copy(mesh.geometry.boundingSphere.center));
  pos.project(camera);
  return {
    x: (pos.x * 0.5 + 0.5) * width,
    y: (-pos.y * 0.5 + 0.5) * height
  };
}

function updateLabels(persuasionScores) {
  for (const [region, displayName] of Object.entries(PERSUASION_DISPLAY)) {
    const el = labelEls[region];
    if (!el) continue;

    // Only update text when scores are provided (not on position-only updates)
    if (persuasionScores) {
      const val = persuasionScores[region];
      el.textContent = displayName + ': ' + (val != null ? val.toFixed(2) : '-');
    }

    // Find a representative mesh for this region
    const meshNames = PERSUASION_REGIONS[region] || [];
    let bestMesh = null;
    for (const mn of meshNames) {
      const m = regionMeshes.get('lh_' + mn.toLowerCase()) || regionMeshes.get('rh_' + mn.toLowerCase());
      if (m) { bestMesh = m; break; }
    }
    if (bestMesh) {
      const pt = projectToScreen(bestMesh);
      if (pt) {
        el.style.left = pt.x + 'px';
        el.style.top = pt.y + 'px';
        el.style.transform = 'translate(-50%, -50%)';
        el.classList.add('visible');
      }
    }
  }
}

// -- updateBrain: called from renderResults --
window.updateBrain = function(neuralData) {
  const persuasion = neuralData.persuasion_scores || {};
  const networks = neuralData.network_scores || {};

  // Compute min/max for persuasion scores
  const pVals = Object.values(persuasion);
  const pMin = pVals.length ? Math.min(...pVals) : 0;
  const pMax = pVals.length ? Math.max(...pVals) : 1;

  // Compute min/max for network scores
  const nVals = Object.values(networks);
  const nMin = nVals.length ? Math.min(...nVals) : 0;
  const nMax = nVals.length ? Math.max(...nVals) : 1;

  for (const [key, mesh] of regionMeshes) {
    // Strip hemisphere prefix to get region name
    const regionName = key.replace(/^(lh_|rh_)/, '');

    // Check if this is a persuasion region
    const persuasionKey = meshToPersuasion[regionName];
    if (persuasionKey && persuasion[persuasionKey] != null) {
      const val = persuasion[persuasionKey];
      const col = activationColor(val, pMin, pMax);
      targetColors.set(key, col);
      // Emissive glow for persuasion regions
      const intensity = Math.max(0, Math.min(1, (val - pMin) / (pMax - pMin + 0.0001)));
      targetEmissive.set(key, new THREE.Color(intensity * 0.3, intensity * 0.05, intensity * 0.02));
    } else {
      // Color by Yeo network
      const net = REGION_TO_NETWORK[regionName];
      if (net && networks[net] != null) {
        const col = activationColor(networks[net], nMin, nMax);
        targetColors.set(key, col);
        targetEmissive.set(key, new THREE.Color(0, 0, 0));
      }
    }
  }

  updateLabels(persuasion);
};

// -- Render loop --
function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Smooth color transitions via lerp
  for (const [key, mesh] of regionMeshes) {
    if (mesh.material) {
      const tc = targetColors.get(key);
      const te = targetEmissive.get(key);
      if (tc) mesh.material.color.lerp(tc, 0.06);
      if (te) mesh.material.emissive.lerp(te, 0.06);
    }
  }

  // Update label positions each frame for orbit
  updateLabels(null);

  renderer.render(scene, camera);
}
animate();

// Handle resize
const ro = new ResizeObserver(() => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});
ro.observe(container);
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    text: str


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
    description="Realtime content manipulation awareness engine - TRIBE v2 on your MacBook",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the HTML demo page."""
    return HTML_TEMPLATE


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    """Analyze text for emotional manipulation using the TRIBE v2 Rust backend.

    Args:
        req.text: The text content to analyze.

    Returns:
        ContentAnalysis as JSON.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        from tribe.backends.router import get_backend

        backend = get_backend()
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
