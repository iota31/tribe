# Tribe — Content Manipulation Awareness Engine

**Tells you what emotional response content is engineered to trigger.**

```
$ tribe analyze https://example.com/article

⚠️  This content is engineered to trigger a FEAR response.

   It uses propaganda to influence perception.

   Manipulation score: 7.2/10
   Primary emotion targeted: Fear
   Secondary: Anger, Outrage

   Techniques: fear appeal (high), loaded language (high)

   Neural analysis: salience network activates 3.4x above
   executive control network (TRIBE v2 backend)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.11+.

## Usage

```bash
# Analyze a URL
tribe analyze https://example.com/article

# Analyze a local file
tribe analyze article.txt

# Pipe text directly
cat article.txt | tribe analyze -

# JSON output
tribe analyze article.txt --json

# Detailed breakdown
tribe analyze article.txt --verbose

# Single-line score
tribe analyze article.txt --quiet
# 7.2/10 — Fear (classifier, 3.9s)

# Force specific backend
tribe analyze article.txt --backend cls       # classifier (no GPU needed)
tribe analyze article.txt --backend tribe     # TRIBE v2 (GPU required)

# Hardware and backend info
tribe backends

# Version info
tribe version
```

## How It Works

**Two analysis backends:**

1. **Classifier backend** (runs everywhere) — Uses lightweight transformer models to detect propaganda techniques and emotional manipulation patterns. No GPU needed, ~150MB total.

2. **TRIBE v2 backend** (GPU required) — Uses Meta's TRIBE v2 model to predict actual brain activation patterns. Shows which brain networks activate when you read content. Requires ~10GB VRAM.

Both backends produce the same unified output — a manipulation score from 0-10 and the primary emotional trigger.

**The manipulation signal:** Content that activates attention-capture and emotional brain networks while suppressing rational evaluation networks is engineered to bypass your critical thinking. TRIBE v2 measures this directly. The classifier backend detects it through persuasion techniques.

## Setup

```bash
tribe setup
```

Downloads all required models. For machines without GPU, only downloads classifier models (~150MB). For machines with GPU, also downloads TRIBE v2 (~15GB).

## Hardware Requirements

| Backend | GPU | RAM | VRAM |
|---------|-----|-----|------|
| Classifier | None | 4GB | — |
| TRIBE v2 | NVIDIA RTX 4090 / M-series Mac | 32GB | 10GB+ |

## Architecture

```
URL / file / stdin
        │
        ▼
Content Ingestion
        │
        ▼
Backend Router ─── GPU? ──► TRIBE v2 ──► Neural Analysis ──┐
        │                                                │
        └───── No GPU ──► Classifiers ──► Technique     │
                            Detection                    │
                                                          │
                                           ┌──────────────┘
                                           ▼
                                 ContentAnalysis output
                                           │
                                    Narrative / JSON
```

## Models

- **Propaganda Detection**: IDA-SERICS/PropagandaDetection (DistilBERT)
- **Emotion Classification**: j-hartmann/emotion-english-distilroberta-base
- **Neural Prediction**: facebook/tribev2 (requires GPU)
- **Brain Atlas**: Yeo 2011 7-Network Parcellation (fsaverage5)

## License

Tribe uses a **dual-licensing model**:

### Tribe Package — MIT License

The Tribe content manipulation detection engine (all code in the `tribe/` package, CLI tooling, classifiers, output formatting, and documentation) is licensed under the MIT License.

Copyright (c) 2026 Tushars. See `LICENSE` for full terms.

### TRIBE v2 Components — CC-BY-NC-4.0

The TRIBE v2 neural prediction pipeline (brain atlas files, neural analysis modules, and Meta's TRIBE v2 model integration) is licensed under Creative Commons Attribution-NonCommercial 4.0 International.

TRIBE v2 is by Meta AI and available for non-commercial research use only. See `LICENSE-TRIBE-V2` for full terms and attribution requirements.

**In short:** You can freely use and modify Tribe for non-commercial purposes. Commercial use requires separate licensing — contact the author.

## Why This Exists

Humans detect emotional manipulation in content at ~50% accuracy — a coin flip. Tribe automates the metacognition: noticing that you're being pushed toward a specific emotional response, before that response takes hold.
