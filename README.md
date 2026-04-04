# Tribe — Content Manipulation Awareness Engine

**Tells you what emotional response content is engineered to trigger.**

## Demo

Analyze a manipulative article and see exactly what emotional triggers it uses:

```
$ python3 -m tribe analyze tests/fixtures/manipulative_article.txt

⚠  This content is designed to trigger a FEAR response.

   It uses name calling/labeling: attaches negative labels to dismiss without argument.

   Manipulation score: 4.0/10
   Primary emotion targeted: Fear
   Secondary: Resonance, Anger, Distrust

   Techniques: name calling/labeling (low), loaded language (low), doubt (low), slogans (low)

   Backend: classifier | Time: 222ms
```

The same text analyzed with `--verbose` shows per-technique confidence:

```
$ python3 -m tribe analyze tests/fixtures/manipulative_article.txt --verbose

⚠  This content is designed to trigger a FEAR response.

   It uses name calling/labeling: attaches negative labels to dismiss without argument.

   Manipulation score: 4.0/10
   Primary emotion targeted: Fear
   Secondary: Resonance, Anger, Distrust

   Techniques: name calling/labeling (low), loaded language (low), doubt (low), slogans (low)

   ─── Techniques Detected ───────────────────────────

   Name Calling/Labeling (36% confidence)
     Target emotion: Contempt
     Attaches negative labels to dismiss without argument

   Loaded Language (26% confidence)
     Target emotion: Anger
     Uses emotionally charged words to influence perception

   Doubt (12% confidence)
     Target emotion: Distrust
     Questions credibility to undermine without evidence

   Slogans (8% confidence)
     Target emotion: Resonance
     Uses catchy phrases to replace critical thinking

   Flag-Waving (6% confidence)
     Target emotion: Tribalism
     Appeals to patriotism or group identity over reason

   Backend: classifier | Time: 222ms
```

A neutral informational article scores lower on manipulation:

```
$ python3 -m tribe analyze tests/fixtures/neutral_article.txt

⚠  This content is designed to trigger a CONTEMPT response.

   It uses name calling/labeling: attaches negative labels to dismiss without argument.

   Manipulation score: 3.3/10
   Primary emotion targeted: Contempt
   Secondary: Familiarity, Anger, Distrust

   Techniques: name calling/labeling (low), loaded language (low), doubt (low), repetition (low)

   Backend: classifier | Time: 216ms
```

JSON output for programmatic use:

```
$ python3 -m tribe analyze tests/fixtures/manipulative_article.txt --json
{
  "primary_trigger": "Fear",
  "trigger_confidence": 0.339,
  "manipulation_score": 4.0,
  "techniques": [
    {
      "name": "Name Calling/Labeling",
      "confidence": 0.363,
      "description": "Attaches negative labels to dismiss without argument",
      "emotion_target": "contempt"
    },
    {
      "name": "Loaded Language",
      "confidence": 0.263,
      "description": "Uses emotionally charged words to influence perception",
      "emotion_target": "anger"
    },
    {
      "name": "Doubt",
      "confidence": 0.124,
      "description": "Questions credibility to undermine without evidence",
      "emotion_target": "distrust"
    },
    {
      "name": "Slogans",
      "confidence": 0.078,
      "description": "Uses catchy phrases to replace critical thinking",
      "emotion_target": "resonance"
    },
    {
      "name": "Flag-Waving",
      "confidence": 0.058,
      "description": "Appeals to patriotism or group identity over reason",
      "emotion_target": "tribalism"
    }
  ],
  "emotions": [
    {"name": "fear", "confidence": 0.25},
    {"name": "anger", "confidence": 0.22},
    {"name": "disgust", "confidence": 0.13}
  ],
  "content_type": "text",
  "content_length": 186,
  "backend": "classifier",
  "processing_time_ms": 222
}
```

Quiet mode for CI/CD pipelines and scripts:

```
$ python3 -m tribe analyze tests/fixtures/manipulative_article.txt --quiet
4.0/10 — Fear (classifier, 220ms)

$ python3 -m tribe analyze tests/fixtures/neutral_article.txt --quiet
3.3/10 — Contempt (classifier, 215ms)
```

Check available backends and hardware:

```
$ python3 -m tribe backends
Tribe — Backend Status
────────────────────────────────────────

Hardware:
  GPU: Apple Silicon (MPS) ✓

Backends:
  Classifier (QCRI 18-technique + DistilRoBERTa emotion): ✓ available
  TRIBE v2: ✗ tribev2 package not installed (pip install tribev2)
```

```
$ python3 -m tribe version
Tribe v0.1.0
Content Manipulation Awareness Engine

Models:
  Technique: QCRI/PropagandaTechniquesAnalysis-en-BERT (18-class)
  Emotion: j-hartmann/emotion-english-distilroberta-base
  Neural: facebook/tribev2 (requires GPU)
  Atlas: Yeo2011 7-Network Parcellation (fsaverage5)
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
# 4.0/10 — Fear (classifier, 220ms)

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

- **Propaganda Detection**: QCRI/PropagandaTechniquesAnalysis-en-BERT (18-class BERT)
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
