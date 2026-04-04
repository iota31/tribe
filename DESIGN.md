# Tribe — Content Manipulation Awareness Engine

## What This Is

A CLI tool that analyzes any content (text, video, audio) and tells you — in plain language — what emotional response it's engineered to trigger, and whether that response is disproportionate to the information being conveyed.

On machines with a GPU, it uses Meta's TRIBE v2 to predict actual brain activation patterns. On machines without, it uses lightweight classifiers that still produce accurate manipulation detection. Both backends produce the same unified output.

```
$ tribe analyze https://example.com/breaking-news-article

⚠️  This content is engineered to trigger a FEAR response.

   It uses loaded language and urgency to bypass rational evaluation.

   Manipulation score: 7.2/10
   Primary emotion targeted: Fear
   Secondary: Urgency, Outrage

   Techniques: fear appeal (high), loaded language (high),
   false urgency (medium)

   Neural analysis: salience network activates 3.4x above
   executive control network (TRIBE v2 backend)
```

## Why It Matters

Humans detect emotional manipulation in content at ~50% accuracy — a coin flip. Content creators, political operators, and engagement-optimized platforms increasingly engineer content to trigger specific emotional responses (anger, fear, outrage, craving) that bypass rational evaluation. The only defense today is individual mindfulness — noticing your own emotional reaction and questioning it. This tool automates that metacognition.

## Core Principles

1. **Local-first.** All analysis runs on your machine. No content is ever sent to external servers.
2. **TRIBE v2 is central.** On machines that can run it, you get real neural predictions. The lightweight fallback is a concession to hardware, not the product.
3. **Human-first output.** The default output is a plain-language narrative, not a JSON blob. A wise friend telling you what's happening, not a data dump.
4. **Honest about its limits.** The classifier backend never claims neural predictions. The tool is transparent about which backend ran and what it can and can't tell you.
5. **Fast and efficient.** Classifier path: <200ms. TRIBE v2 path: <30s for text. No bloat.

## Architecture

```
                    ┌──────────────────────────┐
   Input:           │                          │
   URL / file /     │    Content Ingestion     │
   stdin            │                          │
                    │  URL → trafilatura       │
                    │  Video → ffmpeg extract  │
                    │  Audio → ffmpeg extract  │
                    │  Text → pass through     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Backend Router         │
                    │                          │
                    │   GPU + ≥10GB VRAM?      │
                    │     YES → TRIBE v2       │
                    │     NO  → Classifiers    │
                    │                          │
                    │   --backend tribe|cls    │
                    │   to override            │
                    └───┬────────────────┬─────┘
                        │                │
          ┌─────────────▼──┐    ┌───────▼──────────┐
          │  TRIBE v2      │    │  Classifier      │
          │  Backend       │    │  Backend         │
          │                │    │                  │
          │  LLaMA 3.2    │    │  DistilBERT      │
          │  V-JEPA2      │    │  propaganda det.  │
          │  Wav2Vec-BERT │    │  (67M params)    │
          │  Fusion       │    │                  │
          │               │    │  DistilRoBERTa   │
          │  → 20,484     │    │  emotion det.    │
          │    vertex map  │    │  (82M params)    │
          └───────┬────────┘    └────────┬─────────┘
                  │                      │
          ┌───────▼────────┐    ┌────────▼─────────┐
          │  Neural        │    │  Technique       │
          │  Interpreter   │    │  Interpreter     │
          │                │    │                  │
          │  Yeo 7-network │    │  SemEval         │
          │  parcellation  │    │  technique →     │
          │  → network     │    │  emotion map     │
          │  activation    │    │                  │
          │  scores        │    │  Aggregate →     │
          │                │    │  manipulation    │
          │  Network       │    │  score           │
          │  imbalance →   │    │                  │
          │  manipulation  │    │                  │
          │  signal        │    │                  │
          └───────┬────────┘    └────────┬─────────┘
                  │                      │
                  └──────────┬───────────┘
                             │
                    ┌────────▼─────────────┐
                    │  Unified Output      │
                    │                      │
                    │  ContentAnalysis     │
                    │  schema              │
                    │                      │
                    │  → Narrative render  │
                    │  → JSON (--json)     │
                    └──────────────────────┘
```

## The Two Backends

### TRIBE v2 Backend (GPU, ≥10GB VRAM)

**What it does:** Feeds content through Meta's TRIBE v2 model — three frozen encoders (LLaMA 3.2-3B for text, V-JEPA2 for video, Wav2Vec-BERT for audio) feeding a fusion transformer that predicts fMRI brain activation patterns at 20,484 cortical vertices.

**The interpretation layer:** Uses the Yeo 2011 7-network parcellation (pre-computed atlas for fsaverage5) to group predicted activations into functional brain networks:

| Network | Role | What it tells us |
|---------|------|-----------------|
| Salience (Ventral Attention) | Detects emotionally relevant stimuli | "This content grabbed your attention" |
| Default Mode | Self-referential processing | "This content made it personal" |
| Frontoparietal (Executive Control) | Rational evaluation | "This content engaged your thinking" |
| Ventral Attention | Stimulus-driven attention capture | "You can't look away from this" |
| Limbic | Emotional/social processing | "This activated emotional memory" |
| Dorsal Attention | Voluntary focused attention | "You're actively analyzing this" |
| Visual / Somatomotor | Sensory processing | Baseline, less relevant for manipulation |

**The manipulation signal:**

```python
manipulation_signal = (
    salience_activation +
    default_mode_activation +
    limbic_activation
) / (
    executive_control_activation +
    dorsal_attention_activation
)
```

High ratio = content activates attention-capture and emotional networks while suppressing rational evaluation. This is the neural fingerprint of manipulation.

**Emotion mapping from dominant activated regions (via Desikan-Killiany atlas within each network):**

| Dominant Region | Maps To |
|----------------|---------|
| Insula (within salience network) | Disgust, visceral aversion |
| Anterior cingulate (within salience) | Anxiety, conflict, moral outrage |
| Medial orbitofrontal (within limbic) | Value-laden processing, desire |
| Temporal pole (within limbic) | Social emotion, tribal outrage |
| Lateral orbitofrontal (within limbic) | Impulsive decision pressure |

### Classifier Backend (CPU, ~150MB)

**What it does:** Runs two lightweight classifiers in parallel:

1. **IDA-SERICS/PropagandaDetection** (DistilBERT, 67M params, ~65MB quantized)
   - Trained on SemEval 2023 Task 3 dataset
   - Detects 14+ propaganda techniques at 90% accuracy
   - Inference: 10-50ms on CPU

2. **j-hartmann/emotion-english-distilroberta-base** (DistilRoBERTa, 82M params)
   - 7-class emotion detection: anger, disgust, fear, joy, neutral, sadness, surprise
   - 66% accuracy across diverse datasets
   - Complementary signal for emotional manipulation

**Technique-to-emotion mapping:**

| SemEval Technique | Primary Emotion Target |
|-------------------|----------------------|
| Appeal to Fear/Prejudice | Fear |
| Loaded Language | Anger/Outrage |
| Name Calling/Labeling | Contempt/Disgust |
| Flag-Waving | Pride/Tribalism |
| Exaggeration/Minimisation | Anxiety/Dismissal |
| Bandwagon | Social pressure/FOMO |
| Appeal to Authority | Deference/Trust bypass |
| Doubt | Uncertainty/Distrust |
| Causal Oversimplification | False clarity/Certainty |
| Black-and-White Fallacy | Urgency/No-choice |
| Slogans | Emotional resonance |
| Thought-Terminating Cliche | Cognitive shutdown |
| Whataboutism/Red Herring | Deflection/Confusion |
| Repetition | Familiarity/Acceptance |

**Aggregate manipulation score:**
```python
# Weighted by confidence and known manipulation potency
score = sum(technique.confidence * technique.potency_weight
            for technique in detected_techniques)
# Normalized to 0-10 scale
# Boosted if emotion classifier confirms matching emotional valence
```

## Unified Output Schema

```python
@dataclass
class ContentAnalysis:
    # What the content is trying to trigger
    primary_trigger: str          # "fear", "anger", "outrage", "anxiety", "desire", etc.
    trigger_confidence: float     # 0.0 - 1.0

    # Is the emotional load disproportionate?
    manipulation_score: float     # 0.0 - 10.0 (0 = informational, 10 = pure manipulation)

    # Specific techniques detected
    techniques: list[Technique]   # name, confidence, description, spans (if available)

    # Detected emotions
    emotions: list[Emotion]       # emotion name, confidence

    # Neural prediction (TRIBE v2 only)
    neural: NeuralAnalysis | None  # network scores, manipulation ratio, dominant regions

    # Content metadata
    content_type: str             # "text", "video", "audio", "multimodal"
    content_length: int           # words for text, seconds for media
    source_url: str | None

    # Analysis metadata
    backend: str                  # "tribe_v2" or "classifier"
    processing_time_ms: int
    model_versions: dict          # which model versions were used


@dataclass
class Technique:
    name: str                     # SemEval technique name
    confidence: float             # 0.0 - 1.0
    description: str              # human-readable explanation
    emotion_target: str           # what emotion this technique targets
    spans: list[TextSpan] | None  # where in the text (classifier only)


@dataclass
class NeuralAnalysis:
    network_scores: dict[str, float]   # Yeo 7-network activation scores
    manipulation_ratio: float           # emotional networks / rational networks
    dominant_network: str               # highest-activated network
    dominant_regions: list[str]         # top activated Desikan-Killiany regions
    interpretation: str                 # human-readable neural summary
```

## CLI Interface

```
tribe analyze <input>              Analyze content for manipulation

  <input> can be:
    URL                            Fetches and analyzes the page
    path/to/file                   Analyzes local file (text, video, audio)
    -                              Reads from stdin

  Options:
    --backend tribe|cls|auto       Force analysis backend (default: auto)
    --json                         Output raw ContentAnalysis as JSON
    --verbose                      Show per-technique details and neural data
    --quiet                        Score only (one line: "7.2/10 — Fear")

tribe setup                        Download models, verify GPU, run self-test
tribe backends                     Show available backends and hardware info
tribe version                      Version and model info
```

### Output Examples

**Default (narrative):**
```
$ tribe analyze https://example.com/article

⚠️  This content is engineered to trigger a FEAR response.

   It uses loaded language and urgency to bypass rational evaluation.

   Manipulation score: 7.2/10
   Primary emotion targeted: Fear
   Secondary: Urgency, Outrage

   Techniques: fear appeal (high), loaded language (high),
   false urgency (medium)

   Neural analysis: salience network activates 3.4x above
   executive control network (TRIBE v2 backend)
```

**Quiet mode:**
```
$ tribe analyze --quiet https://example.com/article
7.2/10 — Fear (tribe_v2, 4.2s)
```

**JSON mode:**
```
$ tribe analyze --json https://example.com/article
{
  "primary_trigger": "fear",
  "trigger_confidence": 0.82,
  "manipulation_score": 7.2,
  "techniques": [
    {"name": "Appeal to Fear/Prejudice", "confidence": 0.87, ...},
    {"name": "Loaded Language", "confidence": 0.74, ...}
  ],
  "neural": {
    "network_scores": {
      "Salience": 0.73,
      "Default_Mode": 0.61,
      "Executive_Control": 0.21,
      ...
    },
    "manipulation_ratio": 3.4,
    "dominant_network": "Salience",
    "dominant_regions": ["insula", "rostralanteriorcingulate"]
  },
  "backend": "tribe_v2",
  "processing_time_ms": 4200
}
```

**Verbose mode (adds technique-level detail):**
```
$ tribe analyze --verbose https://example.com/article

⚠️  This content is engineered to trigger a FEAR response.

   Manipulation score: 7.2/10
   Primary emotion targeted: Fear
   Secondary: Urgency, Outrage

   ─── Techniques Detected ───────────────────────────

   Appeal to Fear/Prejudice (87% confidence)
     Target emotion: Fear
     "government has FAILED to protect our children
      from this deadly threat"

   Loaded Language (74% confidence)
     Target emotion: Anger/Outrage
     "deadly threat", "FAILED", "our children"

   False Urgency (61% confidence)
     Target emotion: Panic/Action pressure
     "Act NOW before it's too late"

   ─── Neural Analysis (TRIBE v2) ────────────────────

   Network Activation:
     Salience (attention capture):    ████████░░ 0.73
     Default Mode (self-referential): ██████░░░░ 0.61
     Limbic (emotional):              █████░░░░░ 0.52
     Executive Control (rational):    ██░░░░░░░░ 0.21
     Dorsal Attention (focused):      ███░░░░░░░ 0.29

   Manipulation ratio: 3.4x
   (emotional networks activate 3.4x more than rational networks)

   Dominant regions: anterior insula, anterior cingulate cortex
   Interpretation: Content strongly activates threat detection
   and emotional salience while suppressing analytical processing.

   Backend: TRIBE v2 | Time: 4.2s | GPU: NVIDIA RTX 4090
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | TRIBE v2 is Python/PyTorch, ML ecosystem is Python |
| CLI framework | `click` or `typer` | Clean CLI with subcommands, auto-help |
| TRIBE v2 | PyTorch, HuggingFace | Direct from Meta's repo |
| Classifiers | ONNX Runtime | Faster inference than raw PyTorch on CPU |
| Brain atlas | nibabel + FreeSurfer annot files | Yeo 7-network parcellation |
| URL extraction | trafilatura | Best article text extraction library |
| Media processing | ffmpeg (subprocess) | Extract audio/frames from video |
| Packaging | pip + pyproject.toml | Standard Python packaging |
| Models | HuggingFace Hub | Automatic download on first run |

## Model Download & Setup

```
$ tribe setup

Tribe — Content Manipulation Awareness Engine
──────────────────────────────────────────────

Checking hardware...
  GPU: NVIDIA RTX 4090 (24GB VRAM) ✓
  CUDA: 12.4 ✓
  RAM: 32GB ✓

Downloading models...
  [1/6] DistilBERT propaganda detector (65MB)     ✓
  [2/6] DistilRoBERTa emotion classifier (85MB)    ✓
  [3/6] Yeo 7-network atlas (2MB)                  ✓
  [4/6] LLaMA 3.2-3B encoder (6GB)                 ✓
  [5/6] V-JEPA2 ViT-G encoder (4GB)                ✓
  [6/6] Wav2Vec-BERT 2.0 encoder (2.3GB)           ✓
  [7/7] TRIBE v2 fusion model (1GB)                 ✓

Running self-test...
  Classifier backend: ✓ (23ms on test text)
  TRIBE v2 backend: ✓ (3.8s on test text)

Setup complete. Both backends available.
Run `tribe analyze <url>` to analyze content.
```

For machines without GPU, setup only downloads models 1-3 (~150MB total).

## Project Structure

```
tribe/
├── pyproject.toml
├── README.md
├── LICENSE                        # MIT (our code) + CC-BY-NC note for TRIBE v2
├── tribe/
│   ├── __init__.py
│   ├── cli.py                     # CLI entry point (click/typer)
│   ├── analyze.py                 # Main analysis orchestrator
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── url.py                 # URL fetching + article extraction
│   │   ├── file.py                # Local file reading
│   │   └── media.py               # Video/audio preprocessing via ffmpeg
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── router.py              # Auto-detect GPU, select backend
│   │   ├── tribe_v2.py            # TRIBE v2 pipeline wrapper
│   │   └── classifier.py          # DistilBERT + DistilRoBERTa pipeline
│   ├── interpretation/
│   │   ├── __init__.py
│   │   ├── neural.py              # Yeo 7-network mapping, manipulation ratio
│   │   ├── technique.py           # SemEval technique → emotion mapping
│   │   └── atlas/                 # Bundled Yeo + DK atlas files
│   │       ├── lh.Yeo2011_7Networks_N1000.annot
│   │       └── rh.Yeo2011_7Networks_N1000.annot
│   ├── schema.py                  # ContentAnalysis, Technique, NeuralAnalysis
│   └── output/
│       ├── __init__.py
│       ├── narrative.py           # Human-readable narrative renderer
│       └── json.py                # JSON output
└── tests/
    ├── test_ingestion.py
    ├── test_classifier.py
    ├── test_tribe_backend.py
    ├── test_interpretation.py
    ├── test_output.py
    └── fixtures/
        ├── manipulative_article.txt
        ├── neutral_article.txt
        └── mixed_article.txt
```

## What "Disproportionate" Means

The hardest philosophical question: when is emotional content manipulative vs. legitimate?

**Our approach: network imbalance relative to content type.**

A war correspondent's report *should* trigger some fear — the events are genuinely frightening. A product review *shouldn't* trigger fear. The manipulation signal isn't "does this trigger emotion?" but "does this trigger emotion in a way that bypasses rational processing?"

The manipulation ratio captures this: content that activates salience/emotional networks *while suppressing executive control* is doing something different from content that activates emotion *alongside* rational processing.

- **Manipulation ratio ~1.0**: Emotional content that also engages rational evaluation. Probably legitimate.
- **Manipulation ratio 2.0-3.0**: Emotional content that somewhat suppresses rational processing. Worth noting.
- **Manipulation ratio 3.0+**: Content that strongly activates emotional capture while bypassing analytical thinking. Likely engineered.

For the classifier backend, we approximate this by looking at the density and confidence of detected techniques. An article using 1 technique at low confidence is probably normal writing. An article using 5+ techniques at high confidence is almost certainly engineered.

## Cheapest Possible Fallback

For machines that can't run anything locally (old hardware, Chromebooks, phones):

**Option 1: Pre-built WASM module.**
The classifier models (DistilBERT + DistilRoBERTa, ~150MB total) can be compiled to ONNX → WASM. Runs in any browser. No server. This is the fallback — we package the classifiers as a WASM module with a simple web UI.

**Option 2: Hosted API (community-run).**
A self-hostable FastAPI server that anyone can run. Community members with GPUs can host instances. We provide Docker images. Distributed, no single point of failure.

The design explicitly does NOT require a centralized cloud service. Every deployment option is either local or self-hosted.
