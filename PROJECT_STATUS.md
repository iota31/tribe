# Tribe — Project Status

**Last updated:** 2026-04-05
**GitHub:** https://github.com/iota31/tribe (public, dual-licensed: MIT + CC-BY-NC-4.0)
**Build status:** Working — 29/29 tests passing, CLI fully functional
**Classifier backend:** Working on Apple Silicon MPS
**TRIBE v2 Rust backend:** Working — MacBook M1 Pro via tribev2-rs + Metal GPU (~4s inference)

---

## What's Working Right Now

### CLI Commands
```bash
python3 -m tribe --help                    # CLI help
python3 -m tribe backends                    # Hardware detection
python3 -m tribe version                     # Version info
python3 -m tribe analyze <file|url> --backend cls   # Analysis
python3 -m tribe analyze <file|url> --json         # JSON output
python3 -m tribe analyze <file|url> --quiet        # Single line
python3 -m tribe analyze <file|url> --verbose       # Detailed breakdown
python3 -m tribe setup                       # Download models + verify
```

### Live Test Results
- Manipulative article (tests/fixtures/manipulative_article.txt):
  - **Rust backend (TRIBE v2, Metal GPU):** 1.2/10 — Focused Attention (27950ms)
  - **Classifier backend (MPS):** 4.0/10 — Fear (222ms)
- Neutral article (tests/fixtures/neutral_article.txt): **3.3/10** — Contempt (216ms)

### TRIBE v2 Rust — How It Works
```
LLaMA 3.2 3B GGUF (Ollama, Metal GPU, ~740ms)
    → Text features (layer 0.5 + 1.0, 6144 dims)
    → Fusion transformer (eugenehp/tribev2, Metal GPU, ~2s)
    → fMRI predictions: 100 timesteps × 20484 vertices (float32)
    → Yeo 2011 7-network interpretation → manipulation_ratio

Total: ~4 seconds on M1 Pro 16GB
```

---

## Project Structure

```
tribe/
├── pyproject.toml              # Package config, dependencies
├── README.md                   # User-facing docs with real output examples
├── DESIGN.md                   # Full product design spec
├── LICENSE                     # MIT — tribe package code
├── LICENSE-TRIBE-V2           # CC-BY-NC-4.0 — TRIBE v2 neural pipeline
├── .gitignore                  # Python standard ignores + atlas/ note
├── tribe/
│   ├── __init__.py             # Package init
│   ├── __main__.py             # `python -m tribe` entry
│   ├── cli.py                  # Click CLI (analyze, backends, setup, version)
│   ├── analyze.py              # Main orchestrator
│   ├── schema.py               # ContentAnalysis, Technique, Emotion, NeuralAnalysis
│   ├── backends/
│   │   ├── base.py             # AnalysisBackend abstract class
│   │   ├── router.py           # GPU detection, backend selection
│   │   ├── classifier.py        # QCRI BERT (18-technique) + DistilRoBERTa emotion
│   │   ├── qcri_architecture.py # QCRI custom model + label mapping
│   │   └── tribe_v2.py         # TRIBE v2 pipeline + Yeo atlas interpretation
│   ├── interpretation/
│   │   ├── atlas/              # Yeo2011_7Networks_N1000 annot files (82KB each, committed)
│   │   ├── technique.py         # 18-technique emotion maps, potency, descriptions
│   │   └── neural.py           # Yeo 7-network → manipulation ratio
│   └── output/
│       ├── narrative.py         # Human-readable terminal output
│       └── json_output.py       # JSON output
└── tests/
    ├── test_schema.py           # 3 tests
    ├── test_ingestion.py        # 8 tests
    ├── test_interpretation.py   # 12 tests (incl. atlas + QCRI coverage)
    ├── test_output.py           # 6 tests
    └── test_cli.py              # 2 tests (setup stdin fix, import)
```

---

## Dependencies

```toml
dependencies = [
    "click>=8.1",
    "httpx>=0.27",           # HTTP — NOT axios
    "trafilatura>=1.12",
    "lxml_html_clean>=0.4",
    "transformers>=4.40",
    "torch>=2.2",
    "nibabel>=5.2",
    "numpy>=1.26",
    "tf-keras>=2.21",
]
```

---

## Architecture

```
Input (URL / file / stdin)
        │
        ▼
Content Ingestion
   (httpx + trafilatura for URLs)
        │
        ▼
Backend Router ── tribev2 installed + GPU? ──┐
        │                                   │
        ├─ NO ──► Classifier Backend ◄────┘
        │        QCRI BERT (18-technique, token-level)
        │        DistilRoBERTa emotion
        │        Parallel inference
        │
        └─ YES ──► TRIBE v2 Backend
                     LLaMA 3.2 + V-JEPA2 + Wav2Vec-BERT
                     Fusion transformer → 20,484 vertex activations
                     Yeo 7-network parcellation → network scores
                     Manipulation ratio = emotional / rational
        │
        ▼
Interpretation Layer
  Classifier: techniques → emotion targets → manipulation score
  TRIBE v2: network scores → manipulation ratio → interpretation
        │
        ▼
Unified ContentAnalysis output
  primary_trigger, manipulation_score, techniques, emotions, neural
        │
        ▼
Output Renderers
  Narrative (human-readable) / JSON / Quiet (single line)
```

---

## Key Technical Details

### Brain Network Mapping (Yeo 2011 7-Network Parcellation)
| Network | Role | Manipulation Signal |
|---------|------|--------------------|
| Salience | Attention capture | HIGH = threat detected |
| Default Mode | Self-referential | HIGH = made it personal |
| Limbic | Emotional processing | HIGH = emotional memory |
| Executive Control | Rational evaluation | LOW = bypassing thinking |
| Dorsal Attention | Focused attention | neutral |

**Manipulation ratio** = (Salience + Default Mode + Limbic) / (Executive Control + Dorsal Attention)

### Classifier Models
- **Technique**: `QCRI/PropagandaTechniquesAnalysis-en-BERT` — BERT-base, 110M params, 18-technique token-level detection. Custom architecture in `qcri_architecture.py`. Aggregates via max-pool logits + softmax.
- **Emotion**: `j-hartmann/emotion-english-distilroberta-base` — 7-class emotion detection
- Both run on MPS (Apple Silicon) with fallback to CPU

### QCRI Technique Labels (18)
Appeal to Authority, Appeal to Fear/Prejudice, Bandwagon, Black-and-White Fallacy, Causal Oversimplification, Doubt, Exaggeration/Minimisation, Flag-Waving, Glittering Generalities, Intentional Vagueness, Loaded Language, Misrepresentation of Someone's Position (Or Quoting), Name Calling/Labeling, Repetition, Slogans, Thought-Terminating Cliche, Transfer, Whataboutism/Red Herring.

See `tribe/interpretation/technique.py` for emotion target and potency mappings.

---

## Next Steps

### Medium Priority
1. **README demo GIF** — record actual terminal session for the repo README
2. **Score calibration** — QCRI softmax-normalized scores give similar values (3.3-4.0) for both articles. Consider adjusting the score formula to increase discrimination.

### Low Priority (TRIBE v2 requires GPU)
3. **Install tribev2** on machine with ≥10GB VRAM
4. **Test neural backend end-to-end** — verify brain activation → manipulation ratio pipeline
5. **Quantize TRIBE v2** for machines with 8GB VRAM

---

## Known Issues

1. ~~**Yeo atlas files not downloaded**~~ — DONE (2026-04-04). Files committed to repo.
2. ~~**Binary propaganda model**~~ — DONE. QCRI model (18-technique multi-label) integrated.
3. ~~**tribe setup stdin bug**~~ — DONE. Setup now uses internal text fixture.
4. ~~**README placeholders**~~ — DONE. README updated with real CLI output.
5. **Score calibration** — QCRI softmax normalization produces similar scores (3.3-4.0) for both manipulative and neutral test fixtures. May need score formula adjustment for better discrimination.
6. **Media analysis** — classifier backend returns placeholder for video/audio (TRIBE v2 handles media).

---

## Philosophy

- **Local-first**: No content sent to external servers. All analysis runs on your machine.
- **TRIBE v2 is central**: The neural prediction IS the differentiator. Lightweight classifiers are the fallback, not the product.
- **Human-first output**: Plain language by default, JSON for pipelines.
- **Honest about limits**: Classifier backend never claims neural predictions.
