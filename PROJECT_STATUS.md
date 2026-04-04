# Tribe — Project Status

**Last updated:** 2026-04-03
**Build status:** Working — 26/26 tests passing, CLI fully functional
**Classifier backend:** Working on Apple Silicon MPS
**TRIBE v2 backend:** Stub complete, awaiting GPU hardware

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
```

### Live Test Results
- Manipulative article (tests/fixtures/manipulative_article.txt): **3.7/10**
- Neutral article (tests/fixtures/neutral_article.txt): **2.0/10**
- First run on MPS: ~3.9s (models cached after first load)

---

## Project Structure

```
tribe/
├── pyproject.toml              # Package config, dependencies
├── README.md                   # User-facing docs
├── DESIGN.md                   # Full product design spec
├── tribe/
│   ├── __init__.py             # Package init
│   ├── __main__.py             # `python -m tribe` entry
│   ├── cli.py                  # Click CLI (analyze, backends, setup, version)
│   ├── analyze.py              # Main orchestrator
│   ├── schema.py               # ContentAnalysis, Technique, Emotion, NeuralAnalysis
│   ├── ingestion/
│   │   ├── url.py              # URL → article text (httpx + trafilatura)
│   │   ├── file.py             # Local file / stdin reading
│   │   └── media.py           # Video/audio file detection
│   ├── backends/
│   │   ├── base.py            # AnalysisBackend abstract class
│   │   ├── router.py          # GPU detection, backend selection
│   │   ├── classifier.py       # DistilBERT + DistilRoBERTa on MPS/CPU
│   │   └── tribe_v2.py        # TRIBE v2 pipeline + Yeo atlas interpretation
│   ├── interpretation/
│   │   ├── technique.py        # SemEval technique → emotion mapping, scoring
│   │   └── neural.py           # Yeo 7-network → manipulation ratio
│   └── output/
│       ├── narrative.py        # Human-readable terminal output
│       └── json_output.py      # JSON output
└── tests/
    ├── test_schema.py          # 3 tests
    ├── test_ingestion.py       # 8 tests
    ├── test_interpretation.py  # 11 tests
    ├── test_output.py          # 6 tests
    └── fixtures/
        ├── manipulative_article.txt
        ├── neutral_article.txt
        └── mixed_article.txt
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
Backend Router ── MPS/CUDA available? ──┐
        │                              │
        ├─ NO ──► Classifier Backend ◄┘
        │        DistilBERT propaganda
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
- **Propaganda**: `IDA-SERICS/PropagandaDetection` — DistilBERT, 67M params, binary propaganda detection
- **Emotion**: `j-hartmann/emotion-english-distilroberta-base` — 7-class emotion detection
- Both run on MPS (Apple Silicon) with fallback to CPU

### Missing: QCRI Technique-Level Model
The current propaganda model is **binary** (propaganda vs not_propaganda). To get per-technique breakdown (fear appeal, loaded language, etc.), need to add:
```
QCRI/PropagandaTechniquesAnalysis-en-BERT (BERT-base, 110M params, 18 techniques)
```

---

## Next Steps

### High Priority
1. **Download Yeo atlas files** — `lh.Yeo2011_7Networks_N1000.annot` and `rh.Yeo2011_7Networks_N1000.annot` need to be in `tribe/interpretation/atlas/`
   - Download from: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011

2. **Add QCRI technique-level model** — swap or supplement PropagandaDetection with the 18-technique BERT model for detailed breakdown

3. **Clean up `tribe setup`** — currently the self-test in `setup` command has a stdin bug

### Medium Priority
4. **GitHub repo** — push to GitHub with proper LICENSE (MIT for our code, CC-BY-NC for TRIBE v2)

5. **README screenshots** — real output examples, not placeholder

### Low Priority (TRIBE v2 requires GPU)
6. **Install tribev2** on machine with ≥10GB VRAM
7. **Test neural backend end-to-end** — verify brain activation → manipulation ratio pipeline
8. **Quantize TRIBE v2** for machines with 8GB VRAM

---

## Known Issues

1. **`tribe setup` self-test reads stdin** — the self-test tries to read stdin when called without piped input, which fails. Needs a text fixture instead.
2. **Binary propaganda model** — current model detects "propaganda vs not propaganda" but doesn't break down *which* technique. QCRI model needed for technique-level analysis.
3. **Yeo atlas files not downloaded** — the `atlas/` directory is empty. Neural interpreter will fail until files are present.
4. **Media analysis** — classifier backend returns placeholder for video/audio (TRIBE v2 handles media).

---

## Philosophy

- **Local-first**: No content sent to external servers. All analysis runs on your machine.
- **TRIBE v2 is central**: The neural prediction IS the differentiator. Lightweight classifiers are the fallback, not the product.
- **Human-first output**: Plain language by default, JSON for pipelines.
- **Honest about limits**: Classifier backend never claims neural predictions.
