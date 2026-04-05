# Paper 1: Plan

## Title

From Network Ratios to Regional Dissociation: Brain Encoding Models Detect Content Manipulation When Interpreted Through Persuasion Neuroscience

## Core Claim

Brain encoding models (TRIBE v2) can detect manipulative content from predicted fMRI activations, but only when interpreted through region-level persuasion neuroscience (vmPFC/dlPFC/TPJ dissociation), not through the popular Yeo 7-network emotional/rational ratio.

## One-Paragraph Summary

We test whether predicted brain responses from TRIBE v2 can distinguish manipulative from non-manipulative text. A straightforward approach - computing the ratio of emotional to rational network activation from the Yeo 7-network parcellation - fails completely (40% accuracy, p=0.41). We trace this failure to three neuroscientific errors: the Default Mode Network is self-referential rather than emotional, the Salience Network responds to importance regardless of manipulation, and vmPFC and dlPFC (both prefrontal) play opposite roles during persuasion but are averaged together. We propose region-level persuasion analysis based on published fMRI studies of persuasion (Falk et al. 2010, 2024), targeting vmPFC (value integration), dlPFC (critical evaluation), and TPJ (motive analysis). This approach achieves 84% accuracy (p<0.001) on the same predicted activations. The brain encoding predictions were always informative - the interpretation layer was wrong.

## Target Venue

ACL 2026 Findings (8 pages + references)
- Deadline: typically mid-January for main, rolling for Findings
- Format: ACL LaTeX template

## Contributions (what the reviewers will evaluate)

1. Negative result: empirical demonstration that network-level emotional/rational ratios fail for manipulation detection from brain encoding (with neuroscience explanation)
2. Positive result: region-level persuasion analysis achieves statistically significant separation (84%, p=0.0004)
3. Bridge: first application of persuasion neuroscience (Falk et al.) to brain encoding model interpretation

## Data We Have

| Data | File | Status |
|------|------|--------|
| 25 paired texts (manipulative + neutral) | tribe/benchmarks/datasets/paired.py | Done |
| Benchmark 001: v1 text, failed | results/benchmarks/001 | Done |
| Benchmark 002: v1 audio, failed | results/benchmarks/002 | Done |
| Benchmark 003: v2 text, 84% p=0.0004 | results/benchmarks/003 | Done |
| Benchmark 004: v2 TTS audio, inverted | results/benchmarks/004 | Done |
| Benchmark 005: v2 real audio, inverted | results/benchmarks/005 | Done |
| Raw region scores (vmPFC, dlPFC, TPJ) | modality_comparison.json | Done |

## Figures Needed

| Figure | Description | Data Source |
|--------|-------------|-------------|
| Fig 1 | Score distributions: v1 vs v2, manipulative vs neutral (violin or box plot) | Benchmarks 001 + 003 |
| Fig 2 | Region activation comparison: vmPFC, dlPFC, TPJ for manip vs neutral | Benchmark 003 raw data |
| Fig 3 | The interpretation pipeline diagram (input -> TRIBE v2 -> Destrieux regions -> persuasion signal) | New diagram |

## Tables Needed

| Table | Description |
|-------|-------------|
| Table 1 | Main results: v1 vs v2, win rate, p-value, mean diff |
| Table 2 | Per-region activation: vmPFC, dlPFC, TPJ, insula, precuneus for manip vs neutral |
| Table 3 | Persuasion signal components and their weights |

## Citations Needed (must verify all via API)

### Core citations (the neuroscience foundation)
- Falk et al. 2010, J Neurosci - "Predicting Persuasion-Induced Behavior Change from the Brain"
- Falk et al. 2024, PNAS - "Deciphering neural responses to a naturalistic persuasive message"
- Yeo et al. 2011, J Neurophysiol - "The organization of the human cerebral cortex"
- Destrieux et al. 2010, NeuroImage - "Automatic parcellation of human cortical gyri and sulci"

### Brain encoding model
- TRIBE v2 (Meta 2026) - the model paper
- eugenehp/tribev2 - the public weights

### Dual-process theory critique
- Review paper on System 1/System 2 being oversimplified (2024, Phenomenology and Cognitive Sciences)
- Review paper on dual-process in behavioral economics (2019, Review of Philosophy and Psychology)

### Default Mode Network
- DMN review showing it's self-referential, not emotional (2025, Biology MDPI)

### Salience Network
- Salience network review (2023, Frontiers in Human Neuroscience)

### Manipulation/propaganda detection (related work)
- SemEval-2020 Task 11 (Da San Martino et al.)
- MentalManip (Wang et al. ACL 2024)
- SpeechMentalManip (Chen et al. 2025)

### Amygdala
- Critical appraisal of amygdala fMRI (2024, PMC)

## Paper Structure

### Abstract (150 words)

### 1. Introduction (1 page)
- Brain encoding models are new - predict fMRI from content
- Can predicted brain responses detect manipulation?
- We test two interpretation approaches
- Contributions: negative result + diagnosis + positive result

### 2. Background (1 page)
- 2.1 Brain encoding models (TRIBE v2)
- 2.2 Yeo 7-network parcellation
- 2.3 Persuasion neuroscience (Falk et al.)
- 2.4 Content manipulation detection (SemEval, MentalManip)

### 3. Method (1.5 pages)
- 3.1 Dataset: 25 controlled pairs
- 3.2 Brain encoding pipeline (TRIBE v2 via Rust binary)
- 3.3 Interpretation v1: Yeo 7-network emotional/rational ratio
- 3.4 Interpretation v2: Region-level persuasion analysis
- 3.5 Evaluation metrics: win rate, paired t-test

### 4. Results (1 page)
- Table 1: main comparison
- Figure 1: score distributions
- Table 2: per-region activation

### 5. Why Network Ratios Fail (1 page)
- 5.1 DMN is not emotional
- 5.2 Salience detects importance, not manipulation
- 5.3 vmPFC/dlPFC cancellation in Executive Control
- Figure 2: region activations showing the dissociation

### 6. Discussion (1 page)
- Brain encoding CAN detect manipulation with right interpretation
- Limitations: 25 pairs, text only, formula weights not learned
- Future: larger datasets, audio/video, learned classifier (Path C)

### 7. Conclusion (0.5 page)

### References

## What's Missing Before We Can Write

1. Generate Figure 1 and 2 from benchmark data
2. Extract per-region stats (vmPFC, dlPFC, TPJ means for manip vs neutral) from raw data
3. Verify all citations via PapersFlow MCP
4. ACL LaTeX template

## Risks

- 25 pairs is a small dataset - reviewers will flag this
  - Mitigation: acknowledge in limitations, note statistical significance, note controlled design
- The persuasion signal weights (0.30, 0.25, 0.20, 0.15, 0.10) are hand-tuned, not learned
  - Mitigation: acknowledge, note Path C as future work, ablation analysis
- We're interpreting predicted fMRI, not actual fMRI
  - Mitigation: clear framing, cite TRIBE v2's validation against real fMRI
