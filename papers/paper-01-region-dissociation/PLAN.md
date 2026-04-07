# Paper 1: Revised Plan

## Title (updated)

Brain Encoding Features for Content Manipulation Detection: A Reality Check on Interpretation Strategies

(Alternative: "From Network Ratios to Learned Classifiers: An Honest Evaluation of Brain Encoding for Propaganda Detection")

## Core Claim (revised)

Brain encoding models (TRIBE v2) produce predicted cortical activations that contain a weak but statistically significant signal for content manipulation detection. Three interpretation strategies were tested: (1) Yeo 7-network emotional/rational ratios fail entirely (AUC ~0.50), (2) hand-tuned region-level persuasion analysis based on Falk et al. works on controlled paired texts (84% win rate) but does not generalize to real propaganda news articles (AUC 0.45), and (3) learned classifiers on raw 20,484-vertex activations achieve a modest but significant effect (SemEval quartile AUC 0.67, full median split AUC 0.58, Spearman rho 0.20 p<0.001). The brain encoding approach shows promise as a feature extractor but is not yet a strong standalone propaganda detector. We release our benchmarks, code, and negative results to save other researchers from the same dead ends.

## One-Paragraph Summary

We systematically evaluated whether TRIBE v2, Meta's brain encoding model, can detect content manipulation from its predicted fMRI activations. We tested three interpretation strategies across five datasets: the intuitive "emotional vs rational network ratio" from Yeo 7-network parcellation (failed empirically: AUC 0.40-0.44, p=0.41), a neuroscience-informed region-level analysis targeting vmPFC/dlPFC/TPJ based on the persuasion neuroscience literature (worked on controlled paired texts: 84% win rate, p=0.0004, but failed to generalize: SemEval rho=-0.07, MentalManip AUC 0.47), and a learned classifier trained directly on raw 20,484-vertex activations (SemEval quartile AUC 0.67, full median split AUC 0.58, continuous Spearman rho 0.20 p<0.001). We trace the failures to three neuroscience misconceptions about network-level interpretation and to content-domain specificity of learned classifiers. TRIBE v2's activations contain a real but weak signal for propaganda content that may be useful as a feature in combination with other approaches, but the predicted fMRI responses alone are not sufficient for strong propaganda detection. We release the full benchmark suite, 10 recorded results, and reproducible code.

## Target Venue (revised)

**Primary:** ACL 2026 Findings or EMNLP 2026 Findings (8 pages)
- Negative result papers welcomed at Findings
- Intersection of NLP + neuroscience fits well
- Focus on methodology and honest reporting

**Alternative:** NeurIPS 2026 Workshop on AI Safety or NeurIPS Workshop on ML for Social Good
- Shorter (4 pages) - less work
- Workshop audience appreciates negative results
- Faster turnaround

## Contributions (honest)

1. **Negative result 1:** Network-level emotional/rational ratios fail (AUC 0.40-0.44)
2. **Negative result 2:** Hand-tuned region-level formulas overfit to controlled data - 84% on paired texts collapses to AUC 0.45 on SemEval
3. **Positive result:** Learned classifiers on raw 20,484-vertex activations detect propaganda content (SemEval AUC 0.67 quartile, rho 0.20 p<0.001 continuous)
4. **Scientific analysis:** Three specific neuroscience misconceptions that cause network-level approaches to fail (DMN not emotional, Salience detects importance not manipulation, vmPFC/dlPFC cancellation in Executive Control)
5. **Open infrastructure:** Full benchmark suite with 10 documented results, crash-resilient runner, reproducible Python package

## Data We Have

| # | Benchmark | Dataset | Method | Result |
|---|-----------|---------|--------|--------|
| 001 | v1 ratio text | 25 paired | Yeo emotional/rational | 40% win, p=0.41 |
| 002 | v1 ratio audio | 25 paired | Yeo emotional/rational | 44% win, p=0.41 |
| 003 | v2 regions text | 25 paired | Falk et al. persuasion formula | 84% win, p=0.0004 |
| 004 | v2 regions audio | 25 paired | Same formula, TTS audio | 28% inverted, p=0.01 |
| 005 | v2 regions real audio | SpeechMentalManip (10) | Same formula | Inverted |
| 006 | Frequency response | Pure tones | Same formula | 3 unique patterns / 16 freqs |
| 007 | MentalManip v2 | 2,915 dialogues | Same formula | AUC 0.47 |
| 008 | SemEval v2 | 327 articles | Same formula | rho -0.07 |
| 009 | Path C paired | 50 items | Learned classifier | CV AUC 0.91 |
| 010 | Path C SemEval | 327 articles | Learned classifier | CV AUC 0.58-0.67 |

## Figures Needed

| Figure | Description | Data Source |
|--------|-------------|-------------|
| Fig 1 | Methodology diagram: TRIBE v2 pipeline + 3 interpretation strategies | New diagram |
| Fig 2 | Results comparison: bar chart of AUC across all methods and datasets | Benchmarks 001-010 |
| Fig 3 | Score distributions: paired data (v1 vs v2) showing both work on this data | Benchmarks 001, 003 |
| Fig 4 | Generalization failure: hand-tuned formula on paired vs SemEval | Benchmarks 003, 008 |
| Fig 5 | PCA variance explained + learned classifier performance | Benchmark 009, 010 |

## Tables Needed

| Table | Description |
|-------|-------------|
| Table 1 | All 10 benchmark results in one table |
| Table 2 | Why network ratios fail: the three neuroscience misconceptions |
| Table 3 | Path C classifier performance across PCA dimensions |

## Citations (verified)

All 8 BibTeX entries already in `paper.bib`:
- Falk et al. 2010 (J Neurosci) - vmPFC predicts persuasion
- Ntoumanis et al. 2024 (PNAS) - DMN in resistance not persuasion
- Yeo et al. 2011 - 7-network parcellation
- Destrieux et al. 2010 - cortical parcellation
- Da San Martino et al. 2020 - SemEval propaganda
- Wang et al. 2024 - MentalManip
- Chen et al. 2026 - SpeechMentalManip
- d'Ascoli et al. 2026 - TRIBE v2

## Paper Structure (8 pages)

### Abstract (150 words)

### 1. Introduction (1 page)
- Brain encoding models predict fMRI from content
- Natural question: can predicted brain responses detect manipulation?
- We systematically test three interpretation strategies
- Contributions: negative results + positive result + benchmark suite

### 2. Background (1 page)
- 2.1 TRIBE v2 brain encoding model
- 2.2 Yeo 7-network parcellation (standard approach)
- 2.3 Persuasion neuroscience: Falk et al., vmPFC/dlPFC dissociation
- 2.4 Content manipulation datasets: paired, SemEval, MentalManip

### 3. Method (1.5 pages)
- 3.1 Pipeline: TRIBE v2 via Rust binary (20,484 vertices, 100 timesteps)
- 3.2 Interpretation strategy 1: Yeo 7-network ratio
- 3.3 Interpretation strategy 2: Region-level persuasion (Destrieux atlas)
- 3.4 Interpretation strategy 3: Learned classifier (PCA + logistic regression)
- 3.5 Datasets and evaluation metrics

### 4. Results (2 pages)
- 4.1 Strategy 1 fails (Benchmarks 001-002)
- 4.2 Strategy 2 works on paired data but not elsewhere (003-008)
- 4.3 Strategy 3 shows weak but significant signal (009-010)
- Tables 1-3, Figures 2-5

### 5. Analysis: Why Strategies 1 and 2 Fail (1.5 pages)
- 5.1 DMN is not emotional (Falk 2024 PNAS)
- 5.2 Salience Network detects importance, not manipulation
- 5.3 vmPFC/dlPFC cancellation at network level
- 5.4 Hand-tuned formulas overfit to controlled data

### 6. Discussion (1 page)
- 6.1 What the signal represents
- 6.2 Why it's weak (text-mediated audio pipeline, modality effects)
- 6.3 Path forward: larger training data, multimodal input, hybrid features
- 6.4 Honest limitations

### 7. Conclusion (0.5 page)

### References

## What's Ready to Write Today

- All benchmark data complete
- 8 citations verified and in BibTeX
- Plan and structure defined
- Code and results open-source at github.com/iota31/tribe
- Benchmark suite reproducible

## What's Still Running

- Benchmark 011: Qbias scale-up (3000 samples, ~21 hours)
- If Qbias gets AUC > 0.70, strengthens the positive result
- If Qbias stays around 0.58, confirms the ceiling

## The Honest Story

We started with a hypothesis (brain encoding detects manipulation via emotional/rational ratio).
We tested it and it failed.
We dug into the neuroscience and built a better hypothesis (region-level persuasion).
It worked on our controlled data but failed to generalize.
We abandoned hand-tuned approaches and let data speak (learned classifier).
Weak but significant signal exists (AUC 0.58-0.67 on real data).
The predicted fMRI contains information about propaganda content but not enough for strong detection alone.
We release everything so others don't repeat our mistakes.

This is a methodology contribution paper: here's what works, here's what doesn't, here's why, here's the open benchmark suite.
