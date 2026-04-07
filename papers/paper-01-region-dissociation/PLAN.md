# Paper 1: Final Plan (Reframed after Phase 5 baseline finding)

## Title

Brain Encoding Models Underperform Bag-of-Words for Propaganda Detection: A Reality Check on TRIBE v2 for NLP Tasks

(Alternative: "tf-idf Beats Brain Encoding: A Cautionary Tale About Applying Neuroscience Models to NLP")

## Core Claim

We tested whether predicted brain responses from Meta's TRIBE v2 brain encoding model can detect propaganda in news articles. They cannot - or rather, they can, but worse than a 1990s bag-of-words baseline. Across 11 benchmarks and three interpretation strategies, we find that the brain decoder layer adds noise rather than signal: tf-idf + logistic regression outperforms TRIBE v2 by 7 AUC points (0.65 vs 0.58) and doubles the Spearman correlation with propaganda density (0.42 vs 0.20). Network-level neuroscience-informed interpretations fail entirely. Hand-tuned region-level formulas overfit to controlled data. Even data-driven learned classifiers on raw cortical activations underperform direct text features. We explain why through three neuroscientific misconceptions and one architectural finding (the audio pipeline collapses to text). We release the full benchmark suite to save other researchers from the same dead end.

## Key Result Table (the hook)

| Method | SemEval AUC | Spearman rho | Compute |
|--------|-------------|--------------|---------|
| TRIBE v2 brain encoding (region-level v2) | rho=-0.07, AUC 0.45 | -0.07 | ~25s/sample (GPU) |
| TRIBE v2 brain encoding (learned, leak-free) | 0.58 | 0.20 | ~25s/sample (GPU) |
| sentence-transformer (free) | 0.63 | 0.40 | <1ms/sample (CPU) |
| **tf-idf + logistic** | **0.65** | **0.42** | **<1ms/sample (CPU)** |

The brain encoding model loses to a bag-of-words baseline.

## Target Venue

**Primary:** ACL 2026 Findings or EMNLP 2026 Findings
- Negative result paper, but with strong baselines and reproducible code
- Cautionary tale framing fits Findings well

**Alternative:** NeurIPS Workshop on AI Safety, AI Failures workshop, or "I Can't Believe It's Not Better" workshop

## Contributions (revised, honest, 3 not 5)

1. **Definitive negative result with strong baselines:** TRIBE v2 brain encoding underperforms tf-idf for propaganda detection by 7 AUC points (0.58 vs 0.65)
2. **Diagnosis of three neuroscience misconceptions** that cause network-level interpretations to fail (DMN not emotional, Salience detects importance, vmPFC/dlPFC cancellation)
3. **Architectural finding:** TRIBE v2's audio pipeline collapses to text - 16 distinct pure tones produce only 3 unique brain activation patterns, indicating Whisper transcription dominates over Wav2Vec-BERT acoustic features in the fusion transformer

## What Changed from Phase 3 Draft

The Phase 5 peer review revealed:
1. We were missing the LLaMA/text-baseline comparison
2. PCA was leaking outside the CV loop (now fixed)
3. The 84% on paired data was on the same data the formula was tuned on (circular)
4. The "weak positive result" was actually NEGATIVE when compared to baselines

The paper is now structured around the baseline finding, not around defending Strategy 3.

## Data We Have (11 benchmarks)

| # | Benchmark | Result | Status |
|---|-----------|--------|--------|
| 001 | v1 ratio paired text | 40% win, p=0.41 | Failed |
| 002 | v1 ratio paired audio | 44% win, p=0.41 | Failed |
| 003 | v2 region paired text | 84% win (CIRCULAR - tuned on this data) | In-sample fit |
| 004 | v2 region paired audio | 28% win (inverted) | Inverted |
| 005 | v2 region real audio | Inverted | Inverted |
| 006 | Pure tones | 3 patterns/16 freqs | Audio pipeline collapse |
| 007 | MentalManip | AUC 0.47 | Wrong dataset type |
| 008 | SemEval (formula) | rho=-0.07 | Failed |
| 009 | Path C paired (leak-free) | CV AUC 0.91 | Real but n=50 |
| 010 | Path C SemEval (leak-free) | AUC 0.58, rho 0.20 | Weak |
| **011** | **NLP baselines on SemEval** | **tf-idf AUC 0.65, rho 0.42** | **Beats brain encoding** |

## Paper Structure (Revised)

### Abstract (150 words)
Lead with: "tf-idf beats TRIBE v2 brain encoding for propaganda detection by 7 AUC points."

### 1. Introduction (1 page)
- Brain encoding models are exciting; people want to use them for NLP
- Natural question: do predicted brain responses help with content classification?
- Empirical answer: no, simple text features work better
- Three contributions: negative result with baselines, neuroscience diagnosis, audio architecture finding

### 2. Background (1 page)
- 2.1 TRIBE v2
- 2.2 Yeo 7-network parcellation
- 2.3 Persuasion neuroscience (Falk, Ntoumanis)
- 2.4 SemEval propaganda task

### 3. Method (1 page)
- 3.1 Pipeline: TRIBE v2 via tribev2-infer
- 3.2 Three interpretation strategies (network ratio, hand-tuned region, learned classifier)
- 3.3 NLP baselines: tf-idf, sentence-transformer
- 3.4 Datasets and evaluation (with leak-free CV protocol)

### 4. Results (2 pages)
- 4.1 Network ratios fail (Benchmarks 001-002)
- 4.2 Hand-tuned region formula overfits to paired data, fails on SemEval (003 + 008)
- 4.3 Learned classifier shows weak signal but underperforms baselines (009-011)
- 4.4 The audio pipeline collapse (006)
- Tables: full benchmark table, baseline comparison table

### 5. Why Brain Encoding Loses (1.5 pages)
- 5.1 The three neuroscience misconceptions (DMN, Salience, vmPFC/dlPFC)
- 5.2 Audio pipeline is text-mediated (Whisper -> LLaMA dominance)
- 5.3 Brain decoder is a lossy text encoder

### 6. Discussion (1 page)
- 6.1 What this paper shows and doesn't show
- 6.2 When brain encoding might still be useful (multimodal, perception studies)
- 6.3 Honest limitations
- 6.4 Path forward: don't bolt brain encoding onto NLP tasks

### 7. Conclusion (0.5 page)
- Brain encoding != better classification
- The decoder adds noise
- Use simple baselines first
- Open benchmark suite released

### Limitations
- Single brain encoding model (TRIBE v2)
- English text only
- One propaganda dataset (SemEval)
- Linear classifiers only (deep models might find different patterns)

### Ethics
- Caution against deploying weak detectors with false neuroscience authority
- Brain encoding language can lend unwarranted credibility to flawed systems

## Phase 4 (re-run integrity check)
- All 12 benchmarks (including new 011)
- All claims must match files
- All citations stay valid (no new ones needed for baseline experiment)

## Phase 5 (re-run review)
- Address all 5 reviewers' concerns
- Critical: now have baselines, leak-free CV, honest Benchmark 003 framing
- Statistical rigor: add bootstrap CIs and multiple-comparison correction

## What's Still TODO
1. Redraft paper.tex with new framing
2. Re-run integrity check on new draft
3. Re-run peer review on new draft
4. Generate figures from benchmark data
5. Polish pass with Fabric improve_academic_writing

## Risks
- Reviewers might still want fine-tuned BERT comparison (we used logistic, not full transformer)
- Reviewers might want a third dataset to confirm the negative result
- "Negative result with baselines" is a stronger paper but still a tougher sell than "we found a new method"

## The Story (revised)

We thought brain encoding might help detect manipulation. We tried three interpretation strategies. They got progressively more sophisticated. None of them beat tf-idf. The brain decoder layer is a lossy compression of text features. We explain why through neuroscience misconceptions and an audio architecture finding. We release everything so others don't waste GPU hours like we did.
