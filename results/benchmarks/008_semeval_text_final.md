# Benchmark 008: SemEval-2020 Propaganda (Region-Level Persuasion v2)

Date: 2026-04-07
Pipeline: --prompt (LLaMA text embeddings)
Interpretation: Destrieux region-level persuasion (v2)
Dataset: SemEval-2020 Task 11 (327 articles with propaganda density)
Status: NO CORRELATION

## Results

- Total: 327 articles (of 371 - some failed to process)
- Spearman rho: -0.0687 (p=0.215)
- Pearson r: -0.0807 (p=0.146)
- Quartile AUC-ROC: 0.454
- Mean high-propaganda: 3.45 +/- 1.16
- Mean low-propaganda: 3.59 +/- 1.00
- Quartile t-stat: -0.801 (p=0.424)

## Analysis

The hand-tuned persuasion signal formula does not correlate with
propaganda density in real news articles. Same as MentalManip (007).

The 84% result on our 25 controlled pairs (003) was specific to
that dataset's style - deliberately extreme manipulative vs neutral.
Real-world propaganda is subtler and the hand-tuned weights don't
capture it.

## Implication

Path C (learned classifier) is essential. The neuroscience direction
(vmPFC/dlPFC/TPJ) may be correct, but the hand-tuned formula
(0.30/0.25/0.20/0.15/0.10 weights) overfits to our paired texts.
A classifier trained on the raw 20,484-vertex activations should
learn the actual signal from data.
