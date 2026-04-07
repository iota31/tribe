# Benchmark 011: NLP Baselines on SemEval (Reviewer-Required)

Date: 2026-04-07
Method: Standard NLP feature extraction + classifier
Dataset: SemEval-2020 (327 articles)
Status: BASELINES BEAT BRAIN ENCODING

## Setup

Same SemEval median-split classification task as Benchmark 010.
Three feature extraction methods compared:

| Method | Features | Dim |
|--------|----------|-----|
| TRIBE v2 brain encoding | Predicted fMRI activation | 20,484 |
| sentence-transformer (all-MiniLM-L6-v2) | Sentence embeddings | 384 |
| tf-idf + bigrams | Bag of words | 5,000 |

All methods: 5-fold leak-free CV with PCA inside the fold (where applicable),
LogisticRegression(C=1.0).

## Results - Binary Classification (Median Split AUC)

| Method | k=10 | k=20 | k=50 | Raw |
|--------|------|------|------|-----|
| TRIBE v2 | 0.574 | 0.579 | 0.578 | - |
| sentence-transformer | 0.580 | 0.595 | 0.614 | 0.631 |
| **tf-idf** | - | - | - | **0.649** |

## Results - Continuous Regression (Spearman rho with propaganda density)

| Method | rho | p-value |
|--------|-----|---------|
| TRIBE v2 (k=20) | 0.189 | 6e-4 |
| sentence-transformer (k=50) | 0.362 | <1e-6 |
| sentence-transformer (k=100) | 0.404 | <1e-6 |
| **tf-idf** | **0.423** | **<1e-6** |

## Critical Finding

A simple bag-of-words tf-idf baseline outperforms TRIBE v2 brain encoding
features on this task by:
- 7 AUC points (0.65 vs 0.58)
- 2x in Spearman correlation (0.42 vs 0.20)

The brain encoding step is adding negative value. Whatever signal TRIBE v2
captures is a subset of what plain text features already contain, and
the brain decoder is degrading it.

## Implication for Paper 1

The original Strategy 3 framing "weak but significant signal in brain
encoding activations" must be revised to:

"Brain encoding features underperform standard NLP baselines for
propaganda detection. The brain decoder layer adds noise rather than
signal. We were measuring text features all along."

This is a stronger negative result and makes the paper's main contribution
clearer: don't use brain encoding models for content classification when
direct text features are available.
