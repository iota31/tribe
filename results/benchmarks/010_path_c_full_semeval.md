# Benchmark 010: Path C on Full SemEval (reality check)

Date: 2026-04-07
Method: PCA + Ridge regression / Logistic regression on 327 SemEval articles
Status: Signal exists but weak

## Continuous Regression (propaganda density)

| PCA | Spearman rho | p-value | Pearson r |
|-----|-------------|---------|-----------|
| 5 | 0.110 | 0.046 | 0.118 |
| 10 | 0.138 | 0.012 | 0.194 |
| 20 | 0.185 | 0.0008 | 0.233 |
| 30 | 0.202 | 0.0002 | 0.253 |
| 50 | 0.200 | 0.0003 | 0.253 |

## Binary Classification (median split, all 327)

| PCA | CV AUC |
|-----|--------|
| 5 | 0.547 |
| 10 | 0.565 |
| 20 | 0.582 |
| 30 | 0.580 |

## Honest Assessment

The earlier 0.67 AUC was on 162 quartile-split items (extreme vs extreme).
On the full range with median split, AUC drops to 0.58. The signal is
statistically significant (rho=0.20, p<0.001) but weak.

The brain encoding activations contain SOME propaganda-discriminating
information but the effect size is small. Scaling to more data will
probably push AUC toward 0.65-0.70, not 0.90.

## Reality

TRIBE v2's predicted brain responses contain a weak but real signal
for propaganda content. This is not a strong detector. It is a
weak neuroscience-informed feature extractor that might be useful
in combination with other features, but on its own it performs
modestly on real-world propaganda classification.

Our 84% on paired texts was an artifact of using deliberately
extreme examples. Real-world content is subtler.
