# Benchmark 009: Path C Learned Classifier

Date: 2026-04-07
Pipeline: --prompt (LLaMA text embeddings)
Method: PCA(n_components) + Logistic Regression on raw 20,484-vertex activations
Status: SIGNAL CONFIRMED IN RAW ACTIVATIONS

## Three Experiments

### Experiment 1: Train on Paired (50), Test within Paired
- Method: 5-fold cross-validation on 25 manipulative + 25 neutral
- CV AUC: 0.912 +/- 0.109
- Leave-one-out accuracy: 88.0% (44/50)
- Result: Signal exists in paired data

### Experiment 2: Train on Paired (50), Test on SemEval (327)
- Training: 50 paired samples
- Testing: 327 SemEval articles
- Quartile AUC: 0.41 (inverted direction)
- Spearman rho: -0.11
- Result: Signal does NOT transfer across content types

### Experiment 3: Train AND Test within SemEval (162 samples, quartile split)
- Samples: 81 high-propaganda vs 81 low-propaganda (top/bottom quartiles)
- PCA=5 (95% var): CV AUC 0.601
- PCA=10 (98% var): CV AUC 0.639
- PCA=20 (100% var): CV AUC 0.670
- Spearman rho with continuous density (PCA=20): 0.355 (p<1e-6)
- Result: Signal EXISTS in raw activations for propaganda detection

## Key Finding

The brain encoding model's raw activations DO contain propaganda-discriminating
information (AUC 0.67 within SemEval, statistically significant correlation
with propaganda density rho=0.35, p<1e-6).

BUT the signal is content-specific. A classifier trained on our 25 paired
texts (extreme fear appeals vs factual reporting) does NOT transfer to
SemEval propaganda articles. The models learn different patterns.

## Implication

Three conclusions:

1. The hand-tuned persuasion formula (v2 interpretation) was wrong - the
   signal is in the activations but needs to be learned, not hand-coded.

2. The 84% on paired texts was real but learned the wrong signal.

3. Path C works when trained on the target domain. SemEval-trained
   classifier achieves AUC 0.67 on SemEval. To build a general propaganda
   detector, we need large-scale training data (NELA-GT: 1.78M articles).
