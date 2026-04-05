# Benchmark 001: Yeo 7-Network Emotional/Rational Ratio (Text)

Date: 2026-04-05
Pipeline: --prompt (LLaMA text embeddings)
Interpretation: Yeo 7-network emotional/rational ratio (v1)
Dataset: 25 controlled pairs (50 texts)
Status: FAILED

## Results

- Win rate: 10/25 (40%)
- Losses: 13, Ties: 2
- Mean manipulative: 1.25 +/- 0.64
- Mean neutral: 1.14 +/- 0.36
- Mean difference: 0.11
- t-statistic: 0.830
- p-value: 0.4145
- Significant: NO

## Interpretation

The emotional/rational network ratio does not separate manipulative from
neutral content. All scores compressed into 0.2-3.4 range. 96% of
manipulation ratios were below 1.0 (rational networks always dominate
for text input).

## Root Cause

The "emotional vs rational" framing is scientifically wrong:
- Default Mode Network is self-referential, not emotional
- Salience Network detects importance, not manipulation
- vmPFC and dlPFC are both in Executive Control but do opposite things
