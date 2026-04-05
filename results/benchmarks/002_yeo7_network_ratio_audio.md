# Benchmark 002: Yeo 7-Network Emotional/Rational Ratio (Audio)

Date: 2026-04-05
Pipeline: --text-path (TTS -> Wav2Vec-BERT audio)
Interpretation: Yeo 7-network emotional/rational ratio (v1)
Dataset: 25 controlled pairs (50 texts via TTS)
Status: FAILED

## Results

- Win rate: 11/25 (44%)
- Losses: 13, Ties: 1
- Mean manipulative: 1.02
- Mean neutral: 0.96
- Mean difference: 0.06
- t-statistic: 0.846
- p-value: 0.4062
- Significant: NO

## Interpretation

Audio-mediated pipeline also fails with the v1 interpretation layer.
The 5-pair quick test showed 80% but collapsed to 44% on full 25 pairs.
The interpretation layer, not the modality, was the bottleneck.
