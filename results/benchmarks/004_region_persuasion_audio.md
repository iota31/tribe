# Benchmark 004: Region-Level Persuasion Analysis (Audio)

Date: 2026-04-06
Pipeline: --text-path (TTS -> Wav2Vec-BERT audio)
Interpretation: Destrieux region-level persuasion (v2)
Dataset: 25 controlled pairs (50 texts via TTS)
Status: INVERTED SIGNAL

## Results

- Win rate: 7/25 (28%) - INVERTED (neutral scores higher)
- Losses: 17, Ties: 1
- Mean manipulative: 4.35 +/- 0.88
- Mean neutral: 5.02 +/- 0.61
- Mean difference: -0.67
- t-statistic: -2.797
- p-value: 0.0100
- Significant: YES (but in wrong direction)

## Interpretation

The audio pipeline produces significantly different activation patterns than
the text pipeline. Scores are much higher overall (4-5 range vs 2-3 for text)
and the separation is inverted - neutral content produces higher persuasion
signal scores than manipulative content.

This suggests the TTS -> audio -> Wav2Vec-BERT pathway activates different
brain regions than the LLaMA text pathway. The persuasion signal formula
(calibrated for text) does not transfer directly to audio activations.

## Key Observation

The audio pathway may need its own calibration. The raw activation patterns
are different enough that the same region-level weights don't apply.
This is scientifically interesting - it suggests modality-specific
interpretation is needed.
