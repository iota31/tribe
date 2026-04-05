# Benchmark 003: Region-Level Persuasion Analysis (Text)

Date: 2026-04-05
Pipeline: --prompt (LLaMA text embeddings)
Interpretation: Destrieux region-level persuasion (v2)
Dataset: 25 controlled pairs (50 texts)
Status: SUCCESS

## Results

- Win rate: 21/25 (84%)
- Losses: 4, Ties: 0
- Mean manipulative: 3.12 +/- 1.06
- Mean neutral: 2.25 +/- 0.51
- Mean difference: 0.86
- t-statistic: 4.132
- p-value: 0.0004
- Significant: YES (p < 0.001)

## Interpretation

Region-level persuasion analysis (vmPFC, dlPFC, TPJ) produces statistically
significant separation between manipulative and neutral content. Same TRIBE v2
predictions as benchmark 001, different interpretation layer.

## Method

Based on Falk et al. (2010 J Neurosci, 2024 PNAS):
- vmPFC activation = value integration (person adopts message)
- dlPFC activation = critical evaluation (person counterargues)
- TPJ activation = motive analysis (person questions intent)
- Persuasion signal = high vmPFC + low dlPFC + low TPJ

Uses Destrieux cortical parcellation (75 regions) mapped to fsaverage5.

## Key Finding

The fMRI predictions were always fine. The v1 interpretation layer
(Yeo 7-network emotional/rational ratio) was reading them through
the wrong neuroscience lens. The v2 layer (region-level persuasion
based on published neuroscience) extracts the signal.
