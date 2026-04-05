# Benchmark 005: Region-Level Persuasion Analysis (Real Audio, Quick Test)

Date: 2026-04-06
Pipeline: --audio-path (real MP3 files from SpeechMentalManip)
Interpretation: Destrieux region-level persuasion (v2)
Dataset: 10 files from SpeechMentalManip (5 manipulative, 5 neutral)
Status: INVERTED SIGNAL (same as TTS audio)

## Results (10 files only - preliminary)

- 5 manipulative: mean score 1.5 (range 0.8-2.9)
- 5 neutral: mean score 3.6 (range 1.8-4.8)
- Direction: INVERTED (neutral scores higher)

## Key Observation

Both TTS audio (benchmark 004) and real audio (this benchmark) show
inverted signals compared to text (benchmark 003). This is consistent
and suggests the audio pathway produces fundamentally different
activation patterns in the persuasion-relevant regions.

The persuasion signal formula (vmPFC/dlPFC/TPJ weights) was calibrated
from text neuroscience literature. Audio may need:
1. Different regional weights
2. Different normalization
3. A modality-specific interpretation layer

This is Paper 3 material (modality effects).

## Processing Time

Real audio files took 58-220 seconds each depending on audio duration.
Much longer than text (~25s). Total for 10 files: ~18 minutes.
