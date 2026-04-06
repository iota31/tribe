# Benchmark 006: Frequency Response (Pure Tones)

Date: 2026-04-06
Pipeline: --audio-path (generated WAV files)
Interpretation: Destrieux region-level persuasion (v2)
Dataset: 16 pure sine tones (20Hz-12000Hz) + silence + white noise
Status: TEXT-DOMINATED - only 3 unique patterns

## Results

Only 3 unique activation patterns across all 18 stimuli:

| Pattern | vmPFC | dlPFC | Score | Stimuli |
|---------|-------|-------|-------|---------|
| A | -0.0162 | +0.0009 | 6.8 | 20Hz, 40Hz, 150Hz, 440Hz, 500Hz, 2kHz, 4kHz, 8kHz, 12kHz, white noise |
| B | -0.0059 | -0.0088 | 5.6 | 60Hz, 100Hz, 200Hz, 432Hz, 800Hz, 1kHz |
| C | -0.0010 | +0.0034 | 5.2 | 300Hz |
| D | -0.0124 | +0.0150 | 5.4 | silence |

## Key Finding

TRIBE v2's audio pipeline is TEXT-DOMINATED. The pipeline:
1. Audio -> Whisper transcription (produces hallucinated text from pure tones)
2. Transcription -> LLaMA features (text drives the prediction)
3. Audio -> Wav2Vec-BERT features (acoustic features are secondary)

Frequencies that produce similar Whisper hallucinations get identical brain
predictions. The raw acoustic signal is not meaningfully influencing the
fMRI prediction through the current pipeline.

## Implication

The audio pathway does NOT process raw acoustic features independently.
It's a text-mediated pipeline where Whisper transcription is the bottleneck.
This explains the inverted signal in benchmarks 004 and 005 - the model
responds to the transcribed TEXT of the dialogue, not the acoustic
properties of the speech.

For true acoustic frequency response, we would need a pipeline that
bypasses Whisper and feeds Wav2Vec-BERT features directly to the
fusion transformer.
