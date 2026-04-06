# Benchmark 007: MentalManip Full Text (Region-Level Persuasion v2)

Date: 2026-04-06
Pipeline: --prompt (LLaMA text embeddings)
Interpretation: Destrieux region-level persuasion (v2)
Dataset: MentalManip consensus (2,915 movie dialogues)
Status: INVERTED - wrong dataset type for this model

## Results

- Total: 2,915 items (2,016 manipulative, 899 neutral)
- AUC-ROC: 0.4687 (below random)
- Mean manipulative: 2.78 +/- 1.15
- Mean neutral: 2.88 +/- 1.15
- Mean difference: -0.10 (inverted)
- t-statistic: -2.213
- p-value: 0.027 (significant but wrong direction)

## Analysis

MentalManip contains movie dialogue interpersonal manipulation - subtle,
conversational, context-dependent (gaslighting, guilt-tripping, shaming).

TRIBE v2 was trained on fMRI from people consuming media content (movies,
podcasts). It predicts brain responses to CONTENT, not to social dynamics
within conversations.

Our 25 paired texts worked (84%) because they mimic engineered media
manipulation: fear appeals, loaded language, urgency, us-vs-them framing.

## Key Finding

Brain encoding models detect engineered media manipulation but not
interpersonal dialogue manipulation. These are fundamentally different
cognitive processes:
- Media manipulation: persuasion techniques in authored content
- Interpersonal manipulation: social dynamics between individuals

This is a dataset-model mismatch, not a model failure.
Paper-worthy finding for the discussion section.
