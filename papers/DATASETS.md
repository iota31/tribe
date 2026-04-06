# Datasets for TRIBE Benchmarking

## Lesson Learned

MentalManip (movie dialogues) was the wrong dataset. AUC 0.468 - worse than random.
Movie dialogues = interpersonal manipulation (subtle, context-dependent).
TRIBE v2 was trained on media consumption. It detects engineered media manipulation.

Our 25 paired texts worked (84%, p=0.0004) because they mimic media manipulation:
fear appeals, loaded language, urgency, us-vs-them framing.

## Priority 1: Propaganda-Labeled News (engineered manipulation)

### SemEval-2020 Task 11 (RUNNING)
- 536 news articles, 18 propaganda techniques, span-level
- Zenodo: https://zenodo.org/records/3952415
- Gold standard for propaganda detection
- Running now after MentalManip completes

### NELA-GT-2022 (TODO - massive scale)
- 1.78M articles from 361 outlets
- Source-level MBFC reliability labels (reliable/mixed/unreliable)
- Harvard Dataverse: https://doi.org/10.7910/DVN/CHMUYZ
- Sample 1000 reliable + 1000 unreliable for our benchmark

### BABE (TODO - expert bias labels)
- 3,700 sentences, expert-labeled bias at sentence level
- Kaggle: https://www.kaggle.com/datasets/timospinde/babe-media-bias-annotations-by-experts
- High quality, balanced across topics

### SemEval-2023 Task 3 (TODO - 23 techniques, multilingual)
- 23 persuasion techniques, paragraph-level
- 9 languages including English
- Topics: COVID, climate, Russia-Ukraine, elections
- https://propaganda.math.unipd.it/semeval2023task3/

## Priority 2: Political Speech

### 2020 US Campaign Speeches (TODO)
- 1,056 speeches from Trump, Pence, Biden, Harris
- CC BY-NC 4.0
- No manipulation labels but clean transcripts
- Could annotate with SemEval technique taxonomy
- https://github.com/ichalkiad/datadescriptor_uselections2020

### LIAR (TODO)
- 12,836 PolitiFact statements, 6-level truthfulness
- https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- Short political claims - quick to benchmark

## Priority 3: Clickbait / Engagement Manipulation

### Webis-Clickbait-17 (TODO)
- 38,517 posts, 4-point clickbait scale
- https://zenodo.org/records/5530410
- Engineered engagement manipulation

## Priority 4: Advertising Persuasion

### Ye & Kovashka Ads Dataset (TODO)
- 64,832 image ads + 3,477 video ads
- Persuasion strategy labels, effectiveness ratings
- https://people.cs.pitt.edu/~kovashka/ads.html

## Completed Benchmarks

| # | Dataset | Type | Result | Status |
|---|---------|------|--------|--------|
| 001 | 25 Paired (v1 ratio) | Text | 40%, p=0.41 | Failed |
| 002 | 25 Paired (v1 ratio) | TTS Audio | 44%, p=0.41 | Failed |
| 003 | 25 Paired (v2 regions) | Text | 84%, p=0.0004 | Success |
| 004 | 25 Paired (v2 regions) | TTS Audio | 28%, p=0.01 | Inverted |
| 005 | SpeechMentalManip (v2) | Real Audio | ~20% (10 files) | Inverted |
| 006 | Pure Tones (v2) | Audio | 3 patterns / 16 freqs | Text-dominated |
| 007 | MentalManip (v2) | Text | AUC 0.468 | Wrong dataset type |
| 008 | SemEval-2020 (v2) | Text | Running... | TBD |
