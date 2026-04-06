# Planned Papers

## Paper 1: Region-Level Persuasion Analysis
**Title:** From Network Ratios to Regional Dissociation: Brain Encoding Models Detect Content Manipulation When Interpreted Through Persuasion Neuroscience

**Status:** Data complete, citations verified, plan written, ready to draft

**Core claim:** Yeo 7-network emotional/rational ratio fails (40%, p=0.41). Region-level persuasion analysis based on vmPFC/dlPFC/TPJ dissociation succeeds (84%, p=0.0004). Same predictions, different lens.

**Target:** ACL 2026 Findings (8 pages)

**Data:**
- Benchmark 001: v1 text, 40% win rate, p=0.41 (failed)
- Benchmark 003: v2 text, 84% win rate, p=0.0004 (works)
- MentalManip text (2,915 items) - running, will validate on external data

**Key citations:** Falk et al. 2010 J Neurosci, Ntoumanis et al. 2024 PNAS, Yeo et al. 2011, Destrieux et al. 2010

**Directory:** papers/paper-01-region-dissociation/

---

## Paper 2: Modality Dissociation
**Title:** Text-Dominant Fusion in Brain Encoding Models: Modality Effects in Predicted fMRI Responses to Manipulative Content

**Status:** Data partially complete, needs full audio benchmark

**Core claim:** TRIBE v2's audio pipeline is text-dominated. The fusion transformer weights transcription-derived text features more heavily than raw Wav2Vec-BERT audio features. Evidence:
- Audio manipulation detection shows inverted signal vs text (benchmarks 004, 005)
- Pure tone frequency test produces only 3 unique patterns across 16 frequencies (benchmark 006)
- Whisper transcription -> LLaMA features dominate the fusion output

**Target:** Interspeech 2026 or NeurIPS Workshop on AI Safety

**Data:**
- Benchmark 004: v2 TTS audio, inverted signal (28%, p=0.01)
- Benchmark 005: v2 real audio (SpeechMentalManip), inverted signal
- Benchmark 006: frequency response, 3 unique patterns across 16 frequencies
- SpeechMentalManip 180-file run - queued

**Key citations:** Chen et al. 2026 (SpeechMentalManip), d'Ascoli et al. 2026 (TRIBE v2), Falk et al. 2010

**Future direction (for discussion section):** Adversarial audio generation targeting specific brain activation patterns. Instead of detecting manipulation, generate audio that TRIBE v2 predicts will activate specific regions (e.g., suppress dlPFC for reduced critical evaluation, activate vmPFC for value adoption). This is essentially an optimization problem - search the acoustic space for audio that produces desired predicted brain response. Blocked by the text-dominant fusion finding - would require improved acoustic feature weighting in the fusion transformer. Ethically dual-use: therapeutic (anxiety reduction, focus enhancement) vs manipulative (persuasion amplification). Must address in ethics section.

**Directory:** papers/paper-02-modality-dissociation/ (to create)

---

## Paper 3: Learned Manipulation Detection from Predicted Cortical Activations
**Title:** (TBD) Training Classifiers on Predicted fMRI for Content Manipulation Detection

**Status:** Blocked on MentalManip activation collection (~20h GPU)

**Core claim:** A simple classifier (logistic regression or SVM) trained on raw 20,484-vertex activation vectors can detect manipulation better than hand-crafted heuristics (both network-level and region-level). Data-driven signal discovery vs neuroscience assumptions.

**Target:** NeurIPS 2026 or AAAI 2027

**Data needed:**
- Raw activations from MentalManip (2,915 items) - collecting now
- Raw activations from SemEval (371 items) - queued
- Raw activations from paired set (50 items) - done
- PCA reduction + logistic regression training

**Key question:** Does the classifier find the same vmPFC/dlPFC/TPJ pattern we identified manually, or something different? If different, that's an even more interesting finding.

**Directory:** papers/paper-03-learned-classifier/ (to create)

---

## Shared Resources

| Resource | Location | Status |
|----------|----------|--------|
| Verified BibTeX | papers/paper-01-region-dissociation/paper.bib | Done (8 entries) |
| Benchmark results 001-006 | results/benchmarks/ | Done |
| MentalManip text run | tribe/benchmarks/results/mentalmanip_scores.jsonl | Running (1070/2915) |
| SpeechMentalManip audio | tribe/benchmarks/data/speech_mentalmanip/ | Downloaded, not run |
| Paired dataset | tribe/benchmarks/datasets/paired.py | Done (25 pairs) |
| Paper writing skills | ~/.claude/skills/academic-research-skills/ | Installed |
| ML paper templates | ~/.claude/skills/AI-Research-SKILLs/ | Installed |
| PapersFlow MCP | ~/.claude.json | Installed |
