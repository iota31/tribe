# Benchmarks

## What TRIBE Measures

TRIBE v2 predicts how the human brain responds to content. It uses Meta's brain encoding model to predict fMRI activation at 20,484 cortical vertices, then maps these to Yeo's 7 functional brain networks:

- **Emotional networks:** Salience, Default Mode, Limbic
- **Rational networks:** Executive Control, Dorsal Attention
- **Sensory networks:** Visual, Somatomotor

The **manipulation ratio** = emotional activation / rational activation. Content that hijacks emotional processing while suppressing analytical thinking produces a high ratio.

## Evaluation Philosophy

TRIBE is not a propaganda classifier -- it doesn't output technique labels like "Loaded Language" or "Appeal to Fear." It predicts brain responses.

We evaluate TRIBE by asking: **Does content that human experts label as manipulative produce higher predicted emotional brain activation than non-manipulative content?**

This is a separation test, not a classification test. The key metrics are:

| Metric | What It Measures |
|--------|-----------------|
| **AUC-ROC** | Can manipulation_score separate manipulative from non-manipulative? (threshold-independent) |
| **Cohen's d** | Effect size -- how far apart are the score distributions? |
| **Spearman rho** | Does manipulation_score correlate with propaganda density? |
| **Win Rate** | In controlled pairs, how often does the manipulative version score higher? |

## Datasets

### SemEval-2020 Task 11 -- Propaganda Detection
- **Source:** [SemEval-2020 Shared Task](https://aclanthology.org/2020.semeval-1.186/)
- **Size:** 371 news articles, 6,129 propaganda spans, 14 techniques
- **Labels:** Span-level propaganda technique annotations
- **Our use:** Compute propaganda density per article, correlate with TRIBE manipulation_score

### MentalManip (ACL 2024) -- Manipulation in Conversations
- **Source:** [Wang et al., ACL 2024](https://aclanthology.org/2024.acl-long.206/)
- **Size:** 2,915 movie dialogues (consensus labels)
- **Labels:** Binary (manipulative / non-manipulative)
- **Our use:** Binary separation test -- AUC-ROC, Cohen's d

### Controlled Pairs (Internal)
- **Size:** 25 topic-matched pairs (50 texts)
- **Labels:** Same topic, manipulative vs neutral framing
- **Our use:** Paired comparison -- win rate, paired t-test

## Reproduction

```bash
# Install benchmark dependencies
pip install -e ".[bench]"

# Download datasets
tribe bench download

# Run all benchmarks (requires tribev2-infer binary)
tribe bench run

# Generate visualizations
tribe bench visualize

# View results
tribe bench results
```

## Limitations

1. **Brain encoding is not brain measurement.** TRIBE predicts how an average brain would respond. It doesn't scan actual brains.
2. **Correlation is not causation.** High emotional activation doesn't necessarily mean manipulation -- genuine emergencies also trigger emotional networks.
3. **Dataset bias.** SemEval articles are English news from specific sources. MentalManip uses movie dialogues. Results may not generalize to all content types.
4. **Model limitations.** TRIBE v2 was trained on 451.6 hours of fMRI data from 25 subjects. The "average brain" it models may not represent all populations.

## Citation

If you use TRIBE's benchmark suite in your research:

```bibtex
@software{tribe2026,
  title={Tribe: Neural Content Analysis},
  author={Tushar},
  year={2026},
  url={https://github.com/iota31/tribe}
}
```
