# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-04-05

### Changed
- Unified pipeline: TRIBE v2 Rust is now the only analysis backend
- CPU fallback: runs on any machine without GPU (slower, same results)
- Removed Classifier backend (QCRI BERT + DistilRoBERTa) — different model, different results
- Removed `--backend` CLI flag — single pipeline, auto GPU/CPU detection
- Slimmed dependencies: removed transformers and tf-keras (no longer needed)

### Removed
- `tribe/backends/classifier.py` — NLP propaganda technique classifier
- `tribe/backends/qcri_architecture.py` — custom QCRI model architecture
- `tribe/interpretation/technique.py` — technique-to-emotion mapping

## [0.2.0] - 2026-04-05

### Added
- TRIBE v2 Rust backend with Metal GPU acceleration on Apple Silicon
- Classifier backend with QCRI BERT (18 techniques) + DistilRoBERTa (8 emotions)
- Web demo server (`tribe serve`) with one-click examples and brain network visualization
- CLI commands: `analyze`, `serve`, `backends`, `version`
- Yeo 2011 7-network brain mapping and interpretation
- Dual licensing: GPL-3.0 (package code) + CC-BY-NC-4.0 (TRIBE v2 neural components)
- Custom SVG architecture and brain network diagrams
- GitHub Actions CI, issue templates, and community health files

### Fixed
- Yeo atlas `.annot` files now tracked in git instead of downloaded at runtime

## [0.1.0] - 2026-04-04

### Added
- Initial release: content manipulation detection engine
- Text ingestion from files and URLs
