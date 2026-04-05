# Contributing to Tribe

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/iota31/tribe.git
cd tribe
pip install -e ".[dev]"

# Run tests
python3 -m pytest tests/ -q
```

## Code Style

- PEP 8 compliant
- Type hints required for all public functions
- Docstrings for all modules and classes

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `python3 -m pytest tests/ -q`
5. Commit with a clear message: `git commit -m "Add feature: ..."`
6. Push and open a PR

## Adding New Backends

1. Inherit from `tribe.backends.base.AnalysisBackend`
2. Implement `analyze_text()` and `analyze_media()`
3. Add to `tribe/backends/router.py` in `get_backend()`
4. Add tests in `tests/`
5. Update this README

## Issues

Found a bug or want a feature? Open an issue at https://github.com/iota31/tribe/issues
