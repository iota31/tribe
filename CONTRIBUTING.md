# Contributing to Tribe

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/iota31/tribe.git
cd tribe
pip install -e ".[dev,lint]"

# Run tests
python3 -m pytest tests/ -q
```

## Code Style

- PEP 8 compliant
- Type hints required for all public functions
- Docstrings for all modules and classes

## Commit Messages

We use [conventional commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding or updating tests
- `refactor:` code change that neither fixes a bug nor adds a feature

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `python3 -m pytest tests/ -q`
5. Commit with a clear message: `git commit -m "feat: add feature description"`
6. Push and open a PR

PRs require passing CI (ruff lint + pytest). A maintainer will review within a few days.

## First Contribution?

Look for issues labeled [`good first issue`](https://github.com/iota31/tribe/labels/good%20first%20issue). If you have questions, open a [Discussion](https://github.com/iota31/tribe/discussions).

## Issues

Found a bug or want a feature? Open an issue at https://github.com/iota31/tribe/issues
