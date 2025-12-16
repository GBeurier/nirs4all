# Contributing to nirs4all

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork and create a virtual environment
3. Install development dependencies: `pip install -e ".[dev,test]"`
4. Run tests: `pytest tests/`

## Development Guidelines

### Code Style

- Follow **PEP 8** guidelines
- Use **Google Style docstrings**
- Target **Python 3.11+**
- Write modular, testable code

### Testing

- Add unit tests for new functionality
- Run the full test suite before submitting: `pytest tests/`
- Run examples as integration tests: `./examples/run.ps1`

### Documentation

- Update relevant docs when adding features
- API changes should update `docs/api/`
- User-facing features need `docs/reference/` updates

## Developer Guides

- [Artifacts Developer Guide](docs/explanations/artifacts_developer_guide.md) - Extending the artifacts system
- [Storage API Reference](docs/api/storage.md) - Artifact storage API

## Extending the Artifacts System

To add new artifact types, serialization formats, or customize the storage:

1. See the [Artifacts Developer Guide](docs/explanations/artifacts_developer_guide.md)
2. Follow the patterns in `nirs4all/pipeline/storage/artifacts/`
3. Add corresponding unit tests in `tests/unit/pipeline/storage/artifacts/`
4. Update the [Manifest Specification](docs/specifications/manifest.md) if schema changes

## Contribution licensing (inbound = outbound)

By submitting a contribution, you agree it is provided under the **same dual license**
as the project (open-source (AGPL/GPL/CeCILL) + commercial option), with no extra restrictions.

If your organization requires a **CLA**, please see `CLA.md` and sign as needed.

© 2025 CIRAD
