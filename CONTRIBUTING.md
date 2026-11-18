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

## Cross-repository version constraints

Versions are `X.Y.Z`. A **patch** (`Z`) release carries fixes and purely
additive, backward-compatible features: existing outputs, wire formats,
persisted schemas (workspace store, bundles), `.n4a` payload semantics and
shared fixtures are unchanged. A **minor** (`Y`) release carries any
compatibility break: a changed or removed API signature, a wire/byte/schema/
fixture change, renamed ids or changed numerical results of existing
operations. `X` is the product version. The package that bumps classifies its
change.

nirs4all packages from different repositories accept every patch of one tested
series: `>=X.Y[.Z],<X.(Y+1)` (for example `nirs4all-methods>=1.4,<1.5`; Rust
`">=1.4, <1.5"`, npm `~1.4.0`; for 0.x `>=0.4,<0.5`, Rust `"0.4"`, npm
`^0.4.0`). A patch floor (`>=1.4.1,<1.5`) is allowed when a fix or feature from
that patch is required. Moving to a new minor series is an explicit, tested
decision. The SDK declares its owner ranges in `pyproject.toml`, the
`requirements*.txt` mirrors and `conda-forge/meta.yaml`; ranges written before
this rule are aligned at the next normal release. Exact, reproducible versions
live only in lockfiles and install constraints (`docker/native-constraints.txt`,
the release workflows' installed cohort). Native Methods libraries are accepted
by their runtime ABI check. The full policy is
[`docs/contracts/release/VERSION_CONSTRAINTS.md`](https://github.com/GBeurier/nirs4all-ecosystem/blob/main/docs/contracts/release/VERSION_CONSTRAINTS.md) in
`nirs4all-ecosystem`; `tests/unit/workflows/test_publish_workflow.py` and
`tests/integration/parity/test_rt_goldens.py` check the declared ranges.

## Contribution licensing (inbound = outbound)

By submitting a contribution, you agree it is provided under the **same dual license**
as the project (open-source (AGPL/GPL/CeCILL) + commercial option), with no extra restrictions.

If your organization requires a **CLA**, please see `CLA.md` and sign as needed.

© 2025 CIRAD
