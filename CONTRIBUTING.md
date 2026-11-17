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

nirs4all packages from different repositories depend on each other at **minor**
granularity, never more precisely: Python `>=0.4,<0.5` (`>=1.3,<2` for 1.x),
Rust `"0.4"`, npm `^0.4.0`, R `pkg (>= 0.4.0), pkg (< 0.5.0)`. The SDK declares
`dag-ml>=0.4,<0.5`, `dag-ml-data>=0.2,<0.3`, `nirs4all-io>=0.3,<0.4`,
`nirs4all-core>=0.5,<0.6`, `nirs4all-methods`/`pls4all>=1.3,<2` in
`pyproject.toml`, the `requirements*.txt` mirrors and `conda-forge/meta.yaml`.
Exact, reproducible versions live in lockfiles and install constraints
(`docker/native-constraints.txt`, the release workflows' installed cohort) and
in the release order, never in these manifests.

In return, a **patch** release of `nirs4all` is backward compatible and changes
no wire format, byte contract, persisted schema (workspace store, bundles),
`.n4a` payload semantics or shared fixture; any such change needs at least a
**minor** release. A fix needed from a specific owner patch travels through the
lockfiles and the release order, not a patch floor; native Methods libraries are
accepted by their runtime ABI check (the Methods explainer needs ABI 2.23, first
carried by Methods 1.3.18). The full policy is
`docs/contracts/release/VERSION_CONSTRAINTS.md` in `nirs4all-ecosystem`;
`tests/unit/workflows/test_publish_workflow.py` and
`tests/integration/parity/test_rt_goldens.py` check the declared ranges.

## Contribution licensing (inbound = outbound)

By submitting a contribution, you agree it is provided under the **same dual license**
as the project (open-source (AGPL/GPL/CeCILL) + commercial option), with no extra restrictions.

If your organization requires a **CLA**, please see `CLA.md` and sign as needed.

© 2025 CIRAD
