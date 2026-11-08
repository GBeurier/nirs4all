# Python SDK contributor instructions

## Supported role and ownership

- The SDK supplies supported public APIs/orchestration, Python ML/DL controllers, SHAP/explain, synthesis, workspace/host integrations and expressly identified scientific references over Core.
- Existing duplicated portable algorithms migrate to their owners, not another Python shim.
- [Core](../nirs4all-core/AGENTS.md) aggregates [Methods](../nirs4all-methods/AGENTS.md), [DAG](../dag-ml/AGENTS.md), [DMD](../dag-ml-data/AGENTS.md), [Formats](../nirs4all-formats/AGENTS.md), IO and Datasets.
- Methods owns numerics/splitter algorithms; DAG execution/variant/fold/selection/scoring context, with its named generic identity-aligned conformal exception.
- Sibling links are optional navigation.
- Preserve supported published API and artifact semantics while following clean V1: obsolete lanes/compatibility alternatives disappear only after affected active callers migrate and parity is established within authorization.
- Historical explicit legacy rollback prose is not a mandate to preserve a removed lane or introduce fallback.
- Read [public interfaces](docs/source/reference/public_interfaces.md) and [native migration](docs/source/migration/native_v1.md) before execution/persistence changes.
- Discover current package/license/version from owner manifests and LICENSE/notices; never freeze old instruction release labels.
- Core owns `.n4a` container/inventory; payload owners keep semantics.
- SDK Python-host bundles use Core V1 transport; BundleLoader adapts shared verified host readers. Joblib payloads require a trusted producer and remain nonportable.
- Route by declared schema/owner/profile/operation, not suffix; a readable artifact is not executable support.
- [Development](docs/development.md) and [agent workflow](docs/AGENT_WORKFLOW.md) retain local commands/mechanics. [Pipeline keywords](docs/source/reference/pipeline_keywords.md) links its generator reference; neither is a complete inventory of all registered tokens.

- Supported public entrypoints include `run`, `predict`, `explain`, `retrain`, `session` and `generate`; preserve their advertised API/artifact semantics while routing work through owners. Explicit execution selectors remain strict; no silent legacy fallback.

- Preserve the distribution-specific licensing boundaries in LICENSE/notices: AGPL-3.0-or-later, the documented GPL-3.0/CeCILL-2.1 variants and commercial terms are not interchangeable redistribution grants.

- `@register_controller` and `OperatorController` are the public Python controller API; see [controllers](docs/source/developer/controllers.md) and the [architecture/module map](docs/source/developer/architecture.md). Examples and the controller inventory exercise them through explicit `engine="legacy"` while that rollback lane exists. Do not assume default `native`/`dag-ml` routing: those selectors fail closed on unsupported shapes. Use this registry, not a parallel dispatcher; generic capabilities go to their owners.

- [Module API](docs/source/api/module_api.md) documents `RunResult` and `NIRSPipeline`.

## Local gates and engineering

```bash
python -m pip install -e ".[dev]"
python -m pip install -r requirements-test.txt
ruff check .
mypy nirs4all
pytest tests/unit/
pytest tests/integration/
pytest tests/ --cov=nirs4all --cov-report=xml
sphinx-build -b html docs/source docs/_build/html --keep-going
python -m build --sdist --wheel --outdir dist
```

- Use the package venv, Python 3.11+, Google docstrings, explicit public types, PEP8/existing patterns, configured Ruff/import ordering/220 columns, snake_case modules/functions/keys and PascalCase classes.
- Retain the documented additional `mypy .` scope and pre-publish/examples commands in workflow.
- Tests mirror unit/integration areas, `test_*.py`/Test*/test_*; use slow/sklearn/tensorflow/torch/keras/jax/gpu/stress markers and small reusable fixtures.
- API changes require `cd examples && ./run.sh`; user-facing changes update docs/examples.
- Backends are lazy-loaded; preserve YAML list roundtrips and supported workspace serialization.
- No impossible-case defenses, one-use abstractions, hard-coded test workarounds or unrelated speculative cleanup; validate at external/user/API boundaries.
- Read files before proposing edits and again before/after edits; preserve local read/output bounds and complete rename coverage in workflow.
- CodeGraph-first supersedes grep-only navigation, not direct/type/string/dynamic-import/re-export/test coverage.
- Keep scoped imperative Conventional Commits and PR issue/context/checks; screenshots only for visualization/docs UI changes. Dependency/license changes update requirements/pyproject/notices. Never force-add private notes/native binaries.
- Local phase/cleanup/review/delegation rules apply within current user authorization; no extra commit, approval, agent, source/frontend or publication authority is created by the workflow.

## Working rules

- Read applicable local and deeper instructions before editing. Preserve dirty/concurrent work; make authorized owner fixes and propagate affected contracts/bindings/consumers. Do not reset another author's work.
- Use CodeGraph FIRST for implementation navigation only if this checkout's root already has `.codegraph/`; use its exact projectPath or run the CLI here. Never auto-index or silently substitute an ancestor index.
- Unindexed code and documentation use ordinary source navigation. Explicit source, tests and build tools are final authority. Owner links are navigation; this policy works without a parent workspace.
- All application/project content is English: UI text/subtitles/tooltips/errors, docs/comments/docstrings, test/fixture prose, reports/issues/PRs, commits and generated material. Preserve identifiers, quotations and historical/provenance meaning; converse with the maintainer in French.
- Do not commit credentials, private data, generated workspaces, caches or build outputs. Follow local licensing/fixture restrictions and report actual commands/exits/skips; a command reference grants no execution/publication permission.
