# Dataset Configuration Specification Review (Draft)

## Scope
Analysis of the draft dataset configuration specification (docs/specifications/dataset_config_specification.md) for usability, maintainability, and feasibility against current nirs4all loading logic.

## High-Level Assessment
- The draft spec is far richer (files/sources schema, column/row filters, partitions, folds, archives, multi-format support) than the current implementation, which only supports the legacy `train_x/train_y/train_group/test_x/...` keys and CSV/NumPy arrays.
- As written, configs that follow the new schema will not be executed by the current loaders and will fail or be ignored; the spec should be flagged as "not implemented" until the refactor lands.
- Implementing the proposed schema requires a new parser/loader pipeline, additional file-format handlers, and significant validation logic not present today.

## Usability vs. Current Capabilities
- **Schema acceptance**: Only legacy keys are parsed in [nirs4all/data/config_parser.py](nirs4all/data/config_parser.py#L221) and no `files`/`sources` handling exists. Users following the spec will see errors or silent no-ops.
- **Column/row selection**: The spec’s `columns.include/exclude`, ranges, regex, and `rows` conditions are unsupported; loader only has basic `y_filter`/`x_filter` hooks, with `x_filter` not implemented in [nirs4all/data/loaders/loader.py](nirs4all/data/loaders/loader.py#L256).
- **Partitioning**: The spec promises column-, percentage-, stratified-, and external-file-based partitions; current logic infers partition solely from separate train/test inputs in [nirs4all/data/config.py](nirs4all/data/config.py#L290).
- **Folds**: Draft defines multiple fold mechanisms, but no fold ingestion exists; SpectroDataset just stores folds without loader support.
- **Multi-source/linking**: Spec includes `sources` and `link` by key; implementation only allows a list of X paths with implicit row alignment (no key-based joins) in [nirs4all/data/loaders/loader.py](nirs4all/data/loaders/loader.py#L91).
- **Formats**: Spec lists Parquet, Excel, MATLAB, NumPy, tar, passworded archives, etc.; runtime supports CSV (+gz/zip) and NumPy arrays only via CSV loader in [nirs4all/data/loaders/csv_loader.py](nirs4all/data/loaders/csv_loader.py).
- **Validation**: Spec promises extensive validation; current checks are limited to basic row/column consistency and file existence.

## Maintainability Considerations
- The proposed schema adds many branches (formats, selectors, partitions, folds, linking) that will need modular handlers to stay maintainable. A monolithic loader will be fragile.
- Backward compatibility must be preserved: legacy keys should map cleanly into the new schema to avoid breaking examples and tests.
- Validation should be centralized (schema + semantic checks) to prevent duplication across loaders.
- Clear lifecycle docs and migration guidance will be needed because the spec materially changes user-facing configuration.

## Feasibility vs. Current Logic
- **Parsing**: A new parser is required to translate `files`/`sources` into internal load plans; the current parser normalizes only legacy aliases (see [nirs4all/data/config_parser.py](nirs4all/data/config_parser.py#L12) and [nirs4all/data/config_parser.py](nirs4all/data/config_parser.py#L101)).
- **Loading engine**: Need per-format loaders (Parquet, Excel, NPZ/MAT, tar/zip member selection) and column/row selection utilities. Today only CSV is implemented.
- **Linking/joining**: Requires key-based joins across files (with merge strategies) rather than positional alignment; not present today.
- **Partition/fold materialization**: Must create partition masks and fold splits from columns, percentages, stratification, and external files. Current flow assumes pre-separated train/test arrays.
- **Signal/headers**: Spec’s header prefix stripping and wavelength unit handling at load time are not implemented; only a simple `header_unit` pass-through exists.

## Recommendations
- Short term: Add a prominent "Draft – not yet implemented" note at the top of the spec and list the currently supported configuration keys to avoid user confusion.
- Refactor plan (incremental):
  1) Define a JSON Schema/Pydantic model for `files`/`sources`, partitions, and folds.
  2) Build a planner that converts the new schema into concrete load steps (per file, per role), including column/row selectors.
  3) Add format-specific loaders (CSV existing; add Parquet, Excel, NPZ/NPY, MAT) and archive member selection.
  4) Implement linking by keys (inner/left/outer), with validation of row counts and key uniqueness.
  5) Add partition/fold builders (column-, percentage-, stratified-, group-based) to populate SpectroDataset indices.
  6) Layer validation (schema + semantic) with actionable errors and tests covering each use case in the spec.
- Migration: Keep legacy keys as a compatibility layer that translates into the new schema to avoid breaking existing examples and tests.

## Conclusion
The draft specification describes a robust, flexible configuration system, but most of its surface is not yet feasible with the current codebase. Proceeding requires a dedicated refactor of parsing, loading, and validation layers; until then, the spec should be marked as aspirational to set accurate user expectations.
