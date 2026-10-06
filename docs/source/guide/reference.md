# 13. Reference and troubleshooting

Use a task guide to choose the operation, then use the reference to check exact names, nested options, defaults and supported profiles. The downloadable schemas and keyword inventories come from the same public exports used by static consumers.

## Configuration and tag index

| Reference | Contents |
|---|---|
| {doc}`/reference/configuration` | Dataset and pipeline input forms, fields, defaults and loading |
| {doc}`/reference/pipeline_syntax` | Step forms, parameters, serialization and composition |
| {doc}`/reference/pipeline_keywords` | Lifecycle paths, nested value schemas, scopes, aliases and engine support |
| {doc}`/reference/generator_keywords` | Generator/modifier types, constraints and expansion semantics |
| {doc}`/reference/native_capability_preflight` | Per-engine executable checks and refusal reasons |
| {doc}`/reference/multimodal_execution_matrix` | Supported multimodal paths and data profiles |
| {doc}`/api/modules` | Generated Python API |
| {doc}`/reference/cli` | CLI groups, flags, inputs, outputs and destinations |

Download the <a href="../_static/keyword-registry.json">keyword inventory</a>, <a href="../_static/keyword-registry.schema.json">keyword schema</a>, <a href="../_static/tuning-summary.schema.json">tuning summary schema</a> and <a href="../_static/robustness-summary.schema.json">robustness summary schema</a>. These static files are emitted by the documentation build from public API exports. A schema validates declared structure; runtime validation additionally checks data identities, fitted state and supported profiles.

## Diagnose a refusal

| Symptom | Inspect first |
|---|---|
| Optional native runtime missing | Product, installed version, CLI/library path and Methods ABI |
| Source schema mismatch | Source IDs, feature order, dtype policy, dimensions, units and coordinates |
| Fold/group or leakage refusal | Independent-unit/group/origin relations and training scope |
| Resume contract mismatch | Data identity, parameter paths, objective, optimizer state and parent checkpoint |
| Archive integrity failure | Member inventory/hashes, outcome/model links and archive/profile version |
| Wrong prediction row order | Sample-ID alignment and named output binding; do not reorder by position |
| Unsupported calibration or browser transport | Producer/consumer artifact profile and qualified direction |
| Failed export leaves no result | Atomic destination rules, existing files and output overlap |

See {doc}`/user_guide/troubleshooting/faq`, {doc}`/user_guide/troubleshooting/dataset_troubleshooting`, {doc}`/user_guide/troubleshooting/migration` and {doc}`/migration/native_v1`. Preserve the original failed evidence when repairing configuration. Do not modify signed artifacts merely to make an unsupported historical profile load.
