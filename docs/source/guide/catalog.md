# 9. Discover methods and operators

Choose an operator by its role in the workflow: preprocessing, feature selection, splitting, filtering, augmentation, prediction, optimization or diagnosis. Then check the host/runtime, parameter types and artifact profile for that operator.

| Role | Reference | Questions to answer |
|---|---|---|
| Spectral preprocessing | {doc}`/reference/transforms` | Required axis, wavelength spacing, learned state and numerical convention |
| Regression/classification | {doc}`/reference/models` | Task, target shape, scale assumptions and serialization |
| Native Methods catalog | [Methods catalog](https://github.com/GBeurier/nirs4all-methods/tree/main/catalog) | Exact native ID, parameters, supported binding and ABI |
| Splitters | {doc}`/reference/splitters` | Unit of independence, stratification, groups and determinism |
| Filters/selection | {doc}`/reference/filters` | Fit scope, row/feature removal and leakage policy |
| Augmentation | {doc}`/reference/augmentations` | Training-only scope, origin identities and stochastic seed |
| Search | {doc}`/reference/generator_keywords` | Expansion or adaptive optimization, constraints and resume |
| Diagnostics/intervals | {doc}`evaluation` | Complete predictor versus branch, calibration evidence and audit purpose |

The {doc}`operator catalog </reference/operator_catalog>` and {doc}`node reference </reference/nodes/index>` provide categorized discovery. They are not a guarantee that every operator can execute in every language. A Methods binding exposing a kernel does not automatically expose it through a nirs4all task facade.

Detailed parsing, numerical and orchestration references belong to their upstream projects: [Formats](https://github.com/GBeurier/nirs4all-formats), [IO](https://github.com/GBeurier/nirs4all-io), [Methods](https://github.com/GBeurier/nirs4all-methods), [DAG-ML](https://github.com/GBeurier/dag-ml) and [DAG-ML-Data](https://github.com/GBeurier/dag-ml-data). This guide explains how those components participate in a complete user workflow.
