# 11. Follow a complete recipe

A tutorial should run from a clean installed package plus explicitly declared optional runtimes. It must identify its input provenance, training/evaluation protocol, output artifacts and the profile required by the replay consumer.

| Recipe | Executable guide | What to retain |
|---|---|---|
| First regression and candidate selection | {doc}`start` and {doc}`/getting_started/quickstart` | Dataset, candidates, OOF, selected refit and export |
| Compare preprocessing and models | {doc}`/examples/user/preprocessing` and {doc}`/examples/user/models` | Comparable validation protocol and metric direction |
| Cross-validation and grouped observations | {doc}`/examples/user/cross_validation` and {doc}`/user_guide/data/aggregation` | Group/repetition/origin IDs and fold memberships |
| NIR + image + series + metadata | {doc}`/user_guide/data/methods_multimodal_u07` | Encoder state, schema, fusion model and no-FIT replay |
| Partial cohorts and late fusion | {doc}`/user_guide/data/multimodal_late_partial` | Source/target masks and held-out meta-model inputs |
| Native R/Octave host workflow | {doc}`/user_guide/data/octave_multimodal` | Host artifact manifests, native scores and fresh-process replay |
| Optimize, resume, refit and export | {doc}`/user_guide/models/native_pls_fold_hpo` and {doc}`/user_guide/models/structural_hpo` | Search contract, parent/checkpoint identity and refit state |
| Conformal intervals | {doc}`/user_guide/models/native_tuning_conformal` | Calibration cohort, calibrator and observed coverage |
| Reuse/deploy a selected model | {doc}`/examples/user/deployment` | Frozen input schema and portable/host-specific artifact profile |

The {doc}`examples index </examples/index>` connects runnable examples with guides. Synthetic datasets illustrate behavior and support deterministic qualification; they do not substitute for evidence on a real acquisition or instrument. Label that distinction in reports and charts.

A successful producer run is only half of an interlanguage tutorial. Verify the actual exported bytes and repeat prediction in the target language with fitting disabled and no access to the training workspace.
