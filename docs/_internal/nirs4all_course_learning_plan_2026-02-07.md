# nirs4all Course Learning Plan (High-Level)

**Date**: 2026-02-07  
**Status**: Draft  
**Audience**: New to intermediate nirs4all users (researchers, ML practitioners, pipeline engineers)

---

## 1. Course Goal

Build practical autonomy with nirs4all in a clear order:
1. Understand the mental model and data semantics.
2. Build and evaluate robust pipelines on real spectroscopy workflows.
3. Progress to advanced orchestration (branching, generators, sessions, internals).
4. Ship a reproducible end-to-end project (capstone).

---

## 2. Design Principles (Theory vs Practice)

- Start with **conceptual anchors** so API choices make sense.
- Move quickly to **hands-on scripts** in `examples/`.
- Keep each module at roughly **30-40% theory / 60-70% practice**.
- Use one capstone thread throughout: same dataset, progressively improved pipeline.
- Introduce internals only after baseline user-level fluency.

Recommended overall split:
- **Theory**: 35%
- **Practice**: 65%

---

## 3. Ideal Learning Order (8 Modules)

### Module 0 - Setup and Orientation

- Duration: 2-3 hours
- Theory/Practice: 20/80
- Concepts first:
  - Installation and environment checks
  - What `run()`, `predict()`, `explain()`, `retrain()` do
  - How examples are organized (`user`, `developer`, `reference`)
- Practice:
  - Run a smoke test of local environment
  - Execute first script end-to-end
- Examples:
  - `examples/user/01_getting_started/U01_hello_world.py`

### Module 1 - Core Mental Model (Data, Workflow, Runtime)

- Duration: 4-5 hours
- Theory/Practice: 50/50
- Concepts first:
  - Three-layer model: Data, Workflow, Runtime
  - `SpectroDataset`, pipeline expansion, controller routing, workspace persistence
  - Why OOF predictions and reproducibility are central
- Practice:
  - Re-run hello world and inspect outputs/artifacts
  - Explain each pipeline step in terms of controller behavior
- Reading:
  - `docs/source/onboarding/mental_models.md`
  - `docs/source/onboarding/data_workflow.md`
  - `docs/source/onboarding/pipeline_workflow.md`
- Examples:
  - `examples/user/01_getting_started/U01_hello_world.py`
  - `examples/user/01_getting_started/U02_basic_regression.py`

### Module 2 - Data Handling and Leakage-Safe Evaluation

- Duration: 6-8 hours
- Theory/Practice: 35/65
- Concepts first:
  - Input formats and `DatasetConfigs`
  - Partitions, folds, group splitting, repetition semantics
  - Multi-source data and target alignment
- Practice:
  - Build a data ingestion notebook/script from your own dataset
  - Validate split strategy for leakage safety
- Examples (in this order):
  - `examples/user/02_data_handling/U01_flexible_inputs.py`
  - `examples/user/02_data_handling/U02_multi_datasets.py`
  - `examples/user/02_data_handling/U03_multi_source.py`
  - `examples/user/05_cross_validation/U01_cv_strategies.py`
  - `examples/user/05_cross_validation/U02_group_splitting.py`

### Module 3 - Spectral Preprocessing Strategy

- Duration: 6-8 hours
- Theory/Practice: 30/70
- Concepts first:
  - Scatter correction, smoothing, derivatives, signal conversion
  - Preprocessing as hypothesis testing, not cookbook stacking
  - Controlled variant generation (`_or_`, `feature_augmentation`)
- Practice:
  - Compare 3-5 preprocessing strategies with identical CV
  - Build a short model card: best preprocessing choice + why
- Examples:
  - `examples/user/03_preprocessing/U01_preprocessing_basics.py`
  - `examples/user/03_preprocessing/U02_feature_augmentation.py`
  - `examples/user/03_preprocessing/U04_signal_conversion.py`

### Module 4 - Modeling, Selection, and Diagnostics

- Duration: 8-10 hours
- Theory/Practice: 30/70
- Concepts first:
  - Baseline-first model strategy
  - Hyperparameter sweeps and fair comparison
  - Ranking metrics and result interpretation
- Practice:
  - Benchmark multiple models on one preprocessing baseline
  - Add tuning and compare stability across folds
- Examples:
  - `examples/user/04_models/U01_multi_model.py`
  - `examples/user/04_models/U02_hyperparameter_tuning.py`
  - `examples/user/05_cross_validation/U04_aggregation.py`
  - `examples/user/01_getting_started/U04_visualization.py`

### Module 5 - Explainability and Deployment Readiness

- Duration: 6-8 hours
- Theory/Practice: 25/75
- Concepts first:
  - What can and cannot be concluded from SHAP in spectroscopy
  - Export bundles and reproducible inference workflow
  - Workspace-centered lifecycle management
- Practice:
  - Generate SHAP reports for top model variants
  - Export and reload a model, run predictions on held-out/new data
- Examples:
  - `examples/user/07_explainability/U01_shap_basics.py`
  - `examples/user/07_explainability/U02_shap_sklearn.py`
  - `examples/user/06_deployment/U01_save_load_predict.py`
  - `examples/user/06_deployment/U02_export_bundle.py`
  - `examples/user/06_deployment/U03_workspace_management.py`

### Module 6 - Advanced Pipeline Composition

- Duration: 8-12 hours
- Theory/Practice: 30/70
- Concepts first:
  - Branching semantics: duplication, separation, merge
  - Generator algebra: `_or_`, `_range_`, nested/n-ary patterns
  - When to use stacking/meta-model patterns
- Practice:
  - Build one branching pipeline and one generator-driven sweep
  - Compare computational cost and interpretability tradeoffs
- Examples (recommended order):
  - `examples/developer/01_advanced_pipelines/D01_branching_basics.py`
  - `examples/developer/01_advanced_pipelines/D03_merge_basics.py`
  - `examples/developer/01_advanced_pipelines/D05_meta_stacking.py`
  - `examples/developer/02_generators/D01_generator_syntax.py`
  - `examples/developer/02_generators/D04_nested_generators.py`

### Module 7 - Internals and Performance Engineering

- Duration: 8-12 hours
- Theory/Practice: 40/60
- Concepts first:
  - Session lifecycle and controller architecture
  - Artifact and cache behavior
  - Performance profiling and reproducibility constraints
- Practice:
  - Use sessions across train/predict/retrain flow
  - Profile a large variant sweep and optimize cache strategy
- Examples:
  - `examples/developer/06_internals/D01_session_workflow.py`
  - `examples/developer/06_internals/D02_custom_controllers.py`
  - `examples/developer/06_internals/D03_cache_performance.py`

---

## 4. Capstone (Final 1-2 Weeks)

Deliver a complete spectroscopy project:

1. Problem framing and dataset definition
2. Leakage-safe split policy (with group/repetition handling if needed)
3. Preprocessing + model sweep with clear ranking criteria
4. Explainability report (SHAP + domain interpretation notes)
5. Exported deployment artifact and reproducible prediction script
6. Brief technical report (assumptions, risks, next experiments)

Capstone success criteria:
- Reproducible run in a clean environment
- Transparent model selection process
- Evidence of leakage-safe evaluation
- Operational handoff (`.n4a` export + prediction workflow)

---

## 5. Suggested Weekly Pacing

- **Week 1**: Modules 0-2 (foundation + safe data workflow)
- **Week 2**: Modules 3-4 (preprocessing + modeling)
- **Week 3**: Module 5 + start capstone
- **Week 4**: Modules 6-7 + finalize capstone

Fast track (intensive):
- 8-10 full days by compressing each module into half-day blocks

---

## 6. Assessment Checkpoints

- Checkpoint A (after Module 2):
  - Learner can load data in multiple formats and explain split safety.
- Checkpoint B (after Module 4):
  - Learner can run fair model comparisons and defend best candidate.
- Checkpoint C (after Module 5):
  - Learner can export/reload model and explain predictions.
- Checkpoint D (after Module 7):
  - Learner can reason about branching/generator design and runtime cost.

---

## 7. Minimal vs Full Curriculum

- Minimal curriculum (for applied users): Modules 0-5
  - Outcome: Strong practitioner ready for production workflows.
- Full curriculum (for power users/maintainers): Modules 0-7
  - Outcome: Advanced pipeline engineer with internal architecture fluency.

---

## 8. Notes for Instructors

- Keep the same dataset context across modules whenever possible.
- Require learners to justify each preprocessing/model choice with evidence.
- Delay internals until learners can already build and deploy a solid baseline.
- Prefer “one concept -> one runnable script -> one reflection” loops.
