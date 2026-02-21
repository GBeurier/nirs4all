# nirs4all Library Backlog Suggestions

After reviewing the current `backlog.md` and the library's architecture, here are suggestions for reorganizing, adding, and removing items to ensure a smoother path to `v1.0.0`.

## Overall Assessment & Key Recommendations

1. **Shift-Left on Core Refactoring**: The current backlog places massive architectural changes (Pipeline as DAG, Pipeline as single transformer, Optuna integration) in `v0.9.x`, *after* adding many new features in `v0.8.x`. This is risky. Building features on an architecture that is about to be heavily refactored leads to wasted effort. **Recommendation**: Move core engine refactoring to `v0.9.0` and delay non-essential features to `v0.10.0` or `v1.x`.
2. **Scope Management for v0.8.x**: `v0.8.x` is currently a "catch-all" milestone containing everything from basic operators to HuggingFace integration and complex model fusion. **Recommendation**: Focus `v0.8.x` strictly on NIRS-specific operators, basic modeling, and visualization. Push advanced ML (HuggingFace, complex stacking) to later releases.
3. **Documentation & DevEx**: While docs are mentioned in `v0.8.0`, a library aiming for `v1.0.0` needs a dedicated track for tutorials, a gallery of examples, and API reference completeness.
4. **MLOps & Tracking**: Basic experiment tracking (e.g., MLflow) is currently in `v2.x+`. Given the complexity of NIRS pipelines (preprocessing sweeps), lightweight tracking should be prioritized earlier (e.g., `v0.9.x` or `v1.0.0`).

---

## Proposed Revised Backlog

### v0.8.0: Release Baseline (Current Focus)
*Keep as is. Focus on stabilizing the current API for the webapp.*
- [x] `[Docs]` Updated docs for the release scope.
- [x] `[Predictions]` Simplify and verify save behavior.
- [x] `[Loader]` Read scientific expressions in CSV.
- [x] `[MB-PLS]` Multi-source and multi-preprocessing support.
- [x] `[Branching]` Clarify preprocessing-to-source vs preprocessing-to-branch behavior.
- [-] `[Design]` Define all services.
- [-] `[SERVICE FUNCTIONS]` Provide easy service functions.

### v0.8.x: NIRS Core Features & Operators
*Focus on expanding the core NIRS capabilities (preprocessing, basic models, visualization) without changing the underlying engine.*
- **Operators & Preprocessing**
  - [-] `[Operators]` Add NorrisWilliams, Whittaker, BandEnergies, FiniteDiffCentral.
  - [x] `[PLS]` Implement variable selection methods (CARS, MC-UVE).
  - [-] `[CSV]` Authorize vertical index in CSV.
- **Modeling & Analysis**
  - [-] `[Metrics]` Add custom losses; manage metrics at global/pipeline/model levels.
  - [-] `[Analysis]` Add t-SNE.
  - [-] `[Clustering]` Add clustering controllers.
- **Visualization**
  - [-] `[Charts]` Aggregate by metadata columns; convert standard indexes to enums.
  - [ ] `[Chart_Controller]` Migrate individual chart controllers into operators.
- **Cleanup**
  - [ ] `[Dummy_Controller]` Remove totally and manage exceptions.

### v0.9.x: Engine Refactoring & Optimization
*Focus on architectural improvements before freezing the API for v1.0.*
- **Pipeline Engine (The DAG Refactor)**
  - [-] `[DAG]` Pipeline as DAG.
  - [ ] `[Pipeline]` Rework as a single transformer (fit/transform/predict).
  - [-] `[Runner]` Verify design logic of execution sequence/history; use cache by default.
  - [ ] `[Observers]` Replace copied analysis data with observers.
- **Optimization**
  - [ ] `[Optuna]` Modularize god class.
  - [ ] `[Pipeline + Optuna]` Treat pipeline as Optuna trial; preprocessing becomes choice parameter.
  - [ ] `[Training]` Add prediction cache for stack sweeps retraining identical pipelines.
- **Deep Learning Foundations**
  - [-] `[transformerMixin]` Implement in PyTorch for full differentiation.
  - [-] `[FCK-PLS]` Full torch model.

### v0.10.x: Advanced ML & MLOps (New Milestone)
*Bridge the gap between core features and v1.0 readiness.*
- **Advanced Modeling**
  - [-] `[Transfer]` Partial layers retraining or partial retrain on new data.
  - [-] `[Stacking]` Stacking from prediction files directly.
  - [ ] `[Fusion]` Mid fusion with multi-head models & Late fusion.
- **MLOps & Tracking**
  - [ ] `[MLflow]` Minimal MLflow integration (params, metrics, artifacts). *(Moved from v2.x)*
  - [-] `[Transfer]` Automate model transfer across machines.
- **Explainability**
  - [-] `[SHAP]` Verify SHAP for TF/Torch/JAX; fix imports and numpy compatibility.

### v1.0.0: Stabilization, Docs & Deploy Gate
*No new features. Focus on hardening, documentation, and distribution.*
- **Hardening**
  - [-] `[GLOBAL REVIEW]` v1.0 signatures freeze; complete tests and production coverage.
  - [-] `[Tests]` Review and clean tests.
  - [-] `[Profiling]` Code optimization and performance improvements.
- **Documentation**
  - [ ] `[Tutorials]` Create a comprehensive gallery of examples (e.g., using `sphinx-gallery`). *(New)*
- **Packaging**
  - [ ] `[Docker]` Provide Docker image and add build/actions.
  - [ ] `[Conda]` Provide Conda package and add build/actions.
  - [ ] `[SERV/CLIENT]` Clustered computation + productization service doc.

### v1.x: Post v1 Features
*Ecosystem integrations and specialized features.*
- [ ] `[onnx]` ONNX export.
- [ ] `[WanDB]` Weights & Biases compatibility.
- [ ] `[HuggingFace]` Evaluate/implement Hugging Face controller. *(Moved from v0.8.x)*
- [-] `[Customizable_Feature_Source]` Refactor dataset for customizable feature source (images, lidar).

### v2.x+: Ecosystem & Research
*Keep as is (Statsmodels, Time series, Advanced DL tooling, etc.).*
