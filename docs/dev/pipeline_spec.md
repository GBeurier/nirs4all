### 1  – Objectives of the Pipeline API

| Goal                                                                                                           | Rationale                                                              |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Declarative configuration** – allow a full ML workflow to be expressed as JSON / YAML or a Python list/dict. | Keeps experiments reproducible and human-readable.                     |
| **Dynamic dispatch of heterogeneous steps** (scalers, models, branching, augmentation, etc.).                  | One engine drives every operator type discovered at runtime.           |
| **Context-aware execution** (branch ids, dataset views).                                                       | Each step can decide what slice of data it touches.                    |
| **Serialization of configs *and* live objects**.                                                               | Enables hot-reloading, hyper-parameter search, interactive notebooks.  |
| **Execution history & bundling**.                                                                              | Record, package, and ship a whole run with fitted artefacts.           |

---

### 2  – Current State

* **Modules in place**

  * `config.py` handles config ingestion/normalisation and runtime-instance embedding.&#x20;
  * `serialization.py` converts any Python object ↔️ JSON-friendly blob.&#x20;
  * `operation.py` chooses a controller for each step.&#x20;
  * `runner.py` walks the (implicit) pipeline, manages branching & parallelism, prints progress.&#x20;
  * `history.py` can track runs and bundle artefacts – but calls are commented out in the runner, so nothing is recorded.&#x20;
  * `pipeline.py` re-applies a *tree* of fitted objects for inference.&#x20;

* **Strengths** – modular, flexible, already supports branching and parallel sections.

* **Gaps** – no explicit DAG object, history disabled, fitted operators stored only as a flat dict keyed by step\_id, not by real topology; re-execution must start from config, not from a saved binary pipeline.

---

### 3  – Objectives of the Upgrade

1. **Build a true runtime DAG** that reflects the *actual* execution order after every dynamic re-queue.
2. **Persist fitted operators per node** with meaningful IDs (human-readable + stable UUID).
3. **Support dual entry points**:

   * *Config ▶ Run* (current flow).
   * *Binary DAG ▶ Replay / Predict* (new).
4. **Keep API surface familiar** – users of `PipelineRunner` and existing configs do not break.
5. **Lay ground for richer observability** (history wiring, structured logging, metrics).

---

### 4  – Phased Road-map

| Phase                        | Scope                                                                                                                                                       | File-level work (⇢ new signatures)                                                                                                                                                                             | Responsibility    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| **0. Bootstrap**             | a) Turn history back on. b) Replace all `print` with `logging`.                                                                                             | `runner.py` – uncomment `history.*` lines, inject `logger`.                                                                                                                                                    | Core maintainer   |
| **1. DAG Recording**         | Capture nodes & edges while the existing runner executes.                                                                                                   | **New `dag/state.py`**<br>`class NodeState(dataclass)`<br>`class PipelineDAG:`<br> `add_node(id: str, state: NodeState)`<br> `add_edge(parent: str, child: str)`<br> `save(folder: str)` / `load(folder: str)` | Core maintainer   |
|                              |                                                                                                                                                             | `runner.py` – factor common logic into `BaseRunner`; add subclass `DagRecordingRunner(BaseRunner)` with:<br>`run_step(..., parent_id=None) -> node_id` <br>and internal counter for id generation.            |                   |
|                              |                                                                                                                                                             | `operation` controllers – at the end of `execute`, write `context["_last_fitted"] = fitted_op`.                                                                                                                | Controller owners |
| **2. Persistence Layer**     | Write one pickle per node + `graph.json`; bundle them.                                                                                                      | `dag/state.py` – use `cloudpickle`.                                                                                                                                                                            | Core maintainer   |
|                              |                                                                                                                                                             | `history.py` – add `create_bundle_dag(dag: PipelineDAG, path)` to place DAG artefacts inside the same zip used today.                                                                                          | Core maintainer   |
| **3. Dag Runner for Replay** | Topologically apply fitted ops on new data.                                                                                                                 | **New `dag/replay.py`**<br>`class DagRunner:`<br> `def predict(self, dag: PipelineDAG, ds: SpectroDataset) -> dict`                                                                                            | ML infra          |
|                              |                                                                                                                                                             | Optional helper `dag_to_tree(dag) -> PipelineTree` to keep `FittedPipeline` backward compatible.                                                                                                               | ML infra          |
| **4. CLI & UX**              | `cli.py` with two commands: <br>`train CONFIG -o model.zip` (calls `DagRecordingRunner`) <br>`predict model.zip data.csv -o preds.csv` (calls `DagRunner`). | Dev-tools team                                                                                                                                                                                                 |                   |
| **5. Refactor & Clean-up**   | a) Move branch/context helpers to a smaller mixin.<br>b) Deprecate the old *tree* saving once DAG pipeline is stable.                                       | Incremental PRs                                                                                                                                                                                                | All               |
| **6. Docs & Examples**       | Sphinx page “Saving & Reloading a DAG Pipeline” + notebook demo.                                                                                            | Tech writing                                                                                                                                                                                                   |                   |
| **7. Observability**         | Integrate OpenTelemetry spans inside `DagRecordingRunner` and `DagRunner`.                                                                                  | Dev-ops                                                                                                                                                                                                        |                   |

---

### 5  – Logging / CI-CD / Tests & Misc

* **Logging**:

  * Declare a single `logger = logging.getLogger("nirs4all.pipeline")` inside `__init__.py`; propagate child loggers (`config`, `runner`, `dag`, …).
  * Provide a YAML or env-driven logging config (level, JSON formatter, file handler).

* **CI-CD**:

  * GitHub Actions matrix: `python{3.9,3.10,3.11}` → `pytest -sv`, `ruff`, `mypy`.
  * Build artefact `wheel` + `sdist`; trigger `twine check`.
  * Optional job: run the tutorial notebook with `papermill` to ensure DAG save/load works end-to-end.

* **Tests** (pytest):

  | Layer                | Key cases                                                   |
  | -------------------- | ----------------------------------------------------------- |
  | `dag/state`          | Node add/edge add/save/load idempotency.                    |
  | `DagRecordingRunner` | Tiny branching pipeline ⇒ DAG has correct 5 nodes, 4 edges. |
  | `DagRunner.predict`  | Re-uses scaler → model, predictions match baseline.         |
  | Regression           | Old `FittedPipeline` still loads previous *tree* bundles.   |

* **Quality Gates**: ≥ 90 % branch coverage, `ruff` + `mypy --strict` must pass.

* **Packaging**: add `pyproject.toml` with `version = "0.5.0"`; declare `networkx`, `cloudpickle` in `install_requires`.

---

#### Executive Summary

The upgrade keeps your current “configuration-driven, operation-centric” engine intact while **making the runtime structure explicit**. A `PipelineDAG` captured transparently by `DagRecordingRunner` is stored as JSON + pickles, zipped alongside history. A lightweight `DagRunner` re-executes it on fresh data. Incremental phases minimise risk; most code churn is isolated to the new `dag/` package and a small mix-in in `runner.py`. Once logging, tests, and CI gates are in place, the framework will be production-ready, fully reproducible, and friendlier for downstream consumers who just want to “load & predict”.

Below is a **forward-looking “wish list”**—the extra capabilities I would add if I were building a *complete* NIRS-oriented ML/DL pipeline library on top of the dataset layer you have already implemented .  The items are grouped by theme so you can pick the ones that matter most to your roadmap.

---

## 1 · Domain-specific Operators & Blocks

| Area                                       | What to add                                                                              | Why it matters for NIRS                                                                                                                                         |
| ------------------------------------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Spectral pre-processing**                | Savitzky–Golay derivatives, SNV, MSC, detrending, baseline correction, wavelet denoising | These are the chemometrics “classics”; giving them first-class `PipelineOperation`s makes configs shorter and prevents every project from re-implementing them. |
| **Spectral feature selection**             | Interval PLS (iPLS), moving-window PCA, genetic algorithms, VIP-based pruning            | High-dimensional (1024–4096 λ) spectra need robust variable reduction before DL.                                                                                |
| **Calibration transfer & standardisation** | DS (Direct Standardisation), PDS, piece-wise direct standardisation                      | Lets a model trained on Instrument A score data from Instrument B without refitting.                                                                            |
| **Augmentation for DL**                    | Random wavelength jitter, Gaussian noise injection, synthetic scattering                 | Helps CNN/Transformer models generalise to real-world acquisition noise.                                                                                        |
| **Outlier & drift detection**              | Hotelling T² / Q-residuals, Mahalanobis distance, COSMO                                  | Production monitoring for lab/field instruments.                                                                                                                |
| **Interpretability**                       | Spectral SHAP values, wavelength-importance heatmaps, gradient-saliency export           | Chemists expect to *see* which λ drove the prediction.                                                                                                          |

*Implementation hint*: expose these as lightweight, stateless `Controller`s so they slot straight into the DAG without boiler-plate.

---

## 2 · Model Zoo & Training Helpers

1. **Built-in chemometric models**

   * PLS, SIMPLS, PCR with cross-validation baked in.
2. **1-D DL templates**

   * “SpectraCNN”, 1-D ResNet, 1-D Transformer, and CLS token-style cross-segment attention.
   * Provide `fit/transform/predict` wrappers so they look like scikit-learn.
3. **Hybrid pipelines** (chemometric preproc → DL head)

   * Make it trivial to chain baseline-SG-SNV → 1-D CNN in one config.
4. **Hyper-parameter sweep adapters**

   * Thin wrappers around Optuna / Ray Tune that can launch grid/random/Bayesian searches using your DAG runner.

---

## 3 · Advanced Execution Engine Features

| Feature                       | What it gives you                                                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Streaming mode**            | For large online datasets, allow `Dataset` → `DataLoader` chunks to flow through the same pipeline DAG on the fly.                                                 |
| **Checkpoint & resume**       | Persist the DAG every *n* steps so interrupted long trainings on GPU clusters can resume.                                                                          |
| **Resource annotations**      | Optional `cpu: x`, `gpu: y`, `ram: z` per step to inform a Dask / Ray or Kubernetes executor.                                                                      |
| **Conditional branches**      | Let a step emit `"skip"`/`"take"` flags so the DAG can prune e.g. *if feature selection → 0 vars then abort training arm*.                                         |
| **Incremental / partial-fit** | Support scikit-learn’s `partial_fit` and torch’s `epoch` loops so a single pipeline config can specify both initial training and later fine-tuning on new batches. |

---

## 4 · Reproducibility & Governance

1. **Full lineage graph**

   * Merge the Dataset `processing` hash with the Pipeline DAG IDs so a single *provenance graph* covers raw spectra → final predictions.
2. **Model cards & datasheets**

   * Auto-generate a Markdown/JSON “card” on `PipelineRunner.complete()` with training metrics, dataset splits, licence, instrument details, and wavelength ranges.
3. **Signed artefacts**

   * Optional GPG signature of the bundle zip; SHA-256 of each node pickle.
4. **License & provenance tags**

   * Each node carries licence info (e.g. CC-BY, proprietary) so downstream redistribution is auditable.

---

## 5 · Observability & Ops

* **Structured logging & metrics** (OpenTelemetry, Prometheus).
* **Live dashboard**

  * Stream step durations, GPU utilisation, sample processing rate.
* **Automated drift monitoring**

  * Compare mean spectrum ± STD of new batches against training distribution; trigger alerts.
* **Audit-ready run bundles**

  * Zip file already planned—extend it with `metrics.json`, `console.log`, and HTML preview of the DAG graph.

---

## 6 · Developer Experience

| Convenience                 | Details                                                                        |
| --------------------------- | ------------------------------------------------------------------------------ |
| **Type-safe config schema** | Pydantic or `attrs` to validate every operator spec at load time.              |
| **Jupyter widgets**         | Drag-and-drop DAG builder that outputs YAML.                                   |
| **Rich CLI**                | `nirspipe train`, `nirspipe tune`, `nirspipe serve`, `nirspipe viz model.zip`. |
| **VS Code snippets**        | Autocomplete for common pipeline blocks.                                       |
| **Template projects**       | Cookiecutter repo with lint, tests, CI already wired.                          |

---

## 7 · CI / CD & Test Add-ons

* **GPU test lane** – run one mini-CNN epoch in GitHub Actions with a CUDA runner.
* **Regression corpus** – tiny public NIRS dataset (e.g. corn protein) kept in LFS; every merge must reproduce ± ϵ baseline RMSEP.
* **Integration tests** – DAG save→load→predict round-trip across chemometric and DL branches.
* **Benchmarks** – `asv` suite tracking throughput and memory for 1 k × 2048 spectra.

---

## 8 · Deployment & Serving

1. **ONNX / TorchScript export** of any DL node.
2. **Realtime REST / gRPC micro-service** wrapper that:

   * accepts *one* spectrum or a small batch,
   * performs the exact same pre-processing DAG,
   * returns prediction + confidence + wavelength attribution.
3. **Edge-device bundle**

   * Generate a single `.tar.gz` with model weights, operator pickles, and a tiny C++/Rust runner for ARM-based spectrometers.

---

### Take-away

Your current spec already covers the *core mechanics* (config → dynamic DAG → persistence).
The items above push the library to:

* **Domain depth** – lots of chemometrics + DL tooling baked-in.
* **MLOps maturity** – lineage, monitoring, reproducible bundles.
* **DX polish** – schema validation, CLI, widgets, templates.
* **Production readiness** – streaming, checkpointing, serving.

Pick and mix depending on time-to-market, but I would prioritise:

1. **Spectral pre-processing + feature selection operators** (quick win, high value).
2. **Hyper-parameter tuning hooks** (brings big accuracy gains).
3. **Signed, metrics-rich run bundles** (governance + reproducibility).
4. **Realtime serving wrapper** (needed by field instruments).

Everything else can follow in iterative releases once those pillars are rock-solid.


### Pipeline Specification (Condensed)

---

#### 1 · Current State

* **Dynamic config → implicit DAG → execution** handled by `PipelineRunner`, `PipelineOperation`, and controller registry.&#x20;
* Operators, spectral blocks, and model-zoo layers already implemented.
* `serialization.py` serialises arbitrary objects; `config.py` embeds them in specs.&#x20;
* `history.py` can log runs & bundle artefacts, but hooks are still disabled.&#x20;

---

#### 2 · Upgrade Objectives

1. **Deterministic runtime DAG**
   *Capture the real step graph (after branching) with stable, human-readable IDs.*

2. **Checkpoint & Resume**
   *Persist DAG snapshots plus fitted operators every *n* steps to survive interruptions.*

3. **Datasheet & lineage artefacts**
   *Auto-generate a Markdown/JSON “model datasheet” bundling: dataset processing IDs, DAG hash, metrics, hyper-params.*

4. **Signed bundles**
   \*Zip = graph.json + *.pkl + datasheet.md; SHA-256 per file & optional GPG signature.*

---

#### 3 · Road-map

| Phase                              | Key work                                                                                               | Notes    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ | -------- |
| **0. Re-enable history & logging** | Uncomment `history.*`, switch `print` → `logging`.                                                     | 1 day    |
| **1. DAG recorder**                | `dag/state.py` + `DagRecordingRunner` to build/save graph.json + operator pickles.                     | 3 days   |
| **2. Checkpoint engine**           | `CheckpointManager` writes incremental zip every *k* nodes; `resume(path)` merges state and continues. | 4 days   |
| **3. Datasheet generator**         | On `runner.complete()`, render Jinja template → `datasheet.md` with metrics & lineage.                 | 2 days   |
| **4. Signed bundles**              | Hash each artefact, store `manifest.sha256`; optional GPG sign zip.                                    | 1 day    |
| **5. Dag replay**                  | `DagRunner.predict()` loads bundle, runs topological order on new data.                                | 2 days   |
| **CI / tests**                     | Pytest cases: DAG ≠ None, resume mid-run, hash verification.                                           | parallel |

*Total: \~2 weeks of focused work.*

---

#### 4 · Future Features (deferred)

* **Resource annotations & job-aware executor** (Ray / Dask).
* **Live drift monitoring + alerting.**
* **Edge-device export (ONNX/TorchScript + C++ runner).**
* **Interactive DAG editor / CLI wizard.**

These can be tackled after the deterministic DAG + checkpoint + signed-bundle foundation is in place.
