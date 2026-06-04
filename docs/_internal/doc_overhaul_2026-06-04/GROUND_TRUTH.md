# Ground-Truth Card — nirs4all public surface (verified from source)

**Verified:** 2026-06-04 by reading source (no memory, no guessing). `nirs4all/__init__.py`, `nirs4all/api/*.py`. Corrected after Codex review round 2.
**Version:** `0.9.1` (`nirs4all/__init__.py:45`). Any doc pinning `0.2.1` / `0.7.1` / `0.8.6` / `0.9.0` / "5.0" is STALE.
**License:** the project is **DUAL-licensed** (`LICENSE:1-13`): open-source **default `AGPL-3.0-or-later`** (optional `GPL-3.0-or-later` / `CeCILL-2.1` variants) **plus** a separate commercial license. So `CeCILL-2.1` is NOT "wrong" — it is a legitimate *optional* variant. The defect is pages that present `CeCILL-2.1` as the *sole/default* license (the default is AGPL-3.0-or-later). Do NOT add a CI gate banning the `CeCILL` token; gate instead on the landing/citation license string matching the `LICENSE` dual-license statement.

Audit agents: treat this card as authoritative. If a doc/example names a symbol, path, or signature that contradicts this card, that is a **defect**. If something is not on this card, verify against source before flagging.

## Public API — `nirs4all.__all__`

Exactly these names are public (`__init__.py:82`):

`run`, `predict`, `explain`, `retrain`, `session`, `load_session`, `Session`, `RunResult`, `PredictResult`, `ExplainResult`, `generate`, `PipelineRunner`, `PipelineConfigs`, `Run`, `RunStatus`, `RunConfig`, `generate_run_id`, `register_controller`, `CONTROLLER_REGISTRY`, `is_tensorflow_available`, `is_gpu_available`, `framework`.

`DatasetConfigs` is documented as advanced-usage but imported from `nirs4all.data` (not in top-level `__all__`).

**NOT public** (common doc errors): `nirs4all.load_dataset` ❌ (does not exist). `is_torch_available` is commented out ❌.

## Entry-point signatures (verified)

```python
nirs4all.run(pipeline, dataset, *, name="", session=None, verbose=1,
    save_artifacts=True, save_charts=True, plots_visible=False,
    random_state=None, refit=True, cache=None, project=None,
    report_naming="nirs", **runner_kwargs) -> RunResult

nirs4all.predict(model=None, data=None, *, chain_id=None, workspace_path=None,
    name="prediction_dataset", all_predictions=False, session=None,
    verbose=0, **runner_kwargs) -> PredictResult
    # store-based (preferred): predict(chain_id="abc", data=X_new)
    # model-based (legacy):     predict(model="exports/m.n4a", data=X_new)

nirs4all.explain(model, data, *, name="explain_dataset", session=None,
    verbose=1, plots_visible=True, n_samples=None,
    explainer_type="auto", **shap_params) -> ExplainResult

nirs4all.retrain(source, data, *, mode="full", name="retrain_dataset",
    new_model=None, epochs=None, session=None, verbose=1,
    save_artifacts=True, **kwargs) -> RunResult
    # mode in {"full","transfer","finetune"}

nirs4all.session(pipeline=None, name="", **kwargs)  # contextmanager -> Session
nirs4all.load_session(path) -> Session

nirs4all.generate(n_samples=1000, *, random_state=None,
    complexity="simple"|"realistic"|"complex", wavelength_range=None,
    components=None, target_range=None, train_ratio=0.8,
    as_dataset=True, name="synthetic_nirs", **kwargs) -> SpectroDataset | (X, y)
# namespace convenience: generate.regression(...), generate.classification(...),
#   generate.builder(...), generate.multi_source(...), generate.product(...),
#   generate.category(...), generate.from_template(...), generate.to_folder(...), generate.to_csv(...)
```

## Result objects (verified public members)

- **`RunResult`** (`api/result.py:214`): props `best`, `best_score`, `best_rmse`, `best_r2`,
  `best_accuracy`, `best_final`, `final`, `final_score`, `cv_best`, `cv_best_score`,
  `models`, `predictions`, `per_dataset`, `artifacts_path`, `num_predictions`;
  methods `top(n=5, **kwargs)`, `filter(**kwargs)`, `get_datasets()`, `get_models()`,
  `export(...)`, `export_model(...)`, `summary()`, `validate(...)`, `detach()`, `close()`.
- **`PredictResult`** (`api/result.py:1008`): fields `y_pred`, `metadata`, `sample_indices`,
  `model_name`, `preprocessing_steps`; props `values`, `shape`, `is_multioutput`;
  methods `to_numpy()`, `to_list()`, `to_dataframe(include_indices=True)`, `flatten()`.
- **`ExplainResult`** (`api/result.py:1144`): `shap_values`, `feature_names`, ...

## Canonical usage form to ENFORCE everywhere

```python
result = nirs4all.run(
    pipeline=[ ...inline steps... ],   # written in one shot, NO external variable
    dataset=...,
)
```
Anti-pattern to flag: `pipeline = [...]` assigned to a variable, then `nirs4all.run(pipeline, ds)`.

## Documentation surface (inventory)

- `docs/source/*.md` narrative pages: ~119 (see `find docs/source -name '*.md' -not -path '*/api/*'`).
- `docs/source/api/*.rst` autodoc stubs: 357.
- Landing toctree (`docs/source/index.md`) currently exposes overlapping front doors:
  `getting_started/`, `concepts/`, `onboarding/`, `developer/`, `user_guide/`, `reference/`, `examples/`, `migration/`, `ai_onboarding.md`.

## Examples surface (inventory)

- `examples/user/`   — 7 sections, U-prefixed (01_getting_started … 07_explainability), ~30 files.
- `examples/developer/` — 6 sections, D-prefixed (01_advanced_pipelines … 06_internals), ~32 files.
- `examples/reference/` — R01,R02,R03,R05,R06,R07 (note: **R04 missing**).
- Examples double as integration tests; runners: `run.sh`, `run_ci_examples.sh`, `ci_example_launcher.py`, `example_utils.py`.

## Known-confirmed defects (calibration set — already hand-verified)

1. `docs/source/index.md` citation pins `version = {0.2.1}` (real: 0.9.1).
2. `docs/source/index.md` license "CECILL-2.1" (real: AGPL-3.0). License statements inconsistent across `index.md`, root `CLAUDE.md`, `README.md`, `LICENSE`.
3. `docs/source/onboarding/persona_paths.md` references nonexistent files: `U01_minimal.py` (real `U01_hello_world.py`), `U02_workspace.py` (real `U02_basic_regression.py`), `examples/developer/01_branching/` (real `01_advanced_pipelines/`), `D04_stacking.py` (real `D05_meta_stacking.py`), and `nirs4all.load_dataset(...)` (not public).
4. Three competing front doors in landing toctree (`getting_started/` vs `concepts/` vs `onboarding/`), plus `developer/` overlapping `onboarding/`.
5. Example counts inconsistent across pages ("50+", "67", "progressive").
