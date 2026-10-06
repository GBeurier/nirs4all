# 3. Choose a function or command

Each operation has an input contract, a returned value and a persistence effect. Decide whether you need a prediction value, a durable experiment, a selected-model archive or a reusable session before choosing the API.

| Task | Python SDK entry | Native product entry | Persisted effect |
|---|---|---|---|
| Train/evaluate/select/refit | `run(pipeline, dataset, ...)` | `run` / R `nirs4all_native_run` | New outcome, results and selected-model state |
| Enumerate candidates | Generator expansion and `PipelineConfigs` | `generate` / native expansion contract | Configuration values; no learned model |
| Search and resume | `run(..., tuning=...)` | `tune`, `resume_tuning`; browser `tuneBrowser` | Trials and native checkpoint/profile |
| Predict | `predict(model, data, ...)` | `predict` | New-data predictions; no new fit |
| Retrain | `retrain(source, data, ...)` | `retrain` | New fitting campaign and new selected state |
| Export/reload | `result.export`, `load_session` | `export`, `load` | Self-contained model/profile or workflow record |
| Persist/compare results | Prediction/workspace APIs | Experiment/result APIs | Checked results, score table and predictions |
| Calibrate intervals | `calibrate`, `predict_calibrated` | `calibrate`, `predictCalibrated` | Calibrator and coverage-specific intervals |
| Audit robustness | `robustness` | `robustness` | Audit artifact; no replacement fitted model |
| Generate fixtures | `generate.regression` / `.classification` | `generate` | Synthetic inputs, not validation evidence about real instruments |

The exact arguments and returned classes differ by language. Use {doc}`interfaces` and {doc}`/api/module_api` for the public SDK signatures, {doc}`/reference/cli` for command flags, and the matching Core/R API documentation for native product names.

## Select the options by purpose

Data options control sources, targets, folds and alignment. Pipeline options control operators and parameters. Search options control sampler, objective, trials and storage. Execution options control runtime, parallelism and resources. Persistence options control workspace/archive destinations. An execution path or a checkpoint URI must not silently change the statistical recipe.

Store enough evidence to reproduce selection: objective name and direction, evaluated variants, fold identities, selected parameters and refit model. For inference, keep the saved input schema and target names but omit training targets from the new cohort.

The CLI supports `workflow run/predict/retrain/export/load`, `results`, `tuning`, `dataset`, `config`, `workspace` and `artifacts`. There is no general top-level `nirs4all run` command. Start with `<group> --help`; a SDK API feature is not automatically a CLI feature.

See {doc}`pipelines` for configuration, {doc}`evaluation` for objective choice, {doc}`results` for persistence and {doc}`deployment` for reuse.
