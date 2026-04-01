# Plot Display Audit

Date: 2026-04-01

## Scope

This audit reviews how plots are created, saved, shown, and closed across:

- example scripts
- pipeline chart controllers
- `PredictionAnalyzer`
- SHAP / transfer-analysis visualization paths

The goal was to explain the long-standing symptoms reported in examples:

- plots sometimes do not appear
- plots sometimes block or hang
- closing windows does not always let the process exit
- chart display during pipeline runtime is especially fragile

## Executive Summary

The plotting stack is currently split across too many layers and the responsibilities are mixed:

1. chart controllers create figures and also call `plt.show()` during pipeline execution
2. examples sometimes call `plt.show()` again at the end
3. `PredictionAnalyzer` always saves figures by default
4. `save_charts` is not actually honored for controller chart outputs
5. the figure-reference plumbing intended for deferred display is disconnected

This produces exactly the inconsistent behavior you described:

- some paths block inside the pipeline
- some paths silently do nothing
- some example scripts add custom event-loop workarounds
- some runs save charts even when the flags say not to

The root problem is architectural, not one isolated bug.

## Concrete Findings

### 1. Pipeline chart controllers call `plt.show()` directly

The controller layer is supposed to generate chart artifacts, but it also drives GUI display:

- `nirs4all/controllers/charts/folds.py:228-233`
- `nirs4all/controllers/charts/targets.py:161-166`
- `nirs4all/controllers/charts/spectra.py:252-257`
- `nirs4all/controllers/charts/spectral_distribution.py:134-138`
- `nirs4all/controllers/charts/augmentation.py:143-146`
- `nirs4all/controllers/charts/exclusion.py:293-296`

Example from `folds.py`:

```python
if runtime_context.step_runner.plots_visible:
    # Store figure reference - user will call plt.show() at the end
    runtime_context.step_runner._figure_refs.append(fig)
    plt.show()
```

The comment says deferred display, but the code does immediate display. That contradiction exists in multiple controllers.

Impact:

- pipeline execution becomes coupled to backend/event-loop behavior
- runtime chart steps can block the run
- behavior depends on backend, OS, window manager, and execution context

### 2. The deferred figure-reference path is broken

There are two different figure-reference lists:

- `PipelineOrchestrator` owns `self._figure_refs` in `nirs4all/pipeline/execution/orchestrator.py:131-132`
- `PipelineRunner` exposes that list in `nirs4all/pipeline/runner.py:249` and `nirs4all/pipeline/runner.py:330`
- controllers append to `runtime_context.step_runner._figure_refs`, not to the orchestrator list

There is no bridge between them.

So the intended "keep figures alive and show later" design is not actually wired together.

Observed probe:

- a synthetic `fold_chart` run with `plots_visible=True` completed with `runner._figure_refs == 0`

Impact:

- deferred show is unreliable
- figure lifetime depends on incidental references
- the API state suggests figures are tracked when they are not

### 3. `save_charts` is effectively ignored for controller chart outputs

`ExecutorBuilder` stores `_save_charts` in `nirs4all/pipeline/execution/builder.py:39`, but `build()` never passes it into `PipelineExecutor`; only `save_artifacts` is passed at `nirs4all/pipeline/execution/builder.py:221-229`.

`PipelineExecutor` then always writes `step_result.outputs` to `workspace/outputs` in `nirs4all/pipeline/execution/executor.py:880-903`, with no `save_charts` guard at all.

Observed probe:

- a synthetic `fold_chart` run with `save_charts=False` still created fresh files in `workspace/outputs/`
- latest files observed after the probe:
  - `workspace/outputs/fold_visualization_1folds_train.png`
  - `workspace/outputs/fold_visualization_2folds_train.png`

Impact:

- `save_charts=False` does not mean "do not save charts"
- examples and users cannot reason correctly about side effects
- chart generation, persistence, and display are conflated

### 4. `PredictionAnalyzer` always saves by default

`PredictionAnalyzer` defaults `output_dir` to `workspace/figures` in `nirs4all/visualization/predictions.py:30-42` and `nirs4all/visualization/predictions.py:149`.

Every chart render path calls `_save_figure()` unconditionally in `_render_chart_variants()`:

- `nirs4all/visualization/predictions.py:374-430`
- `nirs4all/visualization/predictions.py:455-480`

Impact:

- creating a figure is never "just create a figure"
- many examples save charts even when the user did not request plots
- examples need extra branching and cleanup to compensate

### 5. Example CLI semantics are inconsistent and misleading

Many examples declare:

- `--plots`: "Generate plots"
- `--show`: "Display plots interactively"

But a large subset then maps `--plots` to `plots_visible=args.plots`, which means "show during pipeline runtime", not "generate only".

Representative example:

- `examples/user/01_getting_started/U02_basic_regression.py:88-90`
- `examples/user/01_getting_started/U02_basic_regression.py:114-121`
- `examples/user/01_getting_started/U02_basic_regression.py:221-222`

This example says:

- `"chart_2d"` is "only shown if `--plots` is passed"
- `plots_visible=args.plots`
- then later `if args.show: plt.show()`

So the two flags are not describing one coherent contract.

Repo-wide search showed:

- 31 example files contain `plots_visible=args.plots`
- 48 example files contain manual `plt.show(...)`

Impact:

- different examples follow different display models
- users cannot predict what `--plots` and `--show` will do
- controller-runtime display and post-run display get mixed together

### 6. `U04_aggregation.py` contains its own GUI event loop workaround

`examples/user/05_cross_validation/U04_aggregation.py:288-318` manually calls:

- `manager.show()`
- `plt.show(block=False)`
- `while any(plt.fignum_exists(...)): plt.pause(0.1)`

This is a strong smell: the example is compensating for missing framework-level plot lifecycle management.

Observed reproduction:

- `python -u examples/user/05_cross_validation/U04_aggregation.py --plots --show`
- timed out after printing:
  - `Showing top-k comparison. Close the window(s) to continue.`

That reproduces the "process does not exit" class of bug.

Impact:

- example-level workarounds are fragile and backend-specific
- closing a visible window is not a reliable exit condition
- if windows never become visible, the loop can wait forever

### 7. SHAP and transfer-analysis paths repeat the same pattern

SHAP plots call blocking `plt.show()` directly in multiple places:

- `nirs4all/visualization/analysis/shap.py:524-527`
- `nirs4all/visualization/analysis/shap.py:555-558`
- `nirs4all/visualization/analysis/shap.py:807-809`
- `nirs4all/visualization/analysis/shap.py:866-869`

Transfer analysis also calls `plt.show()` directly in `nirs4all/visualization/analysis/transfer.py`.

Impact:

- the same lifecycle problem exists outside controller charts
- there is no single plot policy in the library

## Why The Behavior Feels Random

The current implementation depends on details that change by environment:

- Matplotlib backend
- whether the process has a usable GUI session
- whether the backend blocks on `plt.show()`
- whether the window manager actually maps the windows
- whether the code is showing figures during runtime or after runtime
- whether the figures are tracked, closed, or both

That is why the same codebase can exhibit both of these behaviors:

- "nothing appears"
- "the process hangs forever"

I observed both classes during the audit.

## Recommended Fix Direction

### 1. Introduce one plot lifecycle manager

Add a single `PlotManager` or `VisualizationManager` responsible for:

- registering figures
- saving figures when `save_charts=True`
- deciding whether interactive display is allowed
- showing figures once, in one place
- closing figures after save/show

Controllers and analyzers should only create figures and hand them off.

### 2. Remove all direct `plt.show()` calls from library internals

Specifically remove direct show calls from:

- controller chart `execute()` methods
- SHAP helper methods
- transfer-analysis helpers

Internal plotting code should not run the GUI event loop.

### 3. Make the flags mean one thing everywhere

Recommended semantics:

- `save_charts`: persist chart files
- `plots_visible`: request interactive display

For examples:

- `--plots` should control saving/generation
- `--show` should control interactive display
- example code should use `plots_visible=args.show`
- example code should use `save_charts=args.plots or args.show`

Do not map `--plots` to `plots_visible`.

### 4. Stop auto-saving in `PredictionAnalyzer` by default

Better default:

- `output_dir=None`
- only save when explicitly requested

If backward compatibility is needed, introduce an explicit `save=True/False` switch and migrate the examples first.

### 5. Honor `save_charts` in the pipeline executor

Minimum required fix:

- pass `save_charts` from `ExecutorBuilder` into `PipelineExecutor`
- gate `step_result.outputs` persistence on that flag

Without this, the public API contract is misleading.

### 6. Replace custom example wait loops with one blocking show helper

For example scripts, a simple, correct model is:

1. create all figures
2. call one centralized `show_figures(figures, block=True)` helper
3. close figures

Avoid:

- `manager.show()`
- `plt.show(block=False)`
- manual `while fignum_exists(...): pause(...)`

Those patterns are backend-sensitive and are the likely reason `U04_aggregation.py` hangs.

### 7. Fix or remove the dead `_figure_refs` plumbing

Either:

- register figures into one shared manager/orchestrator-owned list

or:

- remove the unused reference path entirely

Current state is misleading and does not protect figure lifetime in the intended way.

## Suggested Implementation Order

### Phase 1: Stop the worst behavior

- remove direct `plt.show()` from controller charts
- wire `save_charts` correctly
- fix examples to use `plots_visible=args.show`
- replace `U04_aggregation.py` custom loop with one centralized blocking helper

### Phase 2: Normalize higher-level visualization APIs

- make `PredictionAnalyzer` save opt-in
- move SHAP/transfer visualization to the same display helper
- add backend/headless detection and warn instead of hanging

### Phase 3: Add regression coverage

Add tests for:

- controller charts never calling `plt.show()` directly
- `save_charts=False` producing no chart files
- `plots_visible=True` under headless/Agg not hanging
- example smoke tests for `--plots` and `--show`

## Related But Separate Issue Found During Probing

While probing runtime chart behavior, `chart_2d` also failed on a synthetic regression probe with:

- `IndexError: index 122 is out of bounds for axis 0 with size 40`
- from `nirs4all/controllers/charts/spectra.py:197`

That appears to be a chart-data assumption bug, not a display-lifecycle bug, but it reinforces the point that controller-runtime charts are currently brittle.

## Bottom Line

The current behavior is not one bug in one example. It comes from a systemic design problem:

- figure creation
- figure persistence
- interactive display
- figure lifetime
- example CLI behavior

are all mixed together.

The fix should be to centralize plot lifecycle management and make display a top-level concern, not something each controller or example re-implements differently.

## Independent Verification (2026-04-01)

I independently verified the claims in this audit against the current codebase. Findings 1, 3, 4, 5, 6, and 7 are confirmed exactly as described. All six chart controllers call `plt.show()` directly — several with comments that contradict the code ("Store figure reference - user will call plt.show() at the end" followed immediately by `plt.show()`). `ExecutorBuilder` stores `_save_charts` but never passes it to `PipelineExecutor`. `PredictionAnalyzer` always resolves a default output directory and has no opt-out mechanism. The SHAP module has 8 direct `plt.show()` calls (more than the 4 line ranges cited). The `U04_aggregation.py` manual polling loop is confirmed. The example file count for `plots_visible=args.plots` is slightly higher than stated (33 files, 77 occurrences, vs. the "31 files" cited — minor discrepancy). Finding 2 requires a nuance correction: `PipelineRunner` does alias its `_figure_refs` to the orchestrator's list via direct assignment (`self._figure_refs = self.orchestrator._figure_refs`), so the two references point to the same object. The actual disconnect is between *that* shared list and the `runtime_context.step_runner._figure_refs` that controllers append to during execution — the step runner used at execution time may not be the same object, so figures still fail to propagate upward. The net conclusion is the same, but the diagnosis should be corrected to reflect where the break actually occurs. Overall, this is a well-grounded audit. The recommended fix direction — centralizing plot lifecycle into a single manager, removing direct `plt.show()` from internals, and making flags semantically consistent — is the right approach. I would prioritize Phase 1 items (removing `plt.show()` from controllers, wiring `save_charts`, fixing example flag semantics) as they address the most user-visible pain with minimal architectural risk. The `IndexError` in `spectra.py` noted in the related-issue section should be tracked separately as it indicates an index-bounds assumption that breaks on smaller datasets.
