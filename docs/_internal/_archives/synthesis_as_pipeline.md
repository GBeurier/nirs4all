# Design Document: Synthesis as Pipeline

**Status:** Investigation & Design
**Date:** February 2026

---

## Table of Contents

1. [Current Pipeline Architecture](#1-current-pipeline-architecture)
2. [Current Synthesis Architecture](#2-current-synthesis-architecture)
3. [Synthesis as Pipeline -- Design](#3-synthesis-as-pipeline----design)
4. [Finetuning Design for Synthesis](#4-finetuning-design-for-synthesis)
5. [The "Fit" Paradigm for Synthesis Steps](#5-the-fit-paradigm-for-synthesis-steps)
6. [Impact on Existing Architecture](#6-impact-on-existing-architecture)
7. [Scientific and Technical Analysis](#7-scientific-and-technical-analysis)
8. [Missing Steps and Requirements](#8-missing-steps-and-requirements)
9. [Point of View](#9-point-of-view)

---

## 1. Current Pipeline Architecture

### 1.1 Pipeline Configuration

Pipelines in nirs4all are configured as lists of steps. Each step can be:

- A **direct operator instance** (e.g., `MinMaxScaler()`)
- A **dict with a keyword** (e.g., `{"model": PLSRegression(10)}`)
- A **nested list** (subpipeline)
- A **generator dict** (`{"_or_": [SNV, MSC]}`, `{"_range_": [1, 30, 5]}`)

`PipelineConfigs` (`nirs4all/pipeline/config/pipeline_config.py`) wraps one or more named pipeline configurations. Each configuration is a list of steps that is expanded by the generator system into concrete pipeline variants via `expand_spec()`.

The generator system (`nirs4all/pipeline/config/generator.py`) supports combinatorial expansion through keywords: `_or_`, `_range_`, `_log_range_`, `_grid_`, `_zip_`, `_chain_`, `_sample_`, `_cartesian_`. These produce multiple concrete pipelines from a single template, allowing hyperparameter search and preprocessing comparison.

### 1.2 Step Execution Flow

Pipeline execution follows this chain:

```
PipelineRunner.run()
  -> PipelineOrchestrator.execute()
    -> PipelineExecutor.execute(steps, dataset, context, runtime_context)
      -> for each step:
           StepRunner.execute(step, dataset, context, runtime_context)
             -> StepParser.parse(step) -> ParsedStep
             -> ControllerRouter.route(parsed_step) -> controller instance
             -> controller.execute(step_info, dataset, context, runtime_context)
               -> returns (updated_context, StepOutput)
```

`StepParser` normalizes raw step configurations into `ParsedStep` objects with fields: `operator`, `keyword`, `step_type` (WORKFLOW, SERIALIZED, SUBPIPELINE, DIRECT), `original_step`, `metadata`, `force_layout`.

`ControllerRouter` iterates through `CONTROLLER_REGISTRY`, calls `cls.matches(step, operator, keyword)` on each registered controller, and selects the highest-priority match (lowest `priority` number).

### 1.3 Controller Architecture

All controllers inherit from `OperatorController` (`nirs4all/controllers/controller.py`), which defines:

```python
class OperatorController(ABC):
    priority: int = 100

    @classmethod
    def matches(cls, step, operator, keyword) -> bool: ...
    @classmethod
    def use_multi_source(cls) -> bool: ...
    @classmethod
    def supports_prediction_mode(cls) -> bool: ...
    def execute(self, step_info, dataset, context, runtime_context, ...) -> Tuple[ExecutionContext, StepOutput]: ...
```

The `execute()` method receives a `ParsedStep`, a `SpectroDataset`, an `ExecutionContext`, and a `RuntimeContext`. It transforms the dataset (via the `ExecutionContext`'s `DataSelector` and `PipelineState`) and returns an updated context plus a `StepOutput` with any binary artifacts.

Existing controller types include:

| Controller | Priority | Purpose |
|-----------|----------|---------|
| `SklearnModelController` | 6 | sklearn models with fit/predict |
| `TensorFlowModelController` | 5 | TensorFlow/Keras models |
| `PyTorchModelController` | 5 | PyTorch models |
| `JaxModelController` | 5 | JAX models |
| `TransformerMixinController` | 10 | sklearn TransformerMixin instances |
| `BranchController` | 1 | Duplication and separation branching |
| `MergeController` | 1 | Branch merging |
| `SampleAugmentationController` | -- | Sample augmentation operators |
| `TagController` | -- | Tagging samples |
| `ExcludeController` | -- | Excluding samples |
| `SplitterController` | -- | Cross-validation splitters |
| Various chart controllers | -- | Visualization |

### 1.4 ExecutionContext

`ExecutionContext` (`nirs4all/pipeline/config/context.py`) carries pipeline state through execution:

- `DataSelector`: which partition, processing chain, layout, and concat_source settings to use when extracting data from the dataset
- `PipelineState`: current step number, mode (train/predict/explain), y_processing state, model step encountered flag
- `StepMetadata`: additional metadata about the current step

The context accumulates processing information as the pipeline progresses. When a transformer is applied, the processing chain is updated so subsequent steps see the transformed features.

### 1.5 Finetuning / Optuna Integration

Finetuning is handled by `OptunaManager` (`nirs4all/optimization/optuna.py`), which is instantiated by `BaseModelController`. The integration is **model-centric**: finetuning optimizes hyperparameters of a model step by repeatedly training and evaluating.

The workflow is:
1. A model step includes `finetune_params` in its config
2. The model controller detects the finetuning config
3. `OptunaManager.finetune()` creates an Optuna study
4. For each trial, parameters are sampled via `sample_hyperparameters(trial, finetune_params)`
5. A model is instantiated with those parameters, trained, and evaluated
6. The best parameters are returned and used for final training

Optimization strategies: `individual` (per-fold), `grouped` (multi-fold objective), `single` (no folds). Evaluation modes: `best`, `avg`, `robust_best`.

Parameter sampling supports: categorical lists, range tuples `(min, max)`, typed tuples `('float', min, max)`, dict configs, and log-uniform sampling.

**Key limitation for synthesis:** Finetuning is tightly coupled to model controllers. It works by calling `_get_model_instance()`, `_train_model()`, and `_evaluate_model()`. The loss is always the model's prediction error (RMSE for regression, negative accuracy for classification). There is no mechanism for custom loss functions, distribution matching, or multi-objective optimization.

### 1.6 Key Limitations Relevant to Synthesis-as-Pipeline

1. **Pipelines require input data.** The orchestrator always starts from a `SpectroDataset` with features and targets. There is no concept of a pipeline that *creates* data from scratch.

2. **Context assumes existing features.** `DataSelector` references processing chains and partitions that assume features already exist.

3. **Controllers are consumer-oriented.** They transform or model existing data, not generate it.

4. **Finetuning is model-only.** The Optuna integration assumes a train/eval cycle with prediction-based metrics.

5. **No generative loss functions.** The metrics system (`nirs4all/core/metrics.py`) provides regression metrics (RMSE, R2, MAE, etc.) and classification metrics (accuracy, F1, etc.). There are no distribution matching metrics, spectral realism scores, or other generative evaluation metrics.

6. **Step ordering is implicit.** The pipeline enforces sequential execution but has no mechanism to enforce physical ordering constraints (Beer-Lambert must precede scatter, which must precede noise, etc.).

---

## 2. Current Synthesis Architecture

### 2.1 The Generation Chain

`SyntheticNIRSGenerator` (`nirs4all/synthesis/generator.py`) implements an 18-step sequential physical forward model:

| Step | Method | Physics | Parameters |
|------|--------|---------|------------|
| 1 | `generate_concentrations()` | Sample composition | method (dirichlet/uniform/lognormal/correlated), alpha |
| 2 | `_apply_beer_lambert()` | Linear mixing: A = C @ E | Component spectra E from ComponentLibrary |
| 3 | `_apply_path_length()` | Optical path: A *= L_i | `path_length_std` |
| 4 | `_generate_baseline()` | Polynomial drift: b0 + b1x + b2x^2 + b3x^3 | `baseline_amplitude` |
| 5 | `_apply_global_slope()` | NIR upward trend | `global_slope_mean`, `global_slope_std` |
| 6 | `_apply_scatter()` | MSC-like: alpha*A + beta + gamma*x | `scatter_alpha_std`, `scatter_beta_std`, `tilt_std` |
| 7 | `generate_batch_effects()` | Session drift | n_batches, offset/gain params |
| 8 | `_apply_wavelength_shift()` | Calibration errors | `shift_std`, `stretch_std` |
| 9 | `_apply_instrumental_response()` | Gaussian convolution (ILS) | `instrumental_fwhm` |
| 10 | `_apply_detector_effects()` | Detector response curve, shot/thermal noise | Detector config |
| 11 | `_apply_multi_sensor_stitching()` | Multi-sensor junction artifacts | MultiSensorConfig |
| 12 | `_simulate_multi_scan_averaging()` | Scan averaging noise reduction | MultiScanConfig |
| 13 | Temperature effects | Band shifts, broadening | TemperatureAugmenter |
| 14 | Moisture effects | Water band perturbation | MoistureAugmenter |
| 15 | Particle size effects | Scattering wavelength dependence | ParticleSizeAugmenter |
| 16 | EMSC distortions | Multiplicative/additive scatter | EMSCDistortionAugmenter |
| 17 | Edge artifacts | Detector roll-off, stray light, curvature, truncated peaks | EdgeArtifact operators |
| 18 | `_add_artifacts()` | Random spikes, dead bands, saturation | `artifact_prob` |

Steps 1-9 and 18 are internal to the generator. Steps 10-12 use Phase 2 instrument simulation. Steps 13-17 delegate to nirs4all augmentation operators (SpectraTransformerMixin classes).

The output is `(X, Y, E)`: spectra matrix, concentration matrix, and pure component spectra.

### 2.2 Builder and Configuration

`SyntheticDatasetBuilder` (`nirs4all/synthesis/builder.py`) provides a fluent API for constructing synthesis configurations:

```python
dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic")
    .with_targets(distribution="lognormal", range=(0, 100))
    .with_partitions(train_ratio=0.8)
    .build()
)
```

The builder accumulates configuration into a `BuilderState` dataclass, then calls `SyntheticNIRSGenerator` to produce data, wrapping the result into a `SpectroDataset`.

Complexity presets (`COMPLEXITY_PARAMS` in `generator.py`) define default parameter values for `simple`, `realistic`, and `complex` modes. `SyntheticDatasetConfig` in `config.py` provides structured configuration dataclasses.

### 2.3 The Reconstruction Submodule

`nirs4all/synthesis/reconstruction/` implements a physically-grounded inverse pipeline:

1. **CanonicalForwardModel**: Computes `K(lambda) = sum(c_k * epsilon_k(lambda)) + K0(lambda)` on a high-resolution canonical grid
2. **InstrumentModel**: Wavelength warp, ILS convolution, gain/offset, resampling to target grid
3. **DomainTransform**: Absorbance/reflectance transformation
4. **PreprocessingOperator**: Match dataset preprocessing (SG derivatives, SNV, etc.)
5. **ForwardChain**: Composes all the above into a single forward pipeline

The reconstruction workflow:
1. **Global calibration** (`GlobalCalibrator`): Estimate instrument parameters (wavelength shift, ILS sigma) from prototype spectra
2. **Per-sample inversion** (`VariableProjectionSolver`): For each spectrum, solve for concentrations, baseline coefficients, path length, and environmental parameters using NNLS inner solve + nonlinear outer optimization. Uses a multiscale schedule to avoid local minima.
3. **Distribution learning** (`ParameterDistributionFitter`): Fit log-normal, Gaussian, truncated normal, or beta distributions to the recovered parameters
4. **Generation** (`ReconstructionGenerator`): Sample from learned distributions and push through the forward chain
5. **Validation** (`ReconstructionValidator`): Compare synthetic vs real data using residual analysis, PCA overlap, per-wavelength statistics

`InversionResult` contains: concentrations, baseline_coeffs, continuum_coeffs, path_length, wl_shift_residual, temperature_delta, water_activity, scattering_power, scattering_amplitude, fitted_spectrum, residuals, r_squared, rmse.

### 2.4 Current Fittable vs Fixed Parameters

In the reconstruction module:

| Parameter | Fittable | Method |
|-----------|----------|--------|
| Concentrations | Yes | NNLS linear solve |
| Baseline coefficients | Yes | NNLS linear solve |
| Continuum coefficients | Yes | NNLS linear solve |
| Path length | Yes | Nonlinear optimization (outer loop) |
| Wavelength shift | Yes | Nonlinear optimization (outer loop) |
| ILS sigma | Yes | Global calibration |
| Temperature delta | Yes | Nonlinear optimization (with environmental flag) |
| Water activity | Yes | Nonlinear optimization (with environmental flag) |
| Scattering power | Yes | Nonlinear optimization (with environmental flag) |
| Component spectra (E) | Fixed | From ComponentLibrary |
| Noise model | Estimated | From residuals (post-hoc) |
| Scatter model (MSC params) | Not fitted | No inversion for scatter |
| Batch effects | Not fitted | No inversion mechanism |
| Edge artifacts | Not fitted | No inversion mechanism |

The `RealDataFitter` (`fitter.py`) takes a different, more heuristic approach: it analyzes statistical properties of real data (mean, std, slope, curvature, noise, PCA structure) and estimates generator parameters that would produce similar statistics. This does not perform physical inversion.

---

## 3. Synthesis as Pipeline -- Design

### 3.1 Core Concept

The idea is to express the synthesis forward model as a nirs4all pipeline: a list of steps that can be configured, expanded by generators, finetuned with Optuna, and composed using branching and merging. Each physical effect in the generation chain becomes a pipeline step with its own controller.

### 3.2 Two Operating Modes

A synthesis pipeline would need to support two fundamentally different modes:

**Mode A: Pure Generation (no real data)**

The pipeline starts with no input data. The first step generates concentrations, and subsequent steps transform spectra through the physical chain. The output is a synthetic dataset. There is no fit/predict cycle -- just forward execution.

```python
synthesis_pipeline = [
    ConcentrationGenerator(n_samples=1000, method="dirichlet"),
    BeerLambertStep(component_library=my_library),
    PathLengthStep(std=0.08),
    BaselineDriftStep(amplitude=0.04),
    GlobalSlopeStep(mean=0.15, std=0.03),
    ScatterStep(alpha_std=0.05, beta_std=0.02),
    WavelengthShiftStep(shift_std=0.3, stretch_std=0.001),
    InstrumentalBroadeningStep(fwhm=8.0),
    NoiseStep(base=0.003, signal_dep=0.01),
]

result = nirs4all.run(
    pipeline=synthesis_pipeline,
    dataset=None,  # No input data -- generation from scratch
    mode="generate",
)
```

**Mode B: Fitting to Real Data (learn forward model)**

The pipeline has access to real spectra. Each step's parameters are fitted to match the observed data. Concentrations become the "targets" that are solved for, and spectra are the "input." This is essentially the reconstruction module expressed as a pipeline.

```python
fitting_pipeline = [
    ConcentrationSolver(),  # Solve for concentrations from spectra
    BeerLambertStep(component_library=my_library),  # Fixed component spectra
    PathLengthStep(),  # Fit path length distribution
    BaselineDriftStep(),  # Fit baseline polynomial
    ScatterStep(),  # Fit MSC parameters
    NoiseStep(),  # Estimate noise from residuals
]

result = nirs4all.run(
    pipeline=fitting_pipeline,
    dataset=real_spectra,
    mode="fit_synthesis",
)

# Then generate using fitted parameters
synthetic = nirs4all.run(
    pipeline=result.fitted_pipeline,
    dataset=None,
    mode="generate",
    n_samples=5000,
)
```

**Mode C: Augmentation Generation (variations of real data)**

The pipeline takes real data and generates augmented versions by applying controlled perturbations through the physical chain. This is a form of physically-informed data augmentation.

```python
augmentation_pipeline = [
    {"sample_augmentation": PathLengthStep(std=0.05)},
    {"sample_augmentation": ScatterStep(alpha_std=0.03)},
    {"sample_augmentation": NoiseStep(base=0.002)},
]

result = nirs4all.run(
    pipeline=[MinMaxScaler()] + augmentation_pipeline + [PLSRegression(10)],
    dataset=real_spectra,
)
```

**Mode D: Transfer (fit on instrument A, generate for instrument B)**

```python
# Fit forward model to instrument A data
fit_result = nirs4all.run(
    pipeline=fitting_pipeline,
    dataset=instrument_A_data,
    mode="fit_synthesis",
)

# Modify instrument parameters and generate for instrument B
transfer_pipeline = fit_result.fitted_pipeline.with_modifications(
    InstrumentalBroadeningStep={"fwhm": 12.0},
    NoiseStep={"base": 0.005},
    WavelengthShiftStep={"shift_std": 0.5},
)

synthetic_B = nirs4all.run(
    pipeline=transfer_pipeline,
    dataset=None,
    mode="generate",
    n_samples=5000,
)
```

### 3.3 New Controllers Needed

#### 3.3.1 GenerationController (Base)

A new base controller for steps that generate or transform data within the forward model:

```python
class ForwardModelController(OperatorController):
    """Base controller for forward model synthesis steps."""

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True  # Generation is always forward

    def execute(self, step_info, dataset, context, runtime_context, ...):
        operator = step_info.operator

        if context.state.mode == "generate":
            # Forward generation: apply physical effect
            return self._generate(operator, dataset, context, runtime_context)
        elif context.state.mode == "fit_synthesis":
            # Inverse fitting: estimate parameters from data
            return self._fit(operator, dataset, context, runtime_context)
        else:
            # Standard pipeline mode: use as augmentation
            return self._augment(operator, dataset, context, runtime_context)
```

#### 3.3.2 Specific Forward Model Controllers

Each synthesis step would need a controller or a general-purpose controller that dispatches based on operator type:

| Controller | Matches | Priority |
|-----------|---------|----------|
| `ConcentrationGeneratorController` | `ConcentrationGenerator` instances | 3 |
| `BeerLambertController` | `BeerLambertStep` instances | 3 |
| `ForwardEffectController` | All other forward model steps (PathLength, Baseline, Scatter, Noise, etc.) | 3 |

Alternatively, a single `ForwardModelController` could handle all forward model operators, dispatching internally based on operator type. This mirrors how `TransformerMixinController` handles all sklearn transformers.

#### 3.3.3 ConcentrationGeneratorController

This is the most unusual controller because it creates data from nothing:

```python
class ConcentrationGeneratorController(OperatorController):
    def execute(self, step_info, dataset, context, runtime_context, ...):
        operator = step_info.operator  # ConcentrationGenerator

        if dataset is None or dataset.n_samples() == 0:
            # Pure generation: create concentrations and initialize dataset
            C = operator.generate(n_samples=operator.n_samples)
            wavelengths = operator.wavelengths  # or from context

            # Create a new SpectroDataset with concentrations as both X and Y
            # X will be overwritten by subsequent steps
            dataset = SpectroDataset("synthetic")
            dataset.add_samples(C, {"partition": "train"})
            dataset.add_targets(C)

            # Initialize the spectra as zero (Beer-Lambert will fill them)
            context.state.synthesis_concentrations = C
            context.state.synthesis_wavelengths = wavelengths
        else:
            # Dataset exists: use concentrations from targets
            C = dataset.y({"partition": "train"})

        return context, StepOutput(...)
```

### 3.4 How "No Input Data" Works

The fundamental challenge: the pipeline system always starts with a `SpectroDataset`. For pure generation, we need either:

**Option A: Empty dataset + generation step.** The first step (ConcentrationGenerator) creates the dataset. Subsequent steps transform the data. This requires the orchestrator to handle `dataset=None` gracefully and the first controller to create the dataset.

**Option B: Seed dataset.** Provide a minimal dataset (just wavelengths, no samples) as a "seed." The ConcentrationGenerator populates it. This is cleaner because the dataset always exists, but the seed needs to carry wavelength information.

**Option C: Two-phase execution.** A "generation plan" phase produces a dataset, and a "pipeline" phase processes it. This would mean the synthesis pipeline is actually two pipelines chained together.

**Recommended: Option B.** Create a `SeedDataset` factory:

```python
seed = SeedDataset(
    wavelengths=np.arange(1000, 2500, 2),
    n_samples=1000,
    name="synthetic",
)
# seed is a SpectroDataset with n_samples rows of zeros and the correct wavelength grid
```

**Cross-validation interaction**: In pure generation mode, there is no train/test split. The entire generated dataset is the output. If the synthesis pipeline is followed by model training steps, a splitter must be inserted between the generation and training phases to create the necessary partitions. This is a fundamental difference from standard pipelines where the dataset arrives pre-split or a splitter operates on existing data.

### 3.5 Using Existing Pipeline Features

**Generators (`_or_`, `_range_`)**: Compare different physical configurations:

```python
pipeline = [
    ConcentrationGenerator(n_samples=500),
    BeerLambertStep(component_library=my_library),
    {"_or_": [PathLengthStep(std=0.05), PathLengthStep(std=0.1), PathLengthStep(std=0.2)]},
    {"_or_": [ScatterStep(alpha_std=0.03), ScatterStep(alpha_std=0.08)]},
    NoiseStep(base=0.003),
]
# Expands to 6 synthesis pipelines with different physical parameters
```

**Branching**: Generate data with different physical scenarios:

```python
pipeline = [
    ConcentrationGenerator(n_samples=500),
    BeerLambertStep(),
    {"branch": [
        [PathLengthStep(std=0.05), ScatterStep(alpha_std=0.03)],  # Clean scenario
        [PathLengthStep(std=0.15), ScatterStep(alpha_std=0.10)],  # Noisy scenario
    ]},
    {"merge": "concat"},  # Combine into one dataset
    NoiseStep(base=0.003),
]
```

**Stacking/meta-models**: Use synthesis as input to a training pipeline:

```python
pipeline = [
    # Stage 1: Generate synthetic training data
    ConcentrationGenerator(n_samples=2000),
    BeerLambertStep(),
    PathLengthStep(std=0.08),
    ScatterStep(alpha_std=0.05),
    NoiseStep(base=0.003),

    # Transition: mark generated data as train/test (required before model)
    ShuffleSplit(n_splits=5, test_size=0.2),

    # Stage 2: Train a model on the synthetic data
    MinMaxScaler(),
    PLSRegression(n_components=10),
]
```

Note: the transition from generation steps to model steps requires inserting a splitter. Without it, the pipeline has no train/test partitions and model evaluation is impossible. This boundary between "generation phase" and "training phase" is an important design consideration -- the pipeline needs to know when generation ends and training begins. A possible approach is an explicit `{"phase": "training"}` marker step, or implicit detection when a splitter or model is encountered.

### 3.6 Pipeline Syntax Summary

```python
# Pure generation
nirs4all.run(
    pipeline=[ConcentrationGenerator(1000), BeerLambertStep(), ...],
    dataset=SeedDataset(wavelengths=np.arange(1000, 2500, 2)),
    mode="generate",
)

# Fitting to real data
nirs4all.run(
    pipeline=[ConcentrationSolver(), BeerLambertStep(), ...],
    dataset=real_data,
    mode="fit_synthesis",
)

# Synthesis + training (combined pipeline)
nirs4all.run(
    pipeline=[
        ConcentrationGenerator(500),
        BeerLambertStep(),
        ...,
        # After synthesis, train a model
        MinMaxScaler(),
        PLSRegression(10),
    ],
    dataset=SeedDataset(wavelengths=wl),
)

# Augmentation generation (applied to real data)
nirs4all.run(
    pipeline=[
        {"sample_augmentation": PathLengthStep(std=0.05)},
        {"sample_augmentation": ScatterStep(alpha_std=0.03)},
        MinMaxScaler(),
        PLSRegression(10),
    ],
    dataset=real_data,
)
```

---

## 4. Finetuning Design for Synthesis

### 4.1 Custom Loss Functions

Standard model finetuning minimizes prediction error (RMSE, accuracy). Synthesis finetuning requires fundamentally different losses:

#### 4.1.1 Distribution Matching Losses

These measure how well the synthetic distribution matches the real distribution:

- **Kolmogorov-Smirnov (KS) test statistic**: Maximum absolute difference between cumulative distributions. Per-wavelength KS statistic averaged across wavelengths. Sensitive to location and shape differences.
  ```
  L_KS = mean_lambda(KS(F_real(lambda), F_synth(lambda)))
  ```

- **Wasserstein distance (Earth Mover's Distance)**: Minimum "work" to transform one distribution into another. More informative than KS for optimization because it provides gradient information even when distributions do not overlap.
  ```
  L_W = mean_lambda(W1(P_real(lambda), P_synth(lambda)))
  ```

- **Maximum Mean Discrepancy (MMD)**: Kernel-based two-sample test. Uses RBF kernels in spectral space. Differentiable and smooth, making it suitable for gradient-based optimization.
  ```
  L_MMD = ||mu_P - mu_Q||_H^2
  ```
  where H is a reproducing kernel Hilbert space.

- **Frechet Spectral Distance (FSD)**: Analogous to Frechet Inception Distance (FID) in image generation. Compute mean and covariance of real and synthetic spectra in PCA space, then measure their Frechet distance:
  ```
  FSD = ||mu_r - mu_s||^2 + Tr(Sigma_r + Sigma_s - 2*(Sigma_r * Sigma_s)^(1/2))
  ```

#### 4.1.2 Spectral Realism Losses

These evaluate whether individual synthetic spectra are physically plausible:

- **Peak statistics matching**: Compare number, position, width, and intensity of absorption peaks between real and synthetic data.

- **Baseline smoothness**: Penalize unrealistic baseline patterns. Measure via second-derivative energy:
  ```
  L_baseline = mean(||d^2 X / d lambda^2||)
  ```

- **Noise characteristics**: Compare noise variance, autocorrelation, and heteroscedasticity patterns. Estimate noise from high-frequency residuals after smoothing:
  ```
  L_noise = |sigma_real - sigma_synth| + |rho_real - rho_synth|
  ```
  where sigma is noise standard deviation and rho is noise autocorrelation.

- **Signal-to-noise ratio (SNR) matching**: Compare SNR profiles across wavelengths.

#### 4.1.3 Reconstruction Error

When fitting to real data, the reconstruction error measures how well the forward model reproduces observed spectra:

```
L_recon = mean(||X_real - ForwardModel(theta_fitted)||^2)
```

This is already used by the reconstruction module's `VariableProjectionSolver`.

#### 4.1.4 Diversity Metrics

Ensure the synthetic data covers the feature space adequately:

- **PCA coverage**: Compare the volume of PCA subspaces spanned by real and synthetic data.
- **Convex hull coverage**: Fraction of real data's convex hull (in reduced space) covered by synthetic data.
- **Sample diversity**: Average pairwise distance in feature space (to avoid mode collapse).

#### 4.1.5 Composite Loss

In practice, synthesis finetuning would use a weighted combination:

```
L_total = w1 * L_distribution + w2 * L_realism + w3 * L_reconstruction + w4 * L_diversity
```

### 4.2 Optuna Integration for Synthesis

The current `OptunaManager` needs extension to support synthesis-specific optimization:

#### 4.2.1 SynthesisOptunaManager

```python
class SynthesisOptunaManager:
    """Optuna manager for synthesis parameter optimization."""

    def optimize(
        self,
        synthesis_pipeline: List[Any],
        real_data: SpectroDataset,
        loss_fn: Callable,
        n_trials: int = 100,
        parameter_space: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Optimize synthesis pipeline parameters to match real data.

        For each trial:
        1. Sample synthesis parameters from the search space
        2. Execute the synthesis pipeline with those parameters
        3. Evaluate the loss between synthetic and real data
        4. Return the best parameters
        """
        def objective(trial):
            params = self.sample_synthesis_params(trial, parameter_space)
            pipeline_with_params = self.inject_params(synthesis_pipeline, params)

            # Execute synthesis pipeline
            synthetic_data = execute_synthesis(pipeline_with_params)

            # Compute loss
            return loss_fn(real_data, synthetic_data)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
```

#### 4.2.2 Finetuning Config for Synthesis Steps

Each synthesis step could declare its optimizable parameters:

```python
pipeline = [
    ConcentrationGenerator(n_samples=500),
    BeerLambertStep(component_library=my_library),
    {
        "model": PathLengthStep(),
        "finetune_params": {
            "std": ("float", 0.01, 0.3),
            "n_trials": 50,
            "loss": "wasserstein",
        },
    },
    {
        "model": ScatterStep(),
        "finetune_params": {
            "alpha_std": ("float", 0.01, 0.2),
            "beta_std": ("float", 0.005, 0.1),
            "n_trials": 50,
            "loss": "mmd",
        },
    },
]
```

However, this per-step approach is problematic because the loss depends on the final output, not intermediate steps. A better approach is pipeline-level finetuning:

```python
nirs4all.run(
    pipeline=synthesis_pipeline,
    dataset=real_data,
    mode="fit_synthesis",
    finetune={
        "n_trials": 200,
        "loss": "wasserstein+realism",
        "parameters": {
            "PathLengthStep.std": ("float", 0.01, 0.3),
            "ScatterStep.alpha_std": ("float", 0.01, 0.2),
            "NoiseStep.base": ("float_log", 0.0001, 0.01),
        },
    },
)
```

### 4.3 What Parameters Are Optimizable vs Fixed

| Parameter | Optimizable | Rationale |
|-----------|-------------|-----------|
| Component spectra (E) | Fixed | Physical constants (absorptivity profiles) |
| Number of components | Fixed | Domain knowledge |
| Wavelength grid | Fixed | Instrument property |
| Concentration distribution alpha | Optimizable | Controls compositional diversity |
| Path length std | Optimizable | Sample-dependent optical property |
| Baseline amplitude | Optimizable | Instrument + sample dependent |
| Global slope mean/std | Optimizable | Scattering + instrument dependent |
| Scatter alpha/beta/tilt std | Optimizable | Physical scatter variability |
| Wavelength shift/stretch std | Optimizable | Instrument calibration stability |
| Instrumental FWHM | Partially | Known for some instruments, estimable for others |
| Noise base/signal_dep | Optimizable | Detector + electronics dependent |
| Temperature variation | Optimizable if relevant | Environmental control quality |
| Moisture variation | Optimizable if relevant | Sample preparation consistency |
| Particle size params | Optimizable | Sample physical property |
| Artifact probability | Optimizable | Data quality level |

### 4.4 Multi-Objective Optimization

Synthesis quality has multiple dimensions that may conflict. For example:
- Maximizing distribution match may sacrifice spectral realism
- Maximizing diversity may reduce mean reconstruction accuracy
- Matching noise exactly may cause overfitting to a specific measurement session

Optuna supports multi-objective optimization via `optuna.create_study(directions=["minimize", "minimize"])`. A Pareto front of solutions would give the user control over the trade-off:

```python
study = optuna.create_study(
    directions=["minimize", "minimize", "maximize"],
    # L_distribution, L_realism, diversity_score
)
```

### 4.5 How This Differs from Standard Model Finetuning

| Aspect | Model Finetuning | Synthesis Finetuning |
|--------|-----------------|---------------------|
| Objective | Minimize prediction error | Match data distribution |
| Loss | RMSE, accuracy | Wasserstein, MMD, FSD, realism scores |
| Parameters | Model hyperparameters | Physical effect magnitudes |
| Data flow | Input -> Model -> Prediction | Parameters -> Forward model -> Synthetic data |
| Evaluation | Compare y_pred vs y_true | Compare X_synth distribution vs X_real distribution |
| Cross-validation | Split samples | Not applicable (compare distributions) |
| Overfitting risk | Model memorizes training data | Synthesis memorizes measurement-specific artifacts |
| Multi-objective | Rarely needed | Often needed (realism vs diversity vs distribution match) |

---

## 5. The "Fit" Paradigm for Synthesis Steps

### 5.1 What Does fit() Mean for Each Step?

For each synthesis step, "fitting" means estimating the step's parameters from real data. This is the inverse problem: given observed spectra, recover the physical parameters that produced them.

#### 5.1.1 Beer-Lambert Step: Fitting Absorptivity Profiles

**What fit() means**: Given spectra X and (known or estimated) concentrations C, solve for the pure component spectra E in `X = C @ E + residuals`.

**Parameters**: Component spectra matrix E (n_components x n_wavelengths).

**Method**: Non-negative least squares (NNLS) per wavelength: `E_fitted = NNLS(C, X)`.

**Challenges**:
- Requires known concentrations (chicken-and-egg problem if concentrations are unknown)
- The reconstruction module handles this by alternating between concentration estimation and spectrum fitting
- Component spectra are typically treated as fixed physical constants, not fitted. Fitting them risks overfitting to instrument-specific artifacts.

**Recommendation**: Keep E fixed from the component library. If E needs adaptation, fit only small perturbations (scaling and shift) rather than full spectra.

#### 5.1.2 Path Length Step: Fitting Distribution Parameters

**What fit() means**: Estimate the distribution of path length factors across samples.

**Parameters**: `(mean_L, std_L)` for the path length distribution.

**Method**: The reconstruction module fits path length as a per-sample nonlinear parameter. From the fitted per-sample values, the distribution is learned post-hoc. In a pipeline fit, this could be:
1. Estimate per-sample path lengths from the scatter-corrected spectra (using MSC or similar)
2. Fit a distribution to the per-sample values

**Challenges**:
- Path length is entangled with concentration scaling: doubling concentration halves path length equivalently. This causes an identifiability problem.
- The reconstruction module resolves this by constraining concentrations to be non-negative and path length to be positive, plus using reference spectra.

**What loss drives fitting**: Reconstruction error after applying the path length model.

#### 5.1.3 Baseline Step: Fitting Polynomial Coefficients

**What fit() means**: Estimate the distribution of baseline polynomial coefficients across samples.

**Parameters**: Distribution of `(b0, b1, b2, b3)` coefficients.

**Method**:
1. Estimate baselines from real spectra (e.g., using asymmetric least squares, rolling ball, or SNIP)
2. Fit polynomials to the estimated baselines
3. Learn the distribution of polynomial coefficients

**Challenges**:
- Baseline estimation is itself an ill-posed problem. Different baseline correction methods give different results.
- The baseline is entangled with the continuum absorption and low-frequency scatter effects.

**What loss drives fitting**: Smoothness of residuals after baseline removal, plus distribution matching of baseline statistics.

#### 5.1.4 Scatter Step: Fitting MSC Parameters

**What fit() means**: Estimate the distribution of multiplicative scatter (alpha), additive offset (beta), and tilt (gamma) parameters.

**Parameters**: Distributions of `(alpha, beta, gamma)` per sample.

**Method**: Multiplicative Scatter Correction provides these directly:
1. Compute mean spectrum of the dataset
2. For each sample, regress against the mean: `X_i = alpha_i * X_mean + beta_i`
3. Tilt can be estimated from the slope of the residuals

This is one of the most naturally fittable steps because MSC already provides per-sample parameters.

**What loss drives fitting**: Matching the scatter variability (spread of alpha, beta, gamma values).

#### 5.1.5 Temperature/Moisture Steps: Fitting Effect Magnitudes

**What fit() means**: Estimate the range and distribution of temperature/moisture variations.

**Parameters**: Temperature variation range, moisture content range, effect strengths.

**Method**:
- If temperature metadata is available: correlate spectral changes with temperature
- If not: detect temperature-sensitive bands (water OH stretch near 1940nm, CH stretch variations) and estimate effect magnitude from their variability
- The `RealDataFitter` (`fitter.py`) already implements heuristic detection of environmental effects

**Challenges**:
- Without metadata, temperature and moisture effects are confounded with other variability sources
- Effects may be small relative to other sources of variation

**What loss drives fitting**: Matching the temperature-dependent variability patterns in water and OH bands.

#### 5.1.6 Noise Step: Fitting Noise Model Parameters

**What fit() means**: Estimate the noise characteristics (base noise, signal-dependent noise, heteroscedasticity, autocorrelation).

**Parameters**: `(noise_base, noise_signal_dep)` and optionally noise autocorrelation structure.

**Method**:
1. Smooth each spectrum (e.g., Savitzky-Golay)
2. Compute residuals (spectrum - smoothed)
3. Estimate noise variance as a function of signal level
4. Fit `sigma(A) = noise_base + noise_signal_dep * |A|`

The reconstruction module already does this in `estimate_noise_from_residuals()`.

**Challenges**:
- Residuals from smoothing contain both noise and fine spectral structure. A too-aggressive smooth removes real features; a too-gentle smooth leaves structure in the "noise."
- Noise characteristics can vary across the wavelength range (heteroscedastic noise).

**What loss drives fitting**: Matching the noise variance and autocorrelation profiles.

#### 5.1.7 Wavelength Shift/Stretch: Fitting Calibration Parameters

**What fit() means**: Estimate the distribution of wavelength calibration errors across samples.

**Parameters**: `(shift_std, stretch_std)`.

**Method**:
1. Identify sharp spectral features (peaks, zero-crossings of derivatives)
2. Track their apparent position across samples
3. Fit shift and stretch distributions to the positional variability

The reconstruction module fits per-sample wavelength shift as a nonlinear parameter.

**Challenges**:
- Requires identifiable spectral features (peaks, shoulders)
- Shift and stretch are confounded in regions far from the reference wavelength

### 5.2 Stochastic Steps

Several steps are inherently stochastic: noise addition, random artifacts, batch effects. For these:

- **fit() means**: estimate the **distribution parameters** of the random process, not the specific random values
- The fitted model generates new random realizations during generation
- Validation compares statistical properties (variance, frequency, pattern), not individual samples

### 5.3 Ordering Constraints and Dependencies

The synthesis chain has strict ordering requirements dictated by physics:

1. **Concentrations** must come first (they define the chemical composition)
2. **Beer-Lambert** must follow concentrations (it converts chemical to spectral domain)
3. **Path length** must follow Beer-Lambert (it modifies absorbance scale)
4. **Baseline** can follow path length (additive, order-independent with slope)
5. **Scatter** must precede noise (scatter is a physical effect, noise is measurement)
6. **Instrumental effects** (broadening, shift) must precede noise (they are instrument transfer function effects)
7. **Noise** must come near the end (it represents measurement uncertainty)
8. **Artifacts** must come last (they represent post-measurement corruption)

For fitting, the ordering matters even more because each step's fitting depends on what has already been explained by previous steps:

1. Fit concentrations + path length first (they explain the most variance)
2. Fit baseline from residuals of step 1
3. Fit scatter from residuals of steps 1-2
4. Fit instrumental effects from the spectral resolution of residuals
5. Estimate noise from the final residuals

This creates a **sequential fitting dependency** that is different from standard pipeline fitting where each step is fitted independently on the current data. The synthesis fitting is fundamentally an inversion problem, and the reconstruction module's variable projection approach handles this properly.

---

## 6. Impact on Existing Architecture

### 6.1 Changes to generator.py

The current `SyntheticNIRSGenerator` would become a **preset factory** that produces pipeline configurations:

```python
class SyntheticNIRSGenerator:
    """Generates synthesis pipeline configurations."""

    def to_pipeline(self) -> List[Any]:
        """Convert current configuration to a synthesis pipeline."""
        steps = [
            ConcentrationGenerator(
                n_samples=self.n_samples,
                method=self.concentration_method,
                component_library=self.library,
            ),
            BeerLambertStep(E=self.E, wavelengths=self.wavelengths),
            PathLengthStep(std=self.params["path_length_std"]),
            BaselineDriftStep(amplitude=self.params["baseline_amplitude"]),
            GlobalSlopeStep(mean=self.params["global_slope_mean"], std=self.params["global_slope_std"]),
            ScatterStep(alpha_std=self.params["scatter_alpha_std"], beta_std=self.params["scatter_beta_std"], tilt_std=self.params["tilt_std"]),
        ]
        # ... add optional steps based on configuration
        return steps
```

The existing `generate()` method and all the `_apply_*` methods could remain as a non-pipeline fast path, or be gradually deprecated in favor of the pipeline approach. Keeping both paths allows backward compatibility during transition.

### 6.2 Changes to builder.py

`SyntheticDatasetBuilder.build()` could add a `.to_pipeline()` option:

```python
builder = SyntheticDatasetBuilder(n_samples=1000, random_state=42)
builder.with_features(complexity="realistic")

# Existing path: build directly
dataset = builder.build()

# New path: get a pipeline
pipeline = builder.to_pipeline()
result = nirs4all.run(pipeline=pipeline, dataset=SeedDataset(wavelengths=wl))
```

### 6.3 Relationship with reconstruction/ Submodule

The reconstruction submodule already implements the inverse pipeline (fit forward model parameters from real data). It would become the **fitting backend** for the synthesis pipeline's fit mode.

The relationship:
- `ForwardChain` -> becomes the composition of forward model pipeline steps
- `VariableProjectionSolver` -> becomes the fitting algorithm used by `ConcentrationSolver` + `PathLengthStep.fit()`
- `ParameterDistributionFitter` -> becomes the distribution learning applied to fitted per-sample parameters
- `ReconstructionGenerator` -> becomes the generation mode of the fitted pipeline
- `ReconstructionValidator` -> becomes the validation step applied to synthesis pipeline output

Rather than duplicating the reconstruction module, the synthesis pipeline should **delegate to it** for fitting operations. The pipeline provides the configuration interface; the reconstruction module provides the algorithms.

### 6.4 Changes to Pipeline Infrastructure

#### 6.4.1 New Execution Mode

The `PipelineRunner` and `PipelineOrchestrator` need to support new modes:

```python
class PipelineRunner:
    def __init__(self, ..., mode: str = "train"):
        # mode can be: "train", "predict", "explain", "generate", "fit_synthesis"
        ...
```

In `generate` mode:
- No cross-validation (no train/test split)
- No prediction storage
- Output is a `SpectroDataset` (not predictions)
- No model evaluation metrics

In `fit_synthesis` mode:
- The "training" is fitting the forward model
- The "evaluation" uses distribution matching metrics
- The output is a fitted pipeline (parameters) + quality metrics

#### 6.4.2 SeedDataset

A factory for creating empty datasets with wavelength information:

```python
def SeedDataset(wavelengths, n_samples=0, name="seed"):
    """Create a minimal dataset for synthesis pipelines."""
    dataset = SpectroDataset(name)
    if n_samples > 0:
        zeros = np.zeros((n_samples, len(wavelengths)))
        dataset.add_samples(zeros, {"partition": "train"})
    dataset.set_wavelengths(wavelengths)
    return dataset
```

#### 6.4.3 Ordering Validation

A synthesis-specific validator that checks physical ordering constraints:

```python
SYNTHESIS_ORDER = {
    ConcentrationGenerator: 0,
    BeerLambertStep: 1,
    PathLengthStep: 2,
    BaselineDriftStep: 3,
    GlobalSlopeStep: 3,  # Same level as baseline
    ScatterStep: 4,
    WavelengthShiftStep: 5,
    InstrumentalBroadeningStep: 5,
    TemperatureStep: 6,
    MoistureStep: 6,
    NoiseStep: 7,
    ArtifactStep: 8,
}

def validate_synthesis_order(pipeline):
    """Validate that synthesis steps are in physically correct order."""
    last_order = -1
    for step in pipeline:
        order = SYNTHESIS_ORDER.get(type(step), None)
        if order is not None:
            if order < last_order:
                raise ValueError(f"Step {step} is out of physical order")
            last_order = order
```

### 6.5 Changes to Controller System

The controller registry needs no structural changes -- new controllers are added via `@register_controller`. However, the `OperatorController.execute()` signature may need extension to support the `generate` and `fit_synthesis` modes, since the current `mode` parameter only handles `train` and `predict`.

### 6.6 Changes to Metrics/Losses

`nirs4all/core/metrics.py` needs new metric functions:

```python
# Distribution matching metrics
def wasserstein_distance(X_real, X_synth, per_wavelength=True): ...
def ks_statistic(X_real, X_synth, per_wavelength=True): ...
def mmd(X_real, X_synth, kernel="rbf", bandwidth=None): ...
def frechet_spectral_distance(X_real, X_synth, n_components=10): ...

# Spectral realism metrics
def noise_variance_match(X_real, X_synth): ...
def peak_statistics_match(X_real, X_synth, wavelengths): ...
def baseline_smoothness(X, wavelengths): ...

# Diversity metrics
def pca_coverage(X_real, X_synth, n_components=5): ...
def sample_diversity(X, metric="euclidean"): ...
```

### 6.7 Impact on SpectroDataset

Minimal impact. The `SpectroDataset` already supports:
- Adding samples from numpy arrays
- Setting metadata, targets, and folds
- Wavelength information (via signal_type and feature count)

The main addition would be explicit wavelength storage (currently wavelengths are implicit in feature column count, or stored in metadata). A `set_wavelengths()` / `get_wavelengths()` API would be helpful.

---

## 7. Scientific and Technical Analysis

### 7.1 Identifiability Problems

Several synthesis parameters cannot be uniquely determined from spectra alone:

**Concentration-path length ambiguity**: In the Beer-Lambert law `A = L * C @ E`, multiplying all concentrations by a factor k and dividing path length by k produces identical spectra. Resolution: fix the path length mean to 1.0 and allow concentrations to absorb the scaling. Or use known reference concentrations.

**Baseline-scatter ambiguity**: A polynomial baseline and the scatter offset `beta` both add low-frequency components. Resolution: fit them jointly (as the reconstruction module does with the combined design matrix), or use physical constraints (baseline should be smooth, scatter offset should be sample-independent in structure).

**Global slope vs scatter tilt**: The global slope effect and the scatter tilt `gamma*x` are both linear in wavelength. Resolution: absorb one into the other, or constrain one to be sample-independent.

**Instrumental broadening vs noise**: Both reduce spectral contrast, but via different mechanisms (convolution vs additive noise). In practice they are distinguishable because broadening is deterministic and noise is stochastic.

### 7.2 Ill-Conditioning

**Beer-Lambert inversion**: When component spectra are similar (high collinearity in E), the concentration solution is unstable. The condition number of the E matrix determines accuracy. For closely related components (e.g., glucose and fructose), concentrations may be poorly determined.

**Baseline fitting**: High-order polynomial baselines can oscillate and fit noise. Regularization (or constraining to low orders) is essential.

**Multi-parameter joint optimization**: Fitting all synthesis parameters simultaneously is a high-dimensional nonlinear optimization prone to local minima. The reconstruction module addresses this with multiscale scheduling and variable projection (separating linear and nonlinear parameters).

### 7.3 Overfitting Risks

**Noise parameter overfitting**: If the noise model is fitted to a specific dataset, it captures not just the instrument's noise characteristics but also the specific environmental conditions (temperature, humidity) and sample preparation details of that measurement session. Generating synthetic data with these parameters produces data that matches that session but may not generalize to other sessions.

**Mitigation**:
- Use multiple measurement sessions to estimate noise (average across sessions)
- Add regularization that pulls noise parameters toward generic defaults
- Validate on held-out sessions (not held-out samples from the same session)

**Scatter parameter overfitting**: Similar concern. The scatter distribution from one dataset may not represent the general population.

**Mitigation**:
- Use physically motivated constraints (e.g., alpha > 0.7, beta centered at 0)
- Cross-validate with independent measurement sessions

### 7.4 Ordering Sensitivity

**For generation**: The ordering of effects matters physically. Scatter before noise produces different results than noise before scatter (noise gets scattered if scatter comes after). The current generator enforces the correct order.

**For fitting**: The fitting order matters because each step fits residuals from previous steps. Incorrect ordering would attribute variance to the wrong physical cause. For example, fitting noise before fitting scatter would underestimate noise (some "noise" was actually scatter variability) and overestimate scatter.

**Pipeline risk**: If users construct synthesis pipelines with incorrect ordering, the results would be physically unrealistic. The ordering validation (Section 6.4.3) is essential.

### 7.5 Stochastic Steps

For stochastic steps (noise, random artifacts, batch effects):

- **fit() estimates distribution parameters**, not specific random values
- The fitted noise variance, artifact probability, etc., characterize the measurement process
- Generation samples new random realizations from these distributions
- **Validation** must compare statistical properties (variance profiles, artifact frequencies) rather than individual spectra

The conceptual difficulty: "fitting noise" sounds paradoxical. What it really means is "estimating the noise model that explains the residual variance after all deterministic effects are removed." This is a well-established concept in chemometrics (estimating measurement uncertainty from replicate measurements or from model residuals).

### 7.6 Validation Methodology

How to validate a fitted synthesis pipeline:

1. **Reconstruction quality**: For each real spectrum, the forward model (with fitted parameters) should produce a close approximation. Measure by R^2 and RMSE. (Already done by the reconstruction module.)

2. **Distribution comparison**: Synthetic and real data should have similar statistical properties. Use:
   - Per-wavelength mean and variance comparison
   - PCA score distribution overlap (Hotelling's T^2, Q residuals)
   - KS/MMD tests on principal component scores

3. **Downstream task performance**: Train a predictive model on synthetic data, evaluate on real data. If the synthesis is good, model performance should be close to training on real data.

4. **Visual inspection**: Overlay synthetic and real spectra. Check that absorption bands, baselines, and noise levels look realistic.

5. **Hold-out validation**: Fit the synthesis pipeline on a subset of real data, validate on a held-out subset. The held-out spectra should fall within the synthetic distribution.

### 7.7 Computational Cost

Pipeline-level Optuna optimization for synthesis is expensive because each trial requires:
1. Executing the full synthesis pipeline (generating N samples through all physical steps)
2. Computing distribution matching losses between synthetic and real data

For a 500-trial optimization with 1000-sample generation and Wasserstein distance computation, each trial involves O(1000 * n_wavelengths) operations for generation plus O(n_wavelengths * N * log(N)) for per-wavelength Wasserstein distance. With n_wavelengths = 751 and N = 1000, this is manageable (seconds per trial), but scales poorly with sample count.

The reconstruction module's variable projection approach is more efficient for fitting because it solves the inverse problem analytically (linear algebra) rather than by trial-and-error (Optuna). The pipeline-level Optuna approach is best suited for parameters that cannot be fitted analytically: artifact probability, environmental effect magnitudes, and other higher-order parameters.

**Recommendation**: Use the reconstruction module for fitting the core physical parameters (concentrations, path length, baseline, wavelength shift), and reserve Optuna for the secondary parameters (noise level, artifact probability, environmental effects, scatter variability) where analytical fitting is not available.

### 7.8 Comparison with Existing Approaches

**vs Reconstruction module**: The reconstruction module already provides a complete fit-and-generate pipeline. The synthesis-as-pipeline approach would reuse its algorithms but provide a more modular and composable interface. The value add is configurability (swap individual steps, use generators for parameter search) rather than algorithmic novelty.

**vs EMSC fitting**: Extended Multiplicative Scatter Correction (EMSC) fits per-sample scatter parameters. This is a subset of what the full synthesis fitting would do (scatter is just one step). The synthesis pipeline approach would subsume EMSC fitting as one step in a larger chain.

**vs Classical chemometrics**: Classical approaches (MSC, SNV, derivatives) are corrective (remove effects). The synthesis approach is generative (model effects). These are complementary: fitting the synthesis model uses classical techniques (MSC for scatter estimation, derivatives for feature detection), and the generated data can be processed by classical methods.

**vs Deep generative models** (VAE, GAN, diffusion): Deep generative models learn the full data distribution without explicit physical modeling. They can capture complex patterns but are data-hungry, lack interpretability, and may not respect physical constraints. The synthesis pipeline approach is more data-efficient, physically interpretable, and constrained. The trade-off is flexibility: deep models can capture effects not explicitly modeled. A hybrid approach (physics-informed deep generative model) would combine both.

---

## 8. Missing Steps and Requirements

### 8.1 Infrastructure Gaps in the Pipeline System

| Gap | Description | Effort |
|-----|-------------|--------|
| **Generative execution mode** | The orchestrator/executor need to support `mode="generate"` and `mode="fit_synthesis"` | Medium |
| **SeedDataset** | Factory for creating empty datasets with wavelength grids | Small |
| **No-model execution** | Pipelines currently expect a model step for metrics. Generation pipelines have no model. | Medium |
| **Synthesis result type** | A new result type for synthesis (not RunResult/PredictResult) | Small |
| **Pipeline-level finetuning** | Finetuning is currently per-model-step. Synthesis needs pipeline-level optimization. | Large |

### 8.2 Missing Controllers

| Controller | Purpose | Effort |
|-----------|---------|--------|
| `ForwardModelController` (base) | Base class for all forward model step controllers | Medium |
| `ConcentrationGeneratorController` | Generates concentration matrix, initializes dataset | Medium |
| `BeerLambertController` | Applies Beer-Lambert law (C @ E) | Small |
| `ForwardEffectController` | Handles path length, baseline, scatter, slope, shift, broadening, noise | Medium |
| `ConcentrationSolverController` | Inverse: solves for concentrations from spectra | Large (uses reconstruction algorithms) |

### 8.3 Missing Forward Model Operators

Each step needs a corresponding operator class:

| Operator | Complexity | Notes |
|----------|-----------|-------|
| `ConcentrationGenerator` | Medium | Wraps concentration generation logic from generator.py |
| `BeerLambertStep` | Small | Matrix multiply C @ E |
| `PathLengthStep` | Small | Multiplicative scaling |
| `BaselineDriftStep` | Small | Polynomial generation |
| `GlobalSlopeStep` | Small | Linear trend |
| `ScatterStep` | Small | MSC-like transformation |
| `WavelengthShiftStep` | Medium | Interpolation-based |
| `InstrumentalBroadeningStep` | Small | Gaussian convolution |
| `NoiseStep` | Small | Heteroscedastic noise addition |
| `BatchEffectStep` | Medium | Multi-batch generation |
| `ArtifactStep` | Small | Random artifacts |
| `ConcentrationSolver` | Large | Inverse problem solver |

### 8.4 Missing Loss Functions

| Loss | Purpose | Effort |
|------|---------|--------|
| `wasserstein_distance` | Distribution matching | Small (scipy has `wasserstein_distance`) |
| `ks_statistic` | Distribution matching | Small (scipy has `kstest`) |
| `mmd` | Kernel-based distribution matching | Medium (custom kernel computation) |
| `frechet_spectral_distance` | PCA-space distribution distance | Medium |
| `noise_variance_match` | Noise characterization | Small |
| `peak_statistics_match` | Spectral realism | Medium |
| `pca_coverage` | Diversity metric | Small |

### 8.5 Missing Validation Tools

| Tool | Purpose | Effort |
|------|---------|--------|
| Synthesis ordering validator | Ensure physically correct step order | Small |
| Synthesis quality dashboard | Visual comparison of real vs synthetic | Medium |
| Parameter identifiability checker | Warn about ambiguous parameters | Medium |
| Cross-session validation | Validate generalization across measurement sessions | Medium |

### 8.6 Estimated Total Scope

| Category | Items | Estimated Effort |
|----------|-------|-----------------|
| Pipeline infrastructure | 5 items | 3-4 weeks |
| Controllers | 5 controllers | 2-3 weeks |
| Operators | 12 operators | 2 weeks |
| Loss functions | 7 functions | 1-2 weeks |
| Validation tools | 4 tools | 1-2 weeks |
| Integration & testing | Full integration tests, examples | 2-3 weeks |
| Documentation | API docs, examples, design docs | 1 week |
| **Total** | | **12-17 weeks** |

---

## 9. Point of View

### 9.1 Is This a Good Idea?

**Yes, with qualifications.** The synthesis-as-pipeline concept has genuine value, but the scope and complexity are significant. The core insight -- that synthesis steps are analogous to pipeline transformations -- is sound. The ability to compose, configure, and finetune synthesis through the pipeline system would be a powerful capability.

However, the value is unevenly distributed across the operating modes:

- **Pure generation (Mode A)**: Moderate value. The current `SyntheticNIRSGenerator` already handles this well. Expressing it as a pipeline adds composability but the generator's monolithic approach is simpler and faster. The pipeline version would mainly help users who want to customize specific steps without understanding the full generator.

- **Fitting to real data (Mode B)**: High value, but the reconstruction module already provides this capability. The pipeline interface would make it more accessible and composable. However, the fitting algorithm (variable projection with multiscale scheduling) is inherently a joint optimization that does not decompose neatly into independent per-step fitting. Forcing it into a per-step pipeline may actually make it worse.

- **Transfer (Mode D)**: High value. This is the most compelling use case. Being able to fit a synthesis pipeline to one instrument and then modify specific steps for another instrument is exactly the kind of composability that pipelines provide naturally.

- **Augmentation (Mode C)**: High value, and largely achievable without the full synthesis-as-pipeline infrastructure. The augmentation operators (TemperatureAugmenter, MoistureAugmenter, etc.) already work within the pipeline system as `sample_augmentation` steps.

### 9.2 Biggest Risks

1. **Over-engineering**. Building a full pipeline infrastructure for synthesis when the reconstruction module already handles fitting and the generator already handles generation risks creating a more complex interface without sufficient benefit.

2. **Identifiability problems masquerading as configuration issues**. When users construct synthesis pipelines and fitting fails due to parameter ambiguity (Section 7.1), they will perceive it as a bug rather than a fundamental mathematical limitation. The system needs clear diagnostics.

3. **Per-step fitting does not work for the forward model**. The variable projection approach jointly optimizes linear and nonlinear parameters across the entire chain. Decomposing this into per-step fitting would lose the joint optimization and likely produce worse results. The alternative -- pipeline-level fitting that runs the entire chain in each Optuna trial -- is computationally expensive.

4. **Scope creep**. The estimated 12-17 weeks of development is substantial. Each intermediate milestone needs to deliver standalone value.

### 9.3 Biggest Opportunities

1. **Composable instrument transfer**. Fit to instrument A, swap out the noise model and wavelength shift for instrument B. This is a real unmet need in chemometric practice.

2. **Optuna-driven synthesis optimization**. Automatically tune synthesis parameters to match a target dataset. This would replace the manual parameter tuning that users currently do with `RealDataFitter` and `SyntheticDatasetBuilder`.

3. **Hybrid pipelines**. Combining synthesis (to generate training data) with model training in a single pipeline enables end-to-end optimization where the synthesis parameters are tuned to maximize downstream model performance.

4. **Systematic generation of calibration transfer datasets**. By modifying specific physical parameters (temperature, noise, scatter), users can generate targeted augmentation datasets for calibration transfer studies.

### 9.4 Pragmatic First Steps

Rather than building the full system at once, a phased approach:

**Phase 1: Forward model operators (2-3 weeks)**
Create the operator classes (BeerLambertStep, PathLengthStep, etc.) as standalone sklearn-compatible transformers that can be used both inside and outside pipelines. These operators wrap the existing logic from `generator.py`. No new controllers needed yet -- the existing `TransformerMixinController` handles sklearn transformers. The operators work on existing data (Mode C: augmentation).

**Phase 2: Pipeline generation mode (3-4 weeks)**
Add `mode="generate"` support to the pipeline system. Implement `ConcentrationGeneratorController` and `SeedDataset`. Enable pure generation through pipelines (Mode A). The existing generator becomes a preset factory (`.to_pipeline()`).

**Phase 3: Distribution matching metrics (1-2 weeks)**
Add Wasserstein, MMD, FSD metrics to `nirs4all/core/metrics.py`. These are useful independently of the synthesis pipeline (e.g., for evaluating augmentation quality, transfer learning, domain adaptation).

**Phase 4: Pipeline-level finetuning (4-5 weeks)**
Extend `OptunaManager` to support pipeline-level optimization with custom loss functions. Apply to synthesis pipeline fitting. This is the hardest piece but also the most valuable.

**Phase 5: Reconstruction integration (2-3 weeks)**
Integrate the reconstruction module as the fitting backend for Mode B. Wrap `VariableProjectionSolver` as a pipeline step. Add `mode="fit_synthesis"`.

### 9.5 How Does This Compare to the Reconstruction Module Approach?

The reconstruction module (`nirs4all/synthesis/reconstruction/`) is a self-contained pipeline that does everything: calibration, inversion, distribution learning, generation, and validation. It is scientifically rigorous, algorithmically sophisticated, and already functional.

The synthesis-as-pipeline approach would not replace the reconstruction module. Instead, it would provide:

1. **A higher-level interface** for users who want to configure generation without understanding the reconstruction internals
2. **Composability** for mixing physical steps with ML pipeline steps
3. **Optuna integration** for automated parameter tuning
4. **Generator syntax** for exploring parameter spaces
5. **Branching** for generating multi-configuration datasets

The reconstruction module would remain the workhorse for the actual fitting algorithms. The pipeline approach is a *configuration and composition layer* on top of the reconstruction module, not a replacement for it.

### 9.6 Trade-offs

| Dimension | Current System | Synthesis-as-Pipeline |
|-----------|---------------|----------------------|
| **Simplicity** | Generator is a single call: `generate(n_samples=1000)` | Pipeline requires multiple steps, seed dataset, mode flags |
| **Performance** | Direct numpy operations, optimized loops | Pipeline overhead (context management, controller dispatch) |
| **Flexibility** | Limited: configure through constructor params and complexity presets | High: swap, reorder, branch, merge, finetune individual steps |
| **Fitting** | Reconstruction module: joint optimization, multiscale scheduling | Pipeline fitting: either per-step (lossy) or pipeline-level (expensive) |
| **Composability** | Separate: generate first, then train | Unified: generate + train in one pipeline |
| **Learning curve** | Low: one class, one method | Higher: new operators, modes, and concepts |
| **Maintainability** | One generator module to maintain | Many operator classes + controllers + infrastructure |

The fundamental tension is between **simplicity** (the current generator is straightforward) and **composability** (the pipeline approach enables powerful new workflows). The pragmatic path is to build the composability features incrementally, keeping the simple interface as the default, and exposing the pipeline interface for advanced use cases.
