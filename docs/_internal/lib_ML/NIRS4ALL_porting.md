# NIRS4ALL Porting Investigation

> Analysis of the work required to port nirs4all onto a generic ML pipeline library (pipeforge), re-including NIRS-specific operators as a domain plugin.

---

## 1. Porting Strategy Overview

The porting transforms nirs4all from a monolithic NIRS library into two packages:

```
BEFORE:                              AFTER:
nirs4all (monolith)                  pipeforge (generic core)
├── pipeline/  ─────────────────→    ├── pipeline/        (moved as-is, ~85%)
├── controllers/ ───────────────→    ├── controllers/      (moved as-is, ~100%)
├── data/ ──────────────────────→    ├── data/            (refactored, ~60%)
├── operators/ ─────────────────→    │
├── api/ ───────────────────────→    ├── api/             (moved, renamed)
├── sklearn/ ───────────────────→    ├── sklearn/         (moved, renamed)
├── visualization/ ─────────────→    ├── visualization/   (moved, ~80%)
└── config/ ────────────────────→    └── config/          (moved as-is)

                                     pipeforge-nirs (domain plugin)
                                     ├── operators/       (NIRS transforms, models, augmentation)
                                     ├── controllers/     (chart controllers, resampler)
                                     ├── data/            (SignalType, SpectralSource, wavelengths)
                                     ├── splitters/       (KennardStone, SPXY, etc.)
                                     └── filters/         (SpectralQualityFilter)
```

---

## 2. Component-by-Component Analysis

### 2.1 Pipeline Engine → Core (minimal changes)

| File/Module | Lines (approx) | Action | Changes Needed |
|------------|----------------|--------|---------------|
| `pipeline/runner.py` | 764 | Move to core | Remove `report_naming` NIRS toggle; use plugin-configurable naming |
| `pipeline/execution/orchestrator.py` | 2116 | Move to core | Remove `report_naming` parameter; generalize dataset normalization |
| `pipeline/execution/executor.py` | 400 | Move to core | None — already generic |
| `pipeline/execution/builder.py` | ~150 | Move to core | None |
| `pipeline/execution/step_cache.py` | ~200 | Move to core | None |
| `pipeline/execution/refit/` | ~800 | Move to core | None — all refit strategies are generic |
| `pipeline/steps/` | ~580 | Move to core | None — step parsing and routing are generic |
| `pipeline/config/` | ~1000 | Move to core | None — variant generation is generic |
| `pipeline/trace/` | ~500 | Move to core | None |
| `pipeline/storage/` | ~1500 | Move to core | None — DuckDB+Parquet storage is generic |
| `pipeline/bundle/` | ~600 | Move to core | Change `.n4a` → `.pfb` extension; remove NIRS version metadata |
| `pipeline/predictor.py` | ~400 | Move to core | None |
| `pipeline/minimal_predictor.py` | ~300 | Move to core | None |
| `pipeline/explainer.py` | ~200 | Move to core | Remove wavelength-based feature labels |
| `pipeline/retrainer.py` | ~250 | Move to core | None |

**Estimated effort: Low** — mostly file moves with minor string/naming changes.

### 2.2 Controller System → Core (no changes)

| Controller | Action | Changes Needed |
|-----------|--------|---------------|
| `OperatorController` base | Move to core | None |
| `registry.py` | Move to core | Add plugin controller registration API |
| `TransformerMixinController` | Move to core | Remove wavelength extraction (move to NIRS plugin override or hook) |
| `YTransformerMixinController` | Move to core | None |
| `SklearnModelController` | Move to core | None |
| `TensorFlowModelController` | Move to core | None |
| `PyTorchModelController` | Move to core | None |
| `JaxModelController` | Move to core | None |
| `CrossValidatorController` | Move to core | None |
| `BranchController` | Move to core | None |
| `MergeController` | Move to core | None |
| `TagController` | Move to core | None |
| `ExcludeController` | Move to core | None |
| `DummyController` | Move to core | None |
| `SampleAugmentationController` | Move to core | Make source-type agnostic (currently assumes numpy arrays) |
| `FeatureAugmentationController` | Move to core | Same as above |
| `ResamplerController` | **Move to NIRS plugin** | Wavelength-specific logic |
| `SpectraChartController` | **Move to NIRS plugin** | Spectroscopy visualization |
| `FoldChartController` | Move to core | None — generic fold distribution chart |
| `YChartController` | Move to core | None — generic target distribution chart |
| `AugmentationChartController` | Move to core | None — generic augmentation comparison |
| `SpectralDistributionController` | **Move to NIRS plugin** | Spectral distribution visualization |
| `AutoTransferPreprocessingController` | Move to core | None — generic auto-transfer logic |

**Estimated effort: Low** — 3 controllers move to plugin, rest are already generic.

### 2.3 Data Model → Major Refactoring Required

This is the **highest-effort** area. `SpectroDataset` is deeply coupled to NIRS concepts.

#### What stays in core

| Component | Current Location | Changes Needed |
|----------|-----------------|---------------|
| `Indexer` | `data/_dataset/indexer.py` (or similar) | None — already generic (partitions, folds, exclusions, tags) |
| `Targets` | `data/targets.py` | Minor — remove spectroscopy-specific processing names |
| `Metadata` | `data/metadata.py` | None — already generic |
| `Predictions` | `data/predictions.py` | None — already generic |
| Loaders (CSV, Parquet, NumPy, Excel) | `data/loaders/` | None — already generic |
| `DatasetConfigs` | `data/config.py` | Remove signal_type, header_unit, wavelength-specific config keys |

#### What moves to NIRS plugin

| Component | Current Location | Plugin Location |
|----------|-----------------|----------------|
| `SignalType` enum | `data/signal_type.py` | `pipeforge_nirs/data/signal_type.py` |
| Signal type detection | `data/signal_type.py` | `pipeforge_nirs/data/detection.py` |
| Wavelength handling | Spread across `Features`, dataset accessors | `pipeforge_nirs/data/spectral_source.py` |
| Signal conversion | `operators/transforms/signal_conversion.py` | `pipeforge_nirs/operators/transforms/` |
| Header unit parsing | `data/loaders/` (nm, cm⁻¹ detection) | `pipeforge_nirs/data/loaders.py` |
| MATLAB loader | `data/loaders/matlab.py` | Can stay in core (generic) or move to NIRS plugin |

#### SpectroDataset → Dataset refactoring

The key challenge: `SpectroDataset` is used as the universal data container throughout the engine. Every controller, every executor, every step runner receives a `SpectroDataset`.

**Refactoring approach:**

1. Create generic `Dataset` class with source registry pattern
2. `SpectroDataset` becomes a thin subclass or factory:
   ```python
   class SpectroDataset(Dataset):
       """NIRS-specific dataset with wavelength and signal type support."""

       def __init__(self, ...):
           super().__init__(...)
           # Register spectral source type

       @property
       def wavelengths(self):
           return self.get_source("spectra").wavelengths

       def detect_signal_type(self, src=0):
           return self.get_source_by_index(src).detect_signal_type()

       def convert_to_absorbance(self, src=0):
           return self.get_source_by_index(src).convert_to_absorbance()
   ```

3. All engine code references `Dataset` (base class) instead of `SpectroDataset`
4. NIRS plugin registers `SpectralSource` type that adds wavelength/signal_type metadata

**Estimated effort: HIGH** — requires touching every file that references `SpectroDataset` and its NIRS-specific methods. This is the critical path.

### 2.4 Operators → NIRS Plugin (move only)

| Category | Files (approx) | Action |
|----------|----------------|--------|
| **NIRS Transforms** | `operators/transforms/nirs.py`, `scalers.py`, `signal.py`, `signal_conversion.py`, `orthogonalization.py`, `feature_selection.py`, `resampler.py` | Move to NIRS plugin |
| **Spectral Augmentation** | `operators/augmentation/spectral.py`, `synthesis.py`, `splines.py`, `edge_artifacts.py`, `environmental.py`, `scattering.py`, `random.py` | Move to NIRS plugin |
| **PLS Models** | `operators/models/sklearn/` (all PLS variants, AOM-PLS, POP-PLS) | Move to NIRS plugin |
| **NICON DL Models** | `operators/models/tensorflow/nicon/`, `pytorch/nicon/` | Move to NIRS plugin |
| **NIRS Splitters** | `operators/splitters/splitters.py` (KennardStone, SPXY, etc.) | Move to NIRS plugin |
| **Spectral Quality Filter** | `operators/filters/spectral_quality.py` | Move to NIRS plugin |
| **High Leverage Filter** | `operators/filters/high_leverage.py` | Can stay in core (generic) |

#### What stays in core

| Component | Reason |
|----------|--------|
| `SampleFilter` base class | Generic filtering interface |
| `CompositeFilter` | Generic filter composition |
| `YOutlierFilter` | Generic (works on any y values) |
| `XOutlierFilter` | Generic (works on any feature matrix) |
| `MetadataFilter` | Generic |
| `MetaModel` / `StackingConfig` | Generic stacking infrastructure |
| Generic DL model controllers | Framework integration (TF, PyTorch, JAX) |

**Estimated effort: Low** — mostly file moves, no logic changes. Operators are already self-contained.

### 2.5 API Layer → Core (minor renaming)

| Function | Changes Needed |
|----------|---------------|
| `run()` | Rename package references; remove NIRS defaults |
| `predict()` | Change bundle extension |
| `explain()` | Remove wavelength feature labels (move to plugin) |
| `retrain()` | None |
| `session()` | Rename |
| `generate()` | **Move to NIRS plugin** — synthetic data is spectroscopy-specific |
| `RunResult` | Remove NIRS metric names (RPD, bias) from defaults |

**Estimated effort: Low**

### 2.6 sklearn Wrapper → Core (renaming)

| Component | Changes |
|----------|---------|
| `NIRSPipeline` | Rename to `PipelineWrapper` |
| `NIRSPipelineClassifier` | Rename to `ClassifierWrapper` |

**Estimated effort: Trivial**

### 2.7 Visualization → Mostly Core

| Component | Action |
|----------|--------|
| `PredictionAnalyzer` | Move to core — generic prediction charts |
| `TopKComparisonChart` | Move to core |
| `ConfusionMatrixChart` | Move to core |
| `HeatmapChart` | Move to core |
| `CandlestickChart` | Move to core |
| `ScoreHistogramChart` | Move to core |
| Spectral plots | **Move to NIRS plugin** |
| Wavelength-axis charts | **Move to NIRS plugin** |

**Estimated effort: Low**

---

## 3. NIRS Plugin Structure

```
pipeforge-nirs/
├── pyproject.toml                    # Entry point: [project.entry-points."pipeforge.plugins"]
├── pipeforge_nirs/
│   ├── __init__.py                   # NIRSPlugin class, auto-registration
│   ├── plugin.py                     # Plugin definition (register controllers, source types, operators)
│   │
│   ├── data/
│   │   ├── spectral_source.py        # SpectralSource (extends pipeforge.data.Source)
│   │   ├── spectro_dataset.py        # SpectroDataset (extends pipeforge.Dataset) — convenience class
│   │   ├── signal_type.py            # SignalType enum, detection, conversion
│   │   └── wavelength.py             # Wavelength utilities (nm, cm⁻¹, unit conversion)
│   │
│   ├── operators/
│   │   ├── transforms/
│   │   │   ├── nirs.py               # SavitzkyGolay, Haar, WaveletDenoise, MSC, derivatives, baselines
│   │   │   ├── scalers.py            # SNV, RNV, LocalSNV, Normalize, Derivate
│   │   │   ├── signal.py             # Baseline, Detrend, Gaussian
│   │   │   ├── signal_conversion.py  # ToAbsorbance, FromAbsorbance, KubelkaMunk
│   │   │   ├── orthogonalization.py  # OSC, EPO
│   │   │   ├── feature_selection.py  # CARS, MCUVE
│   │   │   └── resampler.py          # Resampler (wavelength interpolation)
│   │   │
│   │   ├── models/
│   │   │   ├── pls_variants/         # PLSDA, IKPLS, OPLS, MBPLS, DiPLS, SparsePLS, etc.
│   │   │   ├── aom_pls.py            # AOMPLSRegressor, AOMPLSClassifier
│   │   │   ├── pop_pls.py            # POPPLSRegressor, POPPLSClassifier
│   │   │   ├── advanced/             # FCKPLS, OKLMPLS, KernelPLS, NLPLS
│   │   │   └── deep_learning/        # NICON (TF, PyTorch, JAX)
│   │   │
│   │   ├── augmentation/
│   │   │   ├── spectral.py           # Noise, baseline drift, wavelength warping, band manipulation
│   │   │   ├── synthesis.py          # PathLength, BatchEffect, InstrumentalBroadening
│   │   │   ├── splines.py            # Spline-based augmentation
│   │   │   ├── edge_artifacts.py     # DetectorRollOff, StrayLight, etc.
│   │   │   ├── environmental.py      # Temperature, Moisture effects
│   │   │   └── scattering.py         # ParticleSize, EMSC distortion
│   │   │
│   │   ├── splitters/
│   │   │   └── splitters.py          # KennardStone, SPXY, SPXYFold, KMeans, etc.
│   │   │
│   │   └── filters/
│   │       ├── spectral_quality.py   # SpectralQualityFilter
│   │       └── high_leverage.py      # HighLeverageFilter
│   │
│   ├── controllers/
│   │   ├── resampler.py              # ResamplerController
│   │   ├── spectra_chart.py          # SpectraChartController
│   │   └── spectral_distribution.py  # SpectralDistributionController
│   │
│   ├── visualization/
│   │   └── spectral_plots.py         # Wavelength-axis charts, spectral overlays
│   │
│   └── generate.py                   # nirs4all.generate() → pipeforge_nirs.generate()
```

### Plugin Registration (pyproject.toml)

```toml
[project.entry-points."pipeforge.plugins"]
nirs = "pipeforge_nirs:NIRSPlugin"
```

### User Experience After Porting

```python
# Install: pip install pipeforge pipeforge-nirs
import pipeforge
from pipeforge_nirs.operators.transforms import SNV, MSC, SavitzkyGolay
from pipeforge_nirs.operators.models import AOMPLSRegressor
from pipeforge_nirs.data import SpectroDataset

# Load NIRS data (SpectroDataset adds wavelength/signal type support)
ds = SpectroDataset.from_csv("spectra.csv", target_column="protein")
ds.detect_signal_type()
ds.convert_to_absorbance()

# Build pipeline — identical syntax to current nirs4all
result = pipeforge.run(
    pipeline=[
        SNV(),
        SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        ShuffleSplit(n_splits=5),
        {"model": AOMPLSRegressor(n_components=15)},
    ],
    dataset=ds,
    verbose=1,
)
```

---

## 4. Migration Path

### Phase 1: Prepare nirs4all (2-3 weeks)

**Goal:** Isolate NIRS-specific code without changing behavior.

1. **Introduce `Dataset` base class** — extract generic interface from `SpectroDataset`
   - Create `Dataset` with source registry, indexer, targets, metadata
   - Make `SpectroDataset` inherit from `Dataset`
   - All engine code should work with `Dataset` type hints (use `SpectroDataset` only for NIRS-specific operations)

2. **Separate operator imports** — ensure NIRS operators are imported only from `operators/` subdirectories, not from engine code
   - Audit all imports in `pipeline/`, `controllers/`, `api/`
   - Remove any inline NIRS operator references from engine code

3. **Extract wavelength handling** — move wavelength-specific logic out of `TransformerMixinController` into a hook or NIRS-specific controller override

4. **Generalize report naming** — replace hardcoded NIRS terminology with configurable labels

### Phase 2: Extract Core (3-4 weeks)

**Goal:** Create the pipeforge package from nirs4all internals.

1. **Create pipeforge package** — new repository with package structure from ML_lib_design.md
2. **Move pipeline engine** — `pipeline/`, `controllers/` (generic ones), `config/`
3. **Move data core** — `Dataset`, `Indexer`, `Targets`, `Metadata`, `Predictions`, loaders
4. **Move API** — `run()`, `predict()`, `explain()`, `retrain()`, `session()`
5. **Move storage** — `WorkspaceStore`, `ArrayStore`, bundle system
6. **Move visualization** — `PredictionAnalyzer`, generic charts
7. **Add plugin infrastructure** — `PluginBase`, entry point discovery, registry extension
8. **Write tests** — port existing generic tests, add plugin integration tests

### Phase 3: Create NIRS Plugin (2-3 weeks)

**Goal:** Package NIRS-specific operators as `pipeforge-nirs`.

1. **Move operators** — all NIRS transforms, models, augmentation, splitters, filters
2. **Move controllers** — spectral chart, resampler, spectral distribution
3. **Create SpectroDataset** — thin wrapper over `Dataset` with NIRS source types
4. **Create SpectralSource** — source type with wavelength/signal_type metadata
5. **Move visualization** — spectral plots, wavelength charts
6. **Move generate()** — synthetic data generation
7. **Write plugin registration** — `NIRSPlugin` class, pyproject.toml entry point
8. **Write tests** — port NIRS-specific tests

### Phase 4: Port nirs4all (1-2 weeks)

**Goal:** Make nirs4all a thin wrapper around `pipeforge + pipeforge-nirs`.

```python
# nirs4all/__init__.py (after porting)
import pipeforge
import pipeforge_nirs  # Triggers plugin auto-registration

# Re-export the API under nirs4all namespace
run = pipeforge.run
predict = pipeforge.predict
explain = pipeforge.explain
retrain = pipeforge.retrain
session = pipeforge.session

# Re-export NIRS-specific classes
from pipeforge_nirs.data import SpectroDataset
from pipeforge_nirs.operators.transforms import SNV, MSC, SavitzkyGolay, ...
from pipeforge_nirs.operators.models import AOMPLSRegressor, POPPLSRegressor, ...
from pipeforge_nirs.generate import generate
```

Alternatively, nirs4all could be **deprecated** in favor of `pipeforge + pipeforge-nirs` directly.

---

## 5. Effort Estimation

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: Prepare nirs4all | 2-3 weeks | Medium — SpectroDataset refactoring is the bottleneck |
| Phase 2: Extract core | 3-4 weeks | Medium — many files to move, test coverage to maintain |
| Phase 3: NIRS plugin | 2-3 weeks | Low — operators are self-contained |
| Phase 4: Port nirs4all | 1-2 weeks | Low — thin wrapper or deprecation |
| **Total** | **8-12 weeks** | |

### Critical Path
1. **SpectroDataset → Dataset abstraction** — everything depends on this
2. **Controller wavelength decoupling** — `TransformerMixinController` passes wavelengths to operators
3. **Test infrastructure** — maintaining test coverage across the split

### Risk Factors
- **API breakage** — users of nirs4all will need to update imports if not using the thin wrapper approach
- **Test duplication** — some tests will need to exist in both pipeforge (generic) and pipeforge-nirs (domain-specific)
- **Performance regression** — plugin indirection could add overhead (likely negligible)
- **Dependency management** — two packages to version and release in sync

---

## 6. Key Refactoring Hotspots

### 6.1 SpectroDataset.x() method

This is the most-called method in the engine. Currently returns NIRS-specific array with wavelength awareness:

```python
# Current: SpectroDataset.x(selector) handles:
# - Multi-source concatenation
# - Per-source processing chains
# - Wavelength-based feature ordering
# - Signal type propagation
# - Augmented sample inclusion/exclusion
# - Tag-based filtering
# - Fold-based partitioning
```

**Refactoring:** Split into generic `Dataset.X(selector)` (handles partitioning, folds, exclusions, tags, multi-source) and source-specific data retrieval via `Source.get_data()`.

### 6.2 TransformerMixinController wavelength injection

Currently extracts wavelengths from dataset and passes to operators that need them:

```python
# Current behavior in TransformerMixinController.execute():
if hasattr(operator, 'wavelengths') and operator.wavelengths is None:
    operator.wavelengths = dataset.wavelengths
```

**Refactoring options:**
1. **Hook-based:** Plugin registers a pre-execute hook that injects source metadata
2. **Source-aware operators:** Operators declare what source metadata they need; the controller provides it from the source registry
3. **NIRS controller override:** NIRS plugin provides `NIRSTransformerController` that extends `TransformerMixinController` with wavelength injection (higher priority)

**Recommended: Option 2** — cleanest separation, no special-casing.

### 6.3 Orchestrator dataset normalization

`PipelineOrchestrator._normalize_datasets()` currently handles NIRS-specific config keys (signal_type, header_unit, wavelengths):

**Refactoring:** Dataset normalization becomes a two-step process:
1. Core normalization (paths, train/test split, task_type)
2. Plugin-specific normalization via registered hooks (signal_type, header_unit)

### 6.4 Report/metric naming

Currently uses NIRS chemometrics terminology in some places (RMSE, RPD, bias, slope):

**Refactoring:** Use generic ML metric names by default; allow plugins to register alternative display names for domain-specific reports.

---

## 7. Backward Compatibility Strategy

### Option A: Thin Wrapper (Recommended)

nirs4all becomes `pipeforge + pipeforge-nirs` bundled under the old namespace:
- **Zero breaking changes** for existing users
- nirs4all pip package depends on both pipeforge and pipeforge-nirs
- All existing imports continue to work
- Gradual deprecation of nirs4all-specific imports

### Option B: Clean Break

nirs4all is archived; users migrate to `pipeforge + pipeforge-nirs`:
- Breaking change for all users
- Cleaner long-term maintainability
- Migration guide + codemod tool to update imports

### Option C: Hybrid

nirs4all v1.x remains as-is; pipeforge is the v2 rewrite:
- No disruption to existing users
- Duplicated maintenance effort during transition
- Clear migration point for new projects

**Recommendation:** Option A for initial release, transition to Option B after 6-12 months.

---

## 8. What nirs4all Gains from the Port

1. **Community contributions** — generic ML users can contribute to the engine
2. **Better tested core** — more diverse usage patterns expose edge cases
3. **Cleaner architecture** — forced separation improves modularity
4. **Plugin ecosystem** — NLP, vision, tabular plugins benefit NIRS users (e.g., NLP augmentation for spectral descriptions, image processing for hyperspectral data)
5. **Reduced maintenance** — core engine maintained by broader community
6. **Credibility** — "built on pipeforge" vs "NIRS-specific library" appeals to broader audience

### What nirs4all Loses

1. **Single-package simplicity** — `pip install nirs4all` becomes `pip install pipeforge pipeforge-nirs` (mitigated by metapackage)
2. **Release coupling** — core and plugin versions must be coordinated
3. **Direct control** — core engine changes require upstream coordination
4. **Performance tuning** — optimizations specific to NIRS data patterns may be harder to implement in generic core

---

## 9. Alternative: Incremental Extraction

Instead of a full port, incrementally extract generic components:

1. **Phase A:** Extract `pipeline/config/_generator/` as standalone `pipeforge-generators` package
2. **Phase B:** Extract `pipeline/storage/` as standalone `pipeforge-store` package
3. **Phase C:** Extract controller registry as standalone `pipeforge-controllers` package
4. **Phase D:** Extract execution engine as standalone `pipeforge-engine` package
5. **Phase E:** Compose into unified `pipeforge` package

This reduces risk but increases complexity and coordination overhead. **Not recommended** for a team of 1-3 developers — full extraction is more efficient.
