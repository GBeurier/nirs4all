# Synthetic NIRS Data Generator: Vision & Roadmap

## Philosophy: Physics-First Generation

The fundamental principle is to generate spectra **the way nature creates them**, then simulate how our imperfect measurement systems observe them:

```
COMPOSITION → IDEAL SPECTRUM → SENSOR → ENVIRONMENT → ARTIFACTS → [PREPROCESSING] → OBSERVED SPECTRUM
     ↓                                                                                      ↓
  PROPORTIONS ─────────────────── TARGET FUNCTION ────────────────────────────────────→ TARGETS
```

### Generation Pipeline (Conceptual)

1. **Component Mix (Ground Truth)**
   - Define proportions of spectral components (covalent bonds, functional groups, aggregates)
   - Each component contributes bands following Beer-Lambert law: A = εcl
   - Result: a "perfect" theoretical absorption profile

2. **Sensor Simulation**
   - Sampling grid (wavelength resolution, range)
   - Multi-sensor stitching (e.g., Si + InGaAs detectors)
   - Signal type transformation (absorbance ↔ transmittance ↔ reflectance ↔ transflectance)
   - Detector response curves and roll-off

3. **Environmental Effects**
   - Temperature (peak shifts, broadening)
   - Moisture/humidity (free vs. bound water)
   - Particle size/granulometry (scattering)
   - Matrix effects

4. **Noise & Artifacts**
   - Detector noise (shot, thermal, 1/f)
   - Baseline drift (linear, polynomial)
   - Edge artifacts (stray light, truncation)
   - Batch effects (session-to-session variation)

5. **Preprocessing (Optional)**
   - Some real datasets come pre-processed
   - Apply SNV, MSC, derivatives, normalization as needed

6. **Target Generation**
   - From initial proportions, generate targets via:
     - Simple linear relationships (c → y)
     - Non-linear transformations (polynomial, log)
     - Noised relationships (measurement error)
     - Multi-target scenarios

---

## Current State: What nirs4all Already Has

### Spectral Components (126 total)
- **Proteins** (12): casein, gluten, albumin, keratin, etc.
- **Lipids** (20): fatty acids, phospholipids, petroleum products
- **Carbohydrates** (18): starch, cellulose, sugars
- **Water/Alcohols** (11): moisture, ethanol, glycerol
- **Pigments** (18): chlorophylls, carotenoids, hemoglobin
- **Polymers** (11): PE, PS, PP, PVC, PET
- **Pharmaceuticals** (10): caffeine, aspirin, paracetamol
- **Minerals** (8): carbonates, clays, silica

### Sensor Simulation
- 20+ instrument archetypes (benchtop, handheld, process, embedded)
- Detector types: Si, InGaAs, Extended InGaAs, PbS, PbSe, MCT
- Multi-sensor stitching with configurable overlap
- Multi-scan averaging with realistic noise reduction

### Measurement Modes
- **Transmittance**: Beer-Lambert law, configurable path length
- **Diffuse Reflectance**: Kubelka-Munk theory, integrating sphere
- **Transflectance**: Double-pass geometry
- **ATR**: Evanescent wave, penetration depth

### Environmental Effects
- Temperature: literature-based shift/broadening parameters (Maeda 1995, Segtnan 2001)
- Moisture: free vs. bound water differentiation
- Scattering: EMSC, Rayleigh, Mie approximation, Kubelka-Munk

### Noise & Artifacts
- Gaussian additive, multiplicative, shot, thermal, 1/f noise
- Baseline drift (linear, polynomial)
- Edge effects (detector roll-off, stray light)
- Wavelength distortions (shift, stretch)

### Target Generation
- Dirichlet-sampled proportions
- Linear and non-linear relationships
- Polynomial interactions, synergistic/antagonistic effects
- Classification via component-based class separation

---

## The Three Evaluation Challenges

### Challenge 1: Statistical Expressiveness
> Can we automatically generate data (mean + variance) identical to real data?

**Question**: Is the generator expressive enough to reproduce any real dataset's distribution?

**Approach**:
1. Take a real dataset (X_real, y_real)
2. Use `RealDataFitter` to infer generation parameters:
   - Component composition
   - Instrument type
   - Environmental conditions
   - Noise levels
3. Generate synthetic dataset with inferred parameters
4. Compare distributions:
   - Wasserstein distance per wavelength
   - PCA space overlap
   - Derivative statistics
   - SNR distribution

**Success Metric**: Wasserstein distance < threshold, KS-test p > 0.05

### Challenge 2: Transfer Learning
> Can we train on synthetic and predict on real with <50% error increase?

**Question**: Does synthetic data capture enough "real" properties for model transfer?

**Approach**:
1. Fit generator to real dataset statistics
2. Generate large synthetic training set (10x real samples)
3. Train model on synthetic data
4. Evaluate on real test set
5. Compare to model trained on real data

**Success Metric**: RMSE_synthetic_trained / RMSE_real_trained < 1.5

### Challenge 3: Discriminator Test
> Can we fool a classifier that tries to distinguish real from synthetic?

**Question**: Are synthetic spectra visually/statistically indistinguishable from real?

**Approach**:
1. Combine real and synthetic datasets
2. Train discriminator (RF, XGBoost, or MLP) on (X, label_is_real)
3. Measure AUC on held-out test set
4. Target: AUC ≈ 0.5 (random guessing)

**Success Metric**: Adversarial AUC < 0.6 (close to random)

---

## Roadmap: Making It Work

### Phase A: Fitter Enhancement (Foundation)

**Goal**: Robust inference of generation parameters from real data.

| Task | Status | Description |
|------|--------|-------------|
| A.1 | ✅ Done | Component identification via spectral fitting |
| A.2 | ✅ Done | Instrument inference (detector type, range) |
| A.3 | ✅ Done | Measurement mode detection |
| A.4 | ⬜ TODO | Noise model inference (SNR decomposition) |
| A.5 | ⬜ TODO | Environmental parameter inference |
| A.6 | ⬜ TODO | Preprocessing detection and compensation |

**Key Files**: `fitter.py`, `validation.py`

### Phase B: Generation Pipeline Refinement

**Goal**: Clean, composable generation workflow.

| Task | Status | Description |
|------|--------|-------------|
| B.1 | ⬜ TODO | Unified `generate_from_real()` function |
| B.2 | ⬜ TODO | Parameter sweep for best fit |
| B.3 | ✅ Done | Multi-source generation |
| B.4 | ⬜ TODO | Consistent random state management |
| B.5 | ⬜ TODO | Generation reproducibility tests |

**Proposed API**:
```python
import nirs4all

# Fit parameters from real data
params = nirs4all.synthetic.fit(real_dataset)

# Generate matching synthetic data
synthetic = nirs4all.generate(
    n_samples=5000,
    from_params=params,
    random_state=42
)

# Or one-liner
synthetic = nirs4all.generate.like(real_dataset, n_samples=5000)
```

### Phase C: Statistical Validation Suite

**Goal**: Automated comparison between synthetic and real data.

| Task | Status | Description |
|------|--------|-------------|
| C.1 | ✅ Partial | Per-wavelength distribution comparison |
| C.2 | ⬜ TODO | PCA space overlap metrics |
| C.3 | ⬜ TODO | Spectral autocorrelation matching |
| C.4 | ⬜ TODO | Target distribution matching |
| C.5 | ⬜ TODO | Comprehensive comparison report |

**Proposed API**:
```python
from nirs4all.data.synthetic import compare_datasets

report = compare_datasets(
    real=real_dataset,
    synthetic=synthetic_dataset,
    metrics=['wasserstein', 'pca_overlap', 'autocorr', 'target_dist']
)
report.summary()  # Text report
report.plot()     # Visual comparison
```

### Phase D: Transfer Learning Evaluation

**Goal**: Measure predictive value of synthetic data.

| Task | Status | Description |
|------|--------|-------------|
| D.1 | ⬜ TODO | Train-on-synthetic evaluation framework |
| D.2 | ⬜ TODO | Error ratio calculation |
| D.3 | ⬜ TODO | Domain adaptation metrics |
| D.4 | ⬜ TODO | Pre-training benefit measurement |

**Proposed API**:
```python
from nirs4all.data.synthetic import evaluate_transfer

result = evaluate_transfer(
    synthetic_train=synthetic_data,
    real_test=real_data,
    pipeline=[SNV(), PLSRegression(10)],
    baseline='real_cv'  # Compare to real data cross-validation
)
print(f"Error ratio: {result.error_ratio:.2f}")
print(f"Transfer efficiency: {result.efficiency:.1%}")
```

### Phase E: Discriminator Evaluation

**Goal**: Adversarial validation of realism.

| Task | Status | Description |
|------|--------|-------------|
| E.1 | ⬜ TODO | Discriminator training framework |
| E.2 | ⬜ TODO | AUC calculation with confidence intervals |
| E.3 | ⬜ TODO | Feature importance for realism gaps |
| E.4 | ⬜ TODO | Iterative improvement feedback loop |

**Proposed API**:
```python
from nirs4all.data.synthetic import adversarial_validation

result = adversarial_validation(
    real=real_dataset,
    synthetic=synthetic_dataset,
    classifier='xgboost',  # or 'rf', 'mlp'
    n_folds=5
)
print(f"Discriminator AUC: {result.auc:.3f} (target: ~0.5)")
print(f"Top distinguishing features: {result.feature_importance[:5]}")
```

### Phase F: Benchmark Suite

**Goal**: Standardized evaluation across multiple real datasets.

| Task | Status | Description |
|------|--------|-------------|
| F.1 | ✅ Partial | Benchmark dataset registry (8 datasets) |
| F.2 | ⬜ TODO | Automated evaluation pipeline |
| F.3 | ⬜ TODO | Leaderboard/scorecard generation |
| F.4 | ⬜ TODO | Per-domain performance analysis |

**Proposed API**:
```python
from nirs4all.data.synthetic import run_benchmark

results = run_benchmark(
    datasets=['corn', 'tecator', 'shootout2002', 'diesel'],
    evaluations=['statistical', 'transfer', 'discriminator']
)
results.to_csv('benchmark_results.csv')
results.plot_summary()
```

---

## Implementation Priority

### Sprint 1: Core Infrastructure
1. **A.4**: Noise model inference
2. **A.5**: Environmental parameter inference
3. **B.1**: `generate_from_real()` function
4. **C.1-C.2**: Basic statistical comparison

### Sprint 2: Transfer Learning
1. **D.1-D.2**: Transfer evaluation framework
2. **D.3**: Domain adaptation metrics
3. **B.4-B.5**: Reproducibility

### Sprint 3: Discriminator & Benchmark
1. **E.1-E.2**: Discriminator framework
2. **E.3**: Feature importance for gaps
3. **F.2-F.3**: Automated benchmark pipeline

### Sprint 4: Polish & Feedback Loop
1. **E.4**: Iterative improvement
2. **F.4**: Domain analysis
3. Documentation and examples

---

## Success Criteria (Definition of Done)

### Statistical Expressiveness
- [ ] Reproduce Corn dataset with Wasserstein < 0.1
- [ ] Reproduce Tecator dataset with Wasserstein < 0.1
- [ ] PCA space overlap > 80% for all benchmark datasets

### Transfer Learning
- [ ] Error ratio < 1.5 on at least 5/8 benchmark datasets
- [ ] Error ratio < 1.3 on at least 3/8 benchmark datasets
- [ ] Pre-training on synthetic improves real data performance

### Discriminator
- [ ] AUC < 0.6 on at least 5/8 benchmark datasets
- [ ] AUC < 0.55 on at least 3/8 benchmark datasets

### Overall
- [ ] All three evaluations pass on at least one domain (proof of concept)
- [ ] Documented workflow for users to evaluate their own datasets
- [ ] Published benchmark results with reproducible code

---

## Appendix: Existing Relevant Code

### Generation
- `nirs4all/data/synthetic/generator.py` - Core generator
- `nirs4all/data/synthetic/components.py` - Spectral components
- `nirs4all/data/synthetic/builder.py` - Fluent API
- `nirs4all/data/synthetic/targets.py` - Target generation

### Fitting & Inference
- `nirs4all/data/synthetic/fitter.py` - Parameter inference from real data
- `nirs4all/data/synthetic/validation.py` - Realism metrics

### Effects & Augmentation
- `nirs4all/operators/augmentation/` - All augmentation operators
- `nirs4all/data/synthetic/instruments.py` - Instrument simulation
- `nirs4all/data/synthetic/environmental.py` - Environmental effects
- `nirs4all/data/synthetic/scattering.py` - Scattering effects

### Benchmarks
- `nirs4all/data/synthetic/benchmarks.py` - Benchmark dataset registry
