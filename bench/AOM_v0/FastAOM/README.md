# FastAOM — Fast multi-operator AOM PLS-Ridge for NIRS

Experimental framework that explores a very large space of preprocessing chains
without naively fitting one model per chain. Built as a sibling of
`bench/AOM_v0/aompls` and reuses its `LinearSpectralOperator`, `ComposedOperator`,
bank presets, NIPALS/SIMPLS engines, and metrics module.

The framework lives entirely in this directory:

```
bench/AOM_v0/FastAOM/
├── __init__.py
├── operator_chain.py        # N-deep OperatorChain with simplify + signature
├── bases.py                 # Nonlinear bases: raw, absorbance, SNV, MSC, EMSC, ASLS, Whittaker
├── grammar.py               # Typed chain grammar (baseline / smoothing / derivative / projection)
├── chain_generator.py       # DFS chain enumeration with grammar pruning
├── lowrank.py               # Truncated SVD per base, kernel-vector products
├── screening.py             # Fast covariance screen + diversity-aware top-k
├── models/
│   ├── _common.py           # PLS-NIPALS, Ridge-on-scores, Ridge-GCV
│   ├── single_chain_pls_ridge.py   # Single chain + PLS-Ridge
│   ├── hard_aom_chain_pls_ridge.py # One chain per latent component (greedy)
│   ├── soft_aom_chain_pls_ridge.py # Sparse non-neg mixture per component
│   ├── sparse_multi_kernel_ridge.py# K_θ = Σ θ_s K_s with greedy selection
│   └── fast_aom_pls_ridge.py       # End-to-end orchestrator
├── benchmarks/
│   ├── run_fast_aom_benchmark.py   # CLI runner (mirrors aompls runner)
│   └── compare_to_baselines.py     # Cross-CSV comparison utility
└── tests/                          # 54 tests (operator chain, grammar, screening,
                                    #   low-rank, four model variants, end-to-end)
```

## Math

Convention follows the parent `aompls` codebase: an operator's matrix `A` acts on
row spectra as `transform(X) = X @ A.T`. The user's preprocessing notation
`X_s = X @ A_s` corresponds to `A_s = A.T`, so:

- **Screening score**
  `score(j, s) = ||A_s^T B_j(X)^T y||^2 / (||B_j(X) A_s^T||_F^2 ||y||^2)`
  Numerator: `chain.apply_cov(g_0)` where `g_0 = B_j(X)^T y`.
  Denominator: `||F_s||_F^2 ||y||^2` with `F_s = chain.apply_cov(V * S[None, :])`
  computed from the truncated SVD `B_j(X) ≈ U S V^T`.

- **Low-rank kernel** `K_s ≈ U C_s U^T` with `C_s = F_s^T F_s`.
  Kernel-vector product is `O(n r + r^2)` per chain.

- **Ridge on latent scores** `c = (T^T T + λ I)^{-1} T^T y` with optional
  component-wise shrinkage `λ_h = λ_0 h^γ`.

## Models

| Class | Mode | Notes |
|---|---|---|
| `SingleChainPLSRidge` | Top-1 chain from screening | Baseline; equivalent to "best preprocessing + PLS-Ridge" |
| `HardAOMChainPLSRidge` | One chain per latent component (greedy argmax) | Faithful AOM-on-chains; early-stop when train loss plateaus |
| `SoftAOMChainPLSRidge` | Sparse non-negative LASSO mixture per component | Coordinate descent (0.5-loss convention) |
| `SparseMultiKernelRidge` | K_θ = Σ θ_s K_s with greedy chain selection | Projected-gradient NNLS for θ, GCV for λ |
| `FastAOMPLSRidge` | End-to-end orchestrator (`FastAOMConfig`) | Bases → chains → SVD → screen → diversity → fit |

## Usage

```python
from FastAOM.models import FastAOMConfig, FastAOMPLSRidge

cfg = FastAOMConfig(
    model="hard_aom_chain",
    primitive_bank="compact",
    max_chain_depth=3,
    rank=200,
    top_global=120,
    top_per_family=6,
    n_components=15,
)
est = FastAOMPLSRidge(config=cfg)
est.fit(X_train, y_train)
yhat = est.predict(X_test)

# Diagnostics include per-component chains, screening counts, and timings
print(est.diagnostics_["per_component_chains"])
print(est.diagnostics_["timings_s"])
```

## Benchmark

```bash
# 11-dataset smoke cohort, all variants, seed 0
python bench/AOM_v0/FastAOM/benchmarks/run_fast_aom_benchmark.py \
    --cohort bench/AOM_v0/Ridge/benchmark_runs/diverse11_cohort.csv \
    --workspace bench/scenarios/runs/paper_aom_fastaom_seed0 \
    --seeds 0 \
    --max-components 15

# Compare against existing aompls baselines
python bench/AOM_v0/FastAOM/benchmarks/compare_to_baselines.py \
    --files bench/scenarios/runs/paper_aom_fastaom_seed0/results.csv \
            bench/AOM_v0/benchmark_runs/smoke_old_11ds/results.csv \
    --cohort bench/AOM_v0/Ridge/benchmark_runs/diverse11_cohort.csv \
    --out bench/scenarios/runs/paper_aom_fastaom_seed0/comparison.csv
```

## Variants benchmarked

| Label | Model | Primitive bank | Depth | Top-K | Notes |
|---|---|---|---|---|---|
| `FastAOM-single-chain-compact` | single_chain | compact (9 ops) | 3 | 200 | Top-1 chain from screening |
| `FastAOM-hard-chain-compact` | hard_aom_chain | compact | 3 | 120 | One chain per component |
| `FastAOM-hard-chain-compact-d4` | hard_aom_chain | compact | 4 | 160 | Deeper chains |
| `FastAOM-hard-chain-default` *(opt-in)* | hard_aom_chain | default (100 ops) | 2 | 200 | Larger primitive bank. Excluded from the default variant list because the 100-op bank pushes fit time past 5 minutes per dataset on large-p NIRS; enable with `--variants FastAOM-hard-chain-default`. |
| `FastAOM-soft-chain-compact` | soft_aom_chain | compact | 3 | 80 | Sparse mixture per component |
| `FastAOM-sparse-mkr-compact` | sparse_mkr | compact | 3 | 60 | Multi-kernel Ridge |
| `FastAOM-hard-chain-shrinkage` | hard_aom_chain | compact | 3 | 120 | λ_h = λ_0 · h component shrinkage |
| `FastAOM-hard-chain-multibase` | hard_aom_chain | compact | 3 | 160 | + SNV / MSC / EMSC bases |

## Known limitations

- **Fixed n_components** (with early-stop on hard-chain). The aompls `cv` mode
  picks n_components by internal CV; FastAOM does not. Comparisons therefore
  conflate model behaviour with component-selection policy. See the
  `delta_rmsep_vs_master_pls` column for the like-for-like comparison.
- **`xcorr_zero_pad` is a Python loop** in the parent `aompls.operators`
  module — chain enumeration of deep chains over a `default` bank can take
  several minutes per dataset on large `p` (≥ 2000). The `compact` bank stays
  ~10-20 s per dataset.
- **Detrend builds a dense `p × p` matrix**; OK up to `p ≈ 5000` but watch the
  RAM for very large spectra.
- **Deterministic given (X, y)**: there is no internal CV randomness, so
  running multiple seeds against the same train/test split yields identical
  rows. Use the seed only when the cohort itself resamples.
