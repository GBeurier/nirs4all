# aompls — AOM-PLS Compact (PLS1) in C++, R, and Python

A header-only C++17 implementation of the **AOM-PLS compact-cv5** variant
identified as the best fast-and-simple regression model in the
`bench/AOM_v0/publication/` ranking over 57 NIRS datasets (mean RMSEP 755 vs
PLS-standard 762). Optional preprocessing: SNV, MSC, OSC, ASLS.

| Layer | Status | Test gate |
|---|---|---|
| C++ core (`cpp/`) | ✅ Tested (machine precision) | `cpp/build/test_operators`, `cpp/build/test_parity_kfold` |
| C API shared lib (`cpp/build/libaompls.so`) | ✅ Tested via C smoke + Julia | `gcc build/c_smoke.c` and load via Julia ccall |
| Python binding (`python/`, pybind11) | ✅ Tested (9/9 pass) | `pytest python/tests/test_parity.py` |
| R binding (`r/aompls/`, Rcpp) | ✅ Tested (54/54 assertions pass) | `R CMD INSTALL` + `testthat::test_dir` |
| MATLAB binding (`matlab/`, MEX) | ⚙️ Code ready, needs MATLAB to compile | `mex aompls_mex.cpp`, `test_parity()` |
| Julia binding (`julia/AompLS/`, ccall) | ⚙️ Code ready, needs Julia runtime | `julia --project=. -e 'using Pkg; Pkg.test()'` |
| JavaScript / WASM (`js/`, Emscripten) | ⚙️ Code ready, needs emsdk to build | `./build.sh && npm test` |

All three layers are validated against the Python `AOMPLSRegressor`
reference (`bench/AOM_v0`) on three datasets (BEER, CORN, ALPINE) over
three CV variants (`kfold5`, `kfold5+oneSE`, `spxy5`). Parity tolerances:
- coefficient max |Δ| < 1e-8
- prediction max |Δ| < 1e-8
- RMSE-curve max |Δ| < 1e-9
- selected operator name + n_components: exact match

## Layout

```
bench/AOM_lib/
├── cpp/
│   ├── include/aompls/      # Header-only C++ library + c_api.h (extern "C")
│   ├── include/Eigen/       # Vendored Eigen 3.4.0 (MPL2, BLAS-free)
│   ├── src/c_api.cpp        # C ABI compiled into libaompls.so
│   ├── tests/               # C++ unit tests + JSON parity fixtures
│   └── CMakeLists.txt       # Optional cmake build
├── python/                  # pip-installable package "aompls" (pybind11)
├── r/aompls/                # R package "aompls" (Rcpp)
├── matlab/                  # MATLAB MEX wrapper + .m helpers + test_parity.m
├── julia/AompLS/            # Julia package — loads libaompls.so via ccall
├── js/                      # Emscripten/WASM build (Node.js + browser)
└── scripts/
    ├── export_reference.py  # Regenerate JSON parity fixtures from Python AOM_v0
    └── sync_headers.sh      # Mirror cpp/include/ → r/aompls/inst/include/
```

## The algorithm in one diagram

```
                ┌─────── compact bank (9 strict-linear operators) ───────┐
                │ 0:identity  1-2:SG-smooth  3-4:SG-deriv1  5:SG-deriv2  │
                │ 6-7:detrend(d=1,2)  8:FD(order=1)                      │
                └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
preproc (optional)        ┌─ AOM global selection ────────────────────────┐
SNV/MSC/OSC/ASLS  ──►  X─►│ for each (operator b, fold f, prefix k):      │──► (best_b, best_k)
                          │   materialized SIMPLS on Xb = X·Aᵀ            │
                          │   B_k = Z[:, :k] (P[:, :k]ᵀZ[:, :k])⁻¹ q[:k]  │
                          │   RMSE(b, f, k) = ‖y − pred(B_k)‖             │
                          │ pick argmin over (b, k); optional 1-SE rule   │
                          └────────────────────────────────────────────────┘
                                       │
                                       ▼
                               refit on full data → coef, intercept
```

## Quick start (Python)

```python
from aompls import AOMPLSCompact

m = AOMPLSCompact(
    max_components=15,
    n_folds=5,
    cv_mode="kfold",        # or "spxy", "holdout", "external"
    one_se_rule=False,
    preproc="none",         # or "snv", "msc", "osc", "asls", "snv+osc", "asls+osc"
).fit(X_train, y_train)
y_pred = m.predict(X_test)

# Inspect
print(m.selected_operator_name_, m.n_components_)
print(m.rmse_curves_.shape)  # (9, 15)
```

## Quick start (R)

```r
library(aompls)
m <- aom_pls(X, y, max_components = 15L, cv_mode = "kfold", preproc = "snv")
pred <- predict(m, X_test)
print(m)
```

## Quick start (C++)

```cpp
#include "aompls/aom_pls.hpp"

aompls::AOMConfig cfg;
cfg.max_components = 15;
cfg.cv_mode = aompls::CVMode::KFOLD;
cfg.preproc = aompls::Preproc::SNV;
aompls::AOMResult m = aompls::fit(X.data(), n, p, y.data(), cfg);

std::vector<double> y_pred(n_new);
aompls::predict(m, X_new.data(), n_new, y_pred.data());
```

## Building

### C++ standalone

```bash
cd cpp
# With cmake:
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build
# Without cmake:
g++ -std=c++17 -O3 -DEIGEN_NO_DEBUG -I include tests/test_operators.cpp -o build/test_operators
./build/test_operators
g++ -std=c++17 -O3 -DEIGEN_NO_DEBUG -I include tests/test_parity_kfold.cpp -o build/test_parity_kfold
./build/test_parity_kfold tests/reference
```

### Python

```bash
cd python
pip install -e .
pytest tests/
```

### R

```bash
# Mirror the headers into r/aompls/inst/include first
scripts/sync_headers.sh
R CMD build r/aompls
R CMD INSTALL aompls_0.1.0.tar.gz
AOMPLS_REF_DIR=$(pwd)/cpp/tests/reference \
  Rscript -e 'library(aompls); library(testthat); test_dir("r/aompls/tests/testthat")'
```

### MATLAB (MEX)

```matlab
cd matlab
build                        % compiles aompls_mex.cpp
test_parity                  % runs parity gate against ../cpp/tests/reference/*.json
```

### Julia

```bash
# 1. Build libaompls.so once from the C++ side:
cd cpp
g++ -std=c++17 -O3 -fPIC -shared -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE \
    -I include src/c_api.cpp -o build/libaompls.so
# 2. Run the Julia parity suite:
cd ../julia/AompLS
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

### JavaScript / WASM

```bash
# Requires emsdk activated (https://emscripten.org)
cd js
./build.sh                   # → dist/aompls.mjs + dist/aompls.wasm
npm test                     # runs parity gate via Node.js
```

## Regenerating the parity fixtures

When the Python `AOM_v0` reference moves, refresh the JSON fixtures:

```bash
.venv/bin/python scripts/export_reference.py
```

The C++ / Python / R parity tests all consume the same `cpp/tests/reference/<DATASET>.json` files.

## API options

| Option | Type | Default | Notes |
|---|---|---|---|
| `max_components` | int | 15 | Auto-prefix search over k ∈ [1, max_components] |
| `n_folds` | int | 5 | Ignored when `cv_mode == "external"` |
| `cv_mode` | enum | `"kfold"` | `"kfold"` (own RNG) / `"spxy"` / `"holdout"` / `"external"` |
| `one_se_rule` | bool | false | One-standard-error parsimony (prefix-curve SE) |
| `center` | bool | true | Strongly recommended; mean-center X and y |
| `random_state` | int | 0 | Seed for the own RNG; NOT bit-compatible with numpy's MT19937 |
| `preproc` | enum | `"none"` | `"snv"`, `"msc"`, `"osc"`, `"asls"`, `"snv+osc"`, `"asls+osc"` |
| `osc_n_components` | int | 1 | Wold-1998 OSC (only when preproc includes OSC) |
| `asls_lam, asls_p, asls_n_iter` | float, float, int | 1e5, 0.01, 10 | Eilers-Boelens 2005 |
| `external_folds` | list<list<int>> | none | Test indices per fold; required when cv_mode = "external" |

## Algorithm details

- **Bank** (load-bearing order; persisted in `bank_names`): identity →
  SG(11,2,0) → SG(21,3,0) → SG(11,2,1) → SG(21,3,1) → SG(11,2,2) →
  detrend(d=1) → detrend(d=2) → FD(order=1). Identity always at index 0;
  the one-SE tiebreak prefers low indices.
- **Engine**: materialized SIMPLS through the chosen operator
  (`X_b = X · Aᵀ`, then standard SIMPLS on `X_b`, then `Z = Aᵀ R`).
  Matches `simpls_materialized_fixed` in `bench/AOM_v0/aompls/simpls.py:125-195`.
- **Scoring**: for each fold, fit once at K = `max_components`, evaluate all
  prefixes via `coef_prefix(k) = Z[:, :k] (P[:, :k]ᵀ Z[:, :k])⁻¹ q[:k]`.
  Mean over folds (np.mean semantics: any inf in a fold → mean is inf).
- **Selection**: argmin over (operator, k). One-SE rule (optional): SE = std
  of winning curve / √K, threshold = best + SE, pick smallest k then smallest
  operator index within threshold.
- **SPXY**: joint normalised X+y euclidean distance; alternating max-min
  fold assignment with `max_size = floor(n/K) + (1 if remainder else 0)`.
- **Preprocessing**:
  - SNV: stateless per-row centering and std-normalisation.
  - MSC: state = training mean spectrum; per-row regression coefficients
    applied at predict.
  - OSC (Wold-1998): state = W, P (p × k); replay
    `X ← X − X W (PᵀW)⁻¹ Pᵀ`.
  - ASLS (Eilers-Boelens 2005): stateless per-spectrum baseline subtraction.

## Parity gate (PLS1, 3 datasets × 3 modes)

| Dataset | Mode | coef \|Δ\| | pred \|Δ\| | curves \|Δ\| |
|---|---|---|---|---|
| BEER | kfold5 | 4.0e-15 | 1.3e-13 | 8.9e-15 |
| BEER | kfold5+oneSE | 4.0e-15 | 1.3e-13 | 8.9e-15 |
| BEER | spxy5 | 9.3e-15 | 8.5e-14 | 4.0e-15 |
| CORN | kfold5 | 2.9e-12 | 5.3e-14 | 8.8e-15 |
| CORN | kfold5+oneSE | 1.9e-12 | 6.0e-14 | 8.8e-15 |
| CORN | spxy5 | 2.9e-12 | 5.3e-14 | 6.8e-15 |
| ALPINE | kfold5 | 4.3e-14 | 6.0e-15 | 1.2e-15 |
| ALPINE | kfold5+oneSE | 1.5e-14 | 3.6e-15 | 1.2e-15 |
| ALPINE | spxy5 | 2.7e-14 | 4.3e-15 | 1.2e-15 |

## Provenance / citation

This library re-implements the algorithm in
`bench/AOM_v0/aompls/{banks.py,simpls.py,selection.py,operators.py}` with
bit-equivalent numerical outputs on the parity fixtures.

## License

CeCILL-2.1 (matches the parent nirs4all repository).
Vendored Eigen 3.4.0 is MPL 2.0.
