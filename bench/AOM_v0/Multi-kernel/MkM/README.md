# MkM — AOM Multi-Kernel Mixed Model

`AOM-MKM` is the probabilistic counterpart of AOM-Ridge / mkR.
Each AOM operator becomes a random effect with its own variance
component, estimated by REML. The output is:

- a prediction (equivalent to mkR with REML-derived weights);
- a per-block **variance contribution** `sigma_b^2` (interpretable as
  relative importance of each preprocessing operator).

```python
from mkm import AOMMultiKernelMixedModel

model = AOMMultiKernelMixedModel(
    operator_bank="compact",
    method="reml",
    n_restarts=10,
    random_state=0,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Variance components:", model.sigma2_blocks_)
print("Residual variance:", model.sigma2_residual_)
print("Relative contributions:", model.relative_contributions_)
```

## Documentation

- `docs/IMPLEMENTATION_PLAN.md` — phase-gated implementation plan.
- `docs/MKM_MATH_SPEC.md` — likelihood, gradient, and conventions.
- `docs/TEST_PLAN.md` — required tests and acceptance criteria.
- `docs/BENCHMARK_PROTOCOL.md` — cohort, variants, output schema.
- `docs/CODEX_REVIEW_WORKFLOW.md` — review gates.

## Tests

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM \
  pytest bench/aom_v0/Multi-kernel/MkM/tests -q
```

## Benchmarks

Smoke (3 datasets):

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM \
  python bench/aom_v0/Multi-kernel/MkM/benchmarks/run_mkm_benchmark.py \
    --workspace bench/aom_v0/Multi-kernel/MkM/benchmark_runs/smoke --cohort smoke
```

Full (57 datasets):

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM \
  python bench/aom_v0/Multi-kernel/MkM/benchmarks/run_mkm_benchmark.py \
    --workspace bench/aom_v0/Multi-kernel/MkM/benchmark_runs/full --cohort full
```
