# Blup — AOM Multi-Kernel BLUP / E-BLUP

`AOM-BLUP` is the prediction-decomposition layer on top of AOM-MKM.
Once variance components are estimated, BLUP gives **per-block
contributions** to each prediction:

```python
from blup import AOMMultiKernelBLUP

model = AOMMultiKernelBLUP(operator_bank="compact", method="reml", random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
components = model.predict_components(X_test)
# {"fixed": (n_test,), "<op_1>": (n_test,), ..., "total": (n_test,)}
assert np.allclose(components["total"], y_pred)
```

The decomposition lets users see, for each test sample, **which AOM
preprocessing operator drove the prediction**.

## Documentation

- `docs/IMPLEMENTATION_PLAN.md`
- `docs/BLUP_MATH_SPEC.md`
- `docs/TEST_PLAN.md`
- `docs/BENCHMARK_PROTOCOL.md`
- `docs/CODEX_REVIEW_WORKFLOW.md`

## Tests

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM:bench/aom_v0/Multi-kernel/Blup \
  pytest bench/aom_v0/Multi-kernel/Blup/tests -q
```

## Benchmarks

```bash
PYTHONPATH=bench/aom_v0/Multi-kernel:bench/aom_v0/Multi-kernel/MKR:bench/aom_v0/Multi-kernel/MkM:bench/aom_v0/Multi-kernel/Blup \
  python bench/aom_v0/Multi-kernel/Blup/benchmarks/run_blup_benchmark.py \
    --workspace bench/aom_v0/Multi-kernel/Blup/benchmark_runs/smoke --cohort smoke
```
