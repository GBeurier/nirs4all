# Codex Code Review: BLUP

Read:

```text
bench/aom_v0/Multi-kernel/Blup/blup/estimator.py
bench/aom_v0/Multi-kernel/Blup/blup/decomposition.py
bench/aom_v0/Multi-kernel/Blup/blup/diagnostics.py
bench/aom_v0/Multi-kernel/Blup/tests/test_blup_no_leakage.py
```

Check:

1. Sklearn API compliance (same items as MKM code review).
2. `predict_components` matches `predict` in mean (test).
3. `train_decompose()` returns the same invariant on training data.
4. `contribution_table` returns a pandas DataFrame with required columns
   (sample_id, block_name, contribution, contribution_norm,
   contribution_relative).
5. No leakage: spy test verifies test-time predictions use only stored
   training statistics.
6. `BLUP` properly delegates fitting to `MKM`; does not duplicate REML
   logic.
7. Numerical operations use `np.float64` and avoid loss of precision in
   block-by-block accumulation.
8. Diagnostic functions return finite values even for degenerate inputs
   (e.g. block at boundary).

Return findings ordered by severity.
