# Beta-Readiness Audit: Operators Module - Release Blockers Only (2026-02-19)

## Active Findings

(none)

## Beta Release Tasks (Open)

(none)

## Resolved Findings
- `O-03 [RESOLVED]` Module docstrings and `# ====` section headers added to `nirs.py` (8 sections: Wavelet, SavGol, Scatter, Area Norm, Log, Derivatives, Reflectance, Baseline) and `scalers.py` (3 sections: SNV variants, Range normalization, Derivatives) for navigability.
- `O-04 [RESOLVED]` Stale "backward compatibility" comments removed or updated across operators: `nirs.py` (`asls_baseline`), `nlpls.py` (NLPLS/KPLS aliases), `nicon.py` (`decon_layer_classification`), `merge.py` (`output_as`), `splitters.py` (SPXYGFold). Incorrect error message in `Derivate.fit()` fixed (was "SavitzkyGolay"). Missing docstrings added to `Derivate`, `SimpleScale`, `MultiplicativeScatterCorrection`.
- `O-01 [RESOLVED]` Regression tests for `LogTransform` fitted-offset behavior added (`tests/unit/operators/transforms/test_log_transform_crop_transformer.py`).
- `O-02 [RESOLVED]` Regression tests for `CropTransformer` state-mutation integrity added (`tests/unit/operators/transforms/test_log_transform_crop_transformer.py`).
