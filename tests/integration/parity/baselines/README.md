# `baselines/` — captured legacy gold baselines

Each `<case>.json` is the **gold-standard observation** of a parity case run on
the **legacy** nirs4all backend — the oracle of record (ADR-01). They are
committed artifacts: parity is judged against them.

Schema (one file per `PipelineCase.name`):

```json
{
  "case": "baseline_vertical_slice",
  "backend": "legacy",
  "pipeline_fingerprint": "<sha256[:16] of repr(pipeline)>",
  "num_predictions": 3,
  "models": ["PLSRegression"],
  "datasets": ["regression"],
  "metrics": {"best_score": ..., "rmse": ..., "r2": ..., "cv_best_score": ...}
}
```

- **Capture / recapture** (after an *intended* change to a case or the engine):
  `pytest tests/integration/parity/test_parity_baseline.py --parity-capture -m slow`
- **Enforce** (default): a fresh legacy run is diffed against the baseline within
  each case's `metric_tolerances`; structural fields must match exactly.
- `pipeline_fingerprint` guards staleness: if a case's pipeline changes, the
  enforce run fails fast asking for a recapture instead of comparing against an
  outdated baseline.

When the dag-ml backend is wired (PARITY_AND_PERF_HARNESS.md, Layer 3) it is
compared against these same baselines — only the backend producing the
observation changes.
