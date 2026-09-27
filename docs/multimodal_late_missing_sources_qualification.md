# Late fusion with missing sources: qualification record

## Current port — 2026-09-28

U13 was ported selectively from the unmerged CM02-ragged worktree onto Python
**1.3.0**. The port retains current fold-local views, native exclusions,
portable output selection and IO 0.2.0 compatibility. Astra and Opus 5.5
reviewed the change before tests; Opus found an example-runner integration
error, which was fixed and reviewed again. Both then returned GO. No release
or tag was made.

The two test modules and the example below describe the intended contract.
Their fixtures use the current late-fusion/ragged/tuning helpers. Tests explicitly
select the DAG-ML host profile, refit, and artifact capture; this is not a
portable cross-language archive profile. U13 reports include the Python library
version so new evidence can be distinguished from historical evidence.

### Current synthetic qualification

The source base was nirs4all `03f541375bea` (Python 1.3.0), with the candidate
IO ragged wheel built from IO main `0644d8f`. The validation environment used
Python 3.11.15, DAG-ML 0.3.30, Data 0.2.12, Core 0.4.0, Methods 1.2.1,
NumPy 2.4.6 and scikit-learn 1.9.1. The IO wheel still declares version
0.2.0; these new ragged changes are **not** in the previously published 0.2.0
artifact. All tests use deterministic synthetic cohorts.

| Gate | Current result |
| --- | --- |
| `pytest -q test_multimodal_late_missing.py test_multimodal_late_missing_export.py` | **30 passed** against the current dependencies; the same 30 passed in the older local test environment. |
| Existing late-fusion, DAG multimodal, tuning and ragged integration modules | **46 passed**, then the isolated-subprocess case passed after installing this checkout in the venv: **47/47**. The initial subprocess failure was `ModuleNotFoundError: nirs4all` from a stale editable path, not a test assertion. |
| `ruff check` on the changed modules/tests/example; `mypy nirs4all` | Passed; mypy checked **549 source files** in the current-dependency environment. |
| U13 continuous and stop-after-one/resume examples | Both completed; trial records, best parameters/value, CV/test scores, source-presence contract and new predictions matched exactly. Resume retained the first trial and completed trials 1–2. |
| `examples/run.sh -n U13_multimodal_late_missing_sources.py -p -s` twice | Both invocations passed and received distinct temporary artifact directories; plot/show flags were not forwarded to U13. |
| Candidate wheel installed in a fresh venv; `uv pip check` | Passed with **47 compatible packages**. Replayed the archive from another directory after moving the training workspace, with provider generation and Ridge fitting forbidden. Exact predictions on 8 partial-source rows and 8 all-series-absent rows; integrity verified and `training_performed=False`. |

The local Python wheel SHA256 is
`45641f65979c97b239b60fca97e096932295c6cb94db0a632b33fd93de228ddb`;
the candidate IO wheel SHA256 is
`aa4fd2a79a3d931ca6034458c07cc5ef742855f36f56368a1d3a868fc6834b07`.
Neither wheel was published. These checks qualify the Python host path; they do
not close the full package suite, Studio UI, R/WASM parity or a release. Per-base
score displays currently include the zero-filled absent rows and should not be
interpreted as source-specific model accuracy; the meta-model's OOF scoring uses
the declared presence columns. The historical interpreter, PYTHONPATH and wheel
pins below are not installation instructions for this port.

## Historical evidence — 2026-09-19 only

Everything under “Historical recorded checks”, including counts, timings,
metrics, wheel hashes, temporary paths and descriptions of executed checks,
is copied from the original CM02-ragged qualification note. Except for the
current gates explicitly listed above, these assertions have **not been
re-executed or independently reverified during the 2026-09-28 port**. The
referenced local logs and temporary artifacts may no longer exist.
They establish provenance for the earlier experiment, not release evidence for
Python 1.3.0 or its current native dependencies. The historical source described
its synthetic regression profile as qualified but unpublished; its frozen wheels
preceded later performance changes.

## Intended contract for the current port

The public opt-in is attached to the source branch:

```python
{"branch": {
    "by_source": True,
    "steps": {"nir": [encoder_nir, regressor_nir],
              "series": [encoder_series, regressor_series]},
    "missing_source_policy": "zero_with_indicator",
}}
```

The branch is followed by `{"merge": "predictions"}` and the meta-regressor.
Targets must be complete; the tested regression cases have one or two outputs.
Each base encoder/model sees only the observations where its own source is
present, within the native fold's fit scope. An empty source-specific fit scope
is refused, including an empty inner fold. A base model that would consume other
source blocks is also refused. Classification and partial targets are outside
this profile. The default `error` policy continues to require complete sources.

Native nested OOF and HPO retain all observation IDs. Each source contributes
`[prediction(s), presence]` in the native OOF source order. Present predictions
are inverse-transformed to the original target space before absent rows are
filled with zero. A source absent for every new prediction row does not execute
its encoder or model. Dense and ragged source blocks retain their native input
representations until their explicit encoders run.

The fitted bases retain their policy, source bindings and prediction widths;
the meta-model retains the policy and native source order. Export records
`multimodal_host.source_presence`, including
the per-source prediction/presence layout and total meta-feature width. Replay
checks this declaration against the fitted payload after the existing artifact
integrity check, and rejects inconsistent policies, widths or source order.

## Historical recorded checks — 2026-09-19

| Scope | Result | Evidence and qualification boundary |
| --- | --- | --- |
| New native fit/HPO integration | **11 passed in 20.05 s** | Parent phase result for `tests/integration/api/test_multimodal_late_missing.py`. |
| New export/replay selection | **19 passed, 16 warnings in 5.71 s** | Terminal log reread at `.qualification/late-missing-export.log`. |
| Existing late fusion and late HPO | **24 passed, 118 warnings in 46.77 s** | Terminal log reread at `.qualification/late-missing-replay-regression.log`. |
| Lowering and existing native paths | 15 passed in 6.52 s, then 18 passed in 42.42 s | Reported lowering phase results: 9 lowering + 6 late-fusion tests, then 18 late-HPO tests. The 24 integration cases overlap the preceding row. |
| Native compilation smoke | Passed | Reported phase result: default/error layouts unchanged; policy-specific graph fingerprints and base/meta metadata; invalid policies and unsupported task/target profiles refused. No fit in this smoke. |
| Static checks | Whole-repository Ruff passed; **mypy passed on 533 source files** | Ruff phase result; terminal mypy log reread at `.qualification/late-missing-mypy-20260919.log`. |
| Consolidated multimodal selection | **351 passed, 4 skipped, 652 warnings in 105.98 s** | 355 collected; terminal `.qualification/late-missing-host-20260919.log`. Includes the preceding integration/operator selections; precedes the provider-view correction below. |
| Provider-view provenance correction | **11 passed in 7.18 s**; touched-module Ruff/mypy passed | Follow-up phase result after the consolidated gate: independent provider evidence on aggregate and child results, with/without late HPO, matching the archive witness. |
| U13 source/install, continuous/resumed | Passed with identical histories, selected parameters, scores and predictions | `source-installed-parity.json` and `native-history-parity.json` in the installation directory below. |
| Fresh installation and isolated replay | **Passed** | Two archives × two cohorts of 8 rows; exact predictions, verified integrity, no fit/generation/source access. Pure Python IO profile only. |

Log paths above are relative to
`/home/delete/nirs4all/_worktrees/CM02-ragged-20260919`.
These selections overlap and are not a new full-suite result.

The 11 native integration cases compare exact fit scopes against the complete
native path, perturb hidden source buffers, refuse empty inner fits, and verify
that changed outer-validation labels do not affect those rows' predictions.
They check durable HPO stop/resume equality, checkpoint binding to presence masks
while ignoring hidden buffers, and dense missing sources with a
`TransformedTargetRegressor` for one/two outputs.

The 19 replay cases include four native export/replay combinations: one/two
outputs with a partially or entirely absent series source. They remove the
training workspace, reorder source mappings and observation IDs, round-trip the
new cohort through JSON, prohibit fit/generation/legacy scheduling, and require
exact prediction equality. The remaining cases exercise inverse-target zero
placement, complete-input presence columns, strict-policy compatibility and
rejection of changed fitted state or manifest declarations. The entire-series
absence case keeps the other modalities present.

## Historical U13 objective and CV score comparison

The winning trial and the selected pipeline's final CV have exactly the same
native meta-model RMSE for each outer fold:

| Fold | Scored observations | RMSE |
| --- | ---: | ---: |
| fold0 | 8 | 0.26076330373004614 |
| fold1 | 8 | 0.21775567362024906 |
| fold2 | 8 | 0.19936771418276433 |

`tuning.best_value = 0.22596223051101985` is the arithmetic mean of those fold
RMSE values. `cv_best_score = 0.22742216208663363` is the pooled OOF RMSE:
`sqrt(sum(n_fold * RMSE_fold**2) / sum(n_fold))`. Its native report has
`fold_id="avg"` and 24 rows. These reductions differ even with equal-sized folds.
The test RMSE is a third, held-out measurement: `0.27312651767442964`.

This was checked without fitting: the winning checkpoint's
`objective_fold_scores` match the final `score_set.json` fold reports exactly.
Recalculation from the three `predictions.parquet` blocks covers 24 unique IDs
and reproduces both reductions within floating-point rounding (3e-17).
Evidence is retained under `/tmp/nirs4all-multimodal.OPccZc/`, including
`late-missing.n4mopt.json` and
`workspace/native_results/20260919T123248681340Z-f4cd73e4/`.

The archive's actual fitted parameters were also read: NIR Ridge `alpha=1.0`,
series `SequenceSummary.include_length=True`, and meta Ridge `alpha=0.1`, exactly
the selected trial-0 parameters. The host explicitly requests
`fold_score_reduction="mean"` in
[multimodal_tuning.py](../nirs4all/pipeline/dagml/multimodal_tuning.py), applies the
selected parameters via
[LateFusionTuningRecipe](../nirs4all/pipeline/dagml/late_tuning.py), and
[RunResult.cv_best](../nirs4all/api/result.py) selects the native `avg` entry.
The observed difference is an aggregation convention, with no evidence here of
changed folds, unapplied parameters or a stochastic-training discrepancy. No
runtime change was made for this audit; HPO selection and pooled reporting must
retain their distinct labels.

## Historical frozen U13 installation and replay

All installation witnesses are under
`/tmp/n4a-late-missing-install-20260919/`: `qualification.json`, `README.md`,
`source-installed-parity.json`, `native-history-parity.json`, `source-snapshot.json`,
`wheels.json`, `commands.jsonl` and `logs/`. The snapshot contains 618 source files
and was taken after the provider-view correction. SHA-256 values were reread:

| Artifact | SHA-256 |
| --- | --- |
| nirs4all 1.0.1, `wheels/nirs4all/nirs4all-1.0.1-py3-none-any.whl` | `2853597c3b685663307782472d26e7ff4e4d884b8b74f4df387e1247a336bd62` |
| nirs4all-io 0.1.18, earlier ragged pure Python wheel | `fc3286e5026bd9e8004c38d648aa6cf315019f04c85b4bf31b5947665f90c0f8` |
| Reused native dag-ml 0.3.25 provider wheel | `681969a3d8eba733061350c819df6dd16f8b604489c55e960f34dc1d58b1a6a5` |

The disposable environment matched all 49 reference package versions and passed
`pip check`. Source and installed continuous/resumed runs have four identical
native three-trial histories. The interrupted invocation completes trial 0;
resume executes only trials 1 and 2. Selected parameters, CV/test scores,
provider provenance and both new prediction vectors also agree exactly.

Independent replay starts Python with `-I`, using copies of the archive and new
cohort JSON. Audit hooks deny source, build-snapshot and training-directory
access; fit, fit-transform, partial-fit, provider materialization and legacy
scheduling are forbidden. Each archive predicts 8 rows with 5/8 series present
and 8 rows with 0/8 series present, with NIR present throughout. The latter also
forbids `SequenceSummary.transform`. Both continuous and resumed archives give
maximum absolute error 0, verified artifact integrity and no scoring or fitting.

This qualifies Python host models with **pure Python IO**: `nirs4all_io._native`
was absent. It does not qualify a native IO distribution or portable model
replay. The three historical provider wheels and earlier ragged proof stayed
unchanged. The installed IO wheel also predates the later packed-Torch opt-in.

After successful qualification, the shared disk reached `ENOSPC`. Only the two
disposable `venv` directories under `/tmp/n4a-ragged-install-20260919` and
`/tmp/n4a-late-missing-install-20260919` were removed. Their
`environment-cleanup.txt` notes confirm that wheels, build snapshots, dependency
manifests, scripts and completed reports remain available to reconstruct the
environments from the recorded pinned artifacts. No Studio environment or job
was changed. The later expanded 2,745-case verification is not a successful
gate claimed by this report; its failures and follow-up belong to the separate
performance tranche.

## Historical reproduction scopes — not current validation

Run from the isolated `nirs4all` checkout, using the existing interpreter
`/home/delete/nirs4all/nirs4all/.venv/bin/python` without installation. Select
the isolated `nirs4all-io/src` and `nirs4all` with `PYTHONPATH`, the former with
`MYPYPATH`, set `PYTHONDONTWRITEBYTECODE=1`, and set `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1`.
These are the targeted selections, shown separately to preserve their counts:

```bash
rtk proxy /home/delete/nirs4all/nirs4all/.venv/bin/python -m pytest tests/integration/api/test_multimodal_late_missing.py -q -p no:cacheprovider
rtk proxy /home/delete/nirs4all/nirs4all/.venv/bin/python -m pytest tests/integration/api/test_multimodal_late_missing_export.py -q -p no:cacheprovider
rtk proxy /home/delete/nirs4all/nirs4all/.venv/bin/python -m pytest tests/integration/api/test_multimodal_late_fusion.py tests/integration/api/test_multimodal_late_tuning.py -q -p no:cacheprovider
```

The [native fit/HPO tests](../tests/integration/api/test_multimodal_late_missing.py)
and [export/replay tests](../tests/integration/api/test_multimodal_late_missing_export.py)
define the port's executable acceptance criteria; they are not yet current passing evidence. The historical tranche covered synthetic Python host models
with native DAG scheduling; no real corpus, cross-language model replay,
classification with absent sources, partial-target late fusion or publication
is claimed. Later packed transport, resolver and sklearn selection changes are
tracked in the [provider performance report](multimodal_provider_performance_qualification.md)
and [isolated backlog](../../BACKLOG_ISOLE.md); their source tests and newer wheels
must not be attributed to this frozen U13 installation.
