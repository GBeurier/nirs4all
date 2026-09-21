# Synthetic late-fusion tuning qualification

Development qualification on 18 September 2026, using synthetic inputs only.
This extends the [targets and missing-source milestone](multimodal_targets_and_missing_sources_qualification.md).
The real corpus and article remain deferred. These changes are local development
builds; no release or publication was performed.

## Behavior

Global `run(tuning=...)` now accepts raw multimodal `by_source` branches,
`merge="predictions"` and a downstream meta-model. Search paths address source
encoders, base models and the meta-model in the same trial:

```python
space = {
    "branches.image.0.n_components": [2, 4],
    "branches.nir.1.alpha": [0.1, 1.0],
    "branches.metadata.0.numeric.with_mean": [False, True],
    "meta.alpha": [0.1, 1.0],
}
```

The native `HostHpoRequest.parameter_bindings` maps each public path to a graph
node and operator parameter. Destinations are checked before callbacks; duplicate
destinations and unmapped proposed parameters are rejected. The score producer
remains `target_node`, independently of the nodes being tuned. An absent or empty
mapping preserves the previous request serialization and checkpoint identity.

Concrete stacking and HPO share the same graph construction. Every candidate
recomputes inner OOF predictions inside each outer training scope. Group identity
is preserved through all native inner, outer and final fits. Python reconstruction
of the upstream transform chain applies that candidate's transform parameters.
There is no precomputed global OOF matrix or Python CV scheduling loop.

Public step indices include `None` entries, while graph node indices count the
actual operators. Nested sklearn paths support dots and `__`. Invalid addresses
fail before fitting or creating a checkpoint. The selected recipe uses private
operator copies and leaves the caller's pipeline unchanged.

The result and direct `export()` refer to the selected ensemble, even when a base
producer has a better score. The archive contains the fitted branches, fitted
meta-model, source schemas, selected recipe and public tuning history. Replay
uses new identities and reordered source declarations without fitting.

Scope: complete sources and targets, fixed topology, plain instantiated sklearn
operators, random N4M proposals, no pruning or queued parameters. Complete
multi-output regression and single-target classification are covered. Model-local
`finetune_params`, `train_params` and `refit_params` are refused in this global
profile. Late fusion with partial sources or targets remains outside this tranche.

## Reproducible demonstration

[U10](../examples/user/02_data_handling/U10_multimodal_late_tuning.py) uses NIRS,
raw images, multivariate series and mixed metadata with grouped folds:

```bash
.venv/bin/python examples/user/02_data_handling/U10_multimodal_late_tuning.py \
  --output /tmp/mm-late --stop-after 2
.venv/bin/python examples/user/02_data_handling/U10_multimodal_late_tuning.py \
  --output /tmp/mm-late --resume
```

The stopped/resumed and continuous runs produced identical four-trial histories,
selected parameters and twelve new predictions (maximum absolute difference 0).
Source evidence is in `/tmp/n4a-u10-late-resume` and
`/tmp/n4a-u10-late-continuous`. The mean-fold tuning RMSE is
`0.0878603283375089`; pooled OOF RMSE is `0.10214110643936614`.
These are distinct native reductions and both are preserved. They are selection
metrics on synthetic fixtures, not unbiased scientific performance estimates.

## Tests and gates

- **18 new integration tests pass**, including actual PCA/base/meta parameter
  changes, per-trial inner OOF fits, stop/resume, failed trials not repeated,
  changed-contract refusal before fitting or checkpoint writes, nested sklearn
  parameters, classification and multiple outputs, archive history and replay.
- The same **18 tests pass with scikit-learn 1.5.2**. The main environment uses
  scikit-learn 1.9.1. A completed-checkpoint test covers unseeded RandomForest
  operators and callbacks that consume the global RNG.
- **79 existing multimodal integrations** and **37 shared stacking/node-runner
  checks** pass after the implementation changes.
- DAG-ML: **849 Rust tests pass, 3 ignored**, plus three real Python HPO binding
  tests and the Methods archive lifecycle witness. Formatting, Clippy, W1/D4
  contracts, graph validation and extension freshness pass.
- Ruff passes across the Python repository; mypy passes on **529 source files**.
- Official `examples/run.sh -c user -n '*multimodal*.py'` passes **4/4** with
  the development environment on `PATH`.
- Sphinx builds the complete HTML documentation with no warnings, from a temporary
  copy that keeps generated API pages outside the checkout.
- The full Python unit/integration suite passes **10,580 tests, 113 skipped and
  6,360 warnings, in 737.97 seconds**. No runtime code changed after this run.

The full suite collected the first 17 cases. One additional `GroupShuffleSplit`
case then passed separately in both environments (8.18 s and 6.91 s). It verifies
resumed/continuous equality, native `partitioned_inner_v1` refit preparation with
24 unique training IDs per source, exclusion of preparation from meta selection
scores, and archive replay after deleting the workspaces. The targeted total is
18; this additional case is not included in the full-suite count above.

Logs: `/tmp/n4a-late-hpo-{full,full-mypy,all-existing,shared,sklearn15,examples}.log`.
Full-suite validation uses the isolated historical Methods 1.0.18 witness via
`PYTHONPATH=/tmp/n4m018-multimodal-qualification`; installed U10 uses Methods 1.0.19.

Commands run from the Python repository through the local `rtk proxy` wrapper:

```bash
env PYTHONPATH=/tmp/n4m018-multimodal-qualification \
  .venv/bin/pytest tests/unit tests/integration -n 8 -q --disable-warnings
.venv/bin/ruff check .
.venv/bin/mypy --no-incremental nirs4all
env PYTHONPATH=/tmp/n4a-sklearn15-multimodal-qualification \
  .venv/bin/pytest tests/integration/api/test_multimodal_late_tuning.py -q --disable-warnings
.venv/bin/python -m pip wheel --no-deps --no-build-isolation . -w /tmp/n4a-late-hpo-wheels
.venv/bin/python -m build --sdist --outdir /tmp/n4a-late-hpo-sdist
```

## Installed distribution and replay

The three wheels were installed without editable packages in a fresh Python
3.11.15 environment, `/tmp/n4a-late-hpo-clean311`; `pip check` passes. Standalone
U07/U10 copies live in `/tmp/n4a-late-hpo-demo`. Training from the installed
packages reproduced the source run's four trial records, native objective,
pooled CV metric and twelve predictions exactly. Proof:
`/tmp/n4a-late-hpo-installed-parity.json`.

Two further installed training cases cover three-class classification and
two-output regression. Their two-trial histories, selected parameters, CV scores
and new predictions exactly match the source runs. Independently replayed archives
return 12 original class labels and a `(12, 2)` regression matrix, respectively,
with fit and access to sources/training directories forbidden. Regression maximum
absolute error is 0; class labels match exactly. Evidence:
`/tmp/n4a-late-hpo-tasks-parity.json`,
`/tmp/n4a-late-hpo-classification-replay.json` and
`/tmp/n4a-late-hpo-multioutput-replay.json`. The standalone qualification script is
`/tmp/n4a-late-hpo-demo/qualify_tasks.py`; this supplements the public U10 example.

An isolated replay receives only the archive, new-input JSON and expected report.
The [verification script](../scripts/verify_multimodal_archive.py) forbids all
relevant fit methods and the legacy runner. Its audit hook denies access to the
repository tree and original training directories. All twelve predictions agree
exactly, with verified artifact integrity and zero fit calls. Proof:
`/tmp/n4a-late-hpo-isolated-replay.json`.

```bash
/tmp/n4a-late-hpo-clean311/bin/python -I /tmp/n4a-late-hpo-demo/verify.py \
  --archive /tmp/n4a-late-hpo-replay/late-fusion.n4a \
  --dataset /tmp/n4a-late-hpo-replay/prediction_dataset.json \
  --expected /tmp/n4a-late-hpo-replay/report.json \
  --deny-root /home/delete/nirs4all --deny-root /tmp/n4a-late-hpo-installed \
  --deny-root /tmp/n4a-u10-late-continuous
```

| Wheel | Bytes | SHA256 |
| --- | ---: | --- |
| `nirs4all-1.0.1-py3-none-any.whl` | 2853749 | `482e12e69878e1206cda80fffc3ff7a2b3ceefaa1d72da9b9be67f3665acf0e1` |
| `dag_ml-0.3.25-cp311-abi3-manylinux_2_34_x86_64.whl` | 9277643 | `200ff1d9a9c6b6385578af8481ef5e321a9cac94cfe74ccb79307a24acb31670` |
| `nirs4all_io-0.1.18-cp311-abi3-manylinux_2_34_x86_64.whl` | 3565069 | `7f2aa94a88d276ab5ec147e074cdd1a43a02da99455a297e046f0feea0b5ea39` |

Wheel directories are `/tmp/n4a-late-hpo-wheels`, `/tmp/dag-late-hpo-wheels`
and `/tmp/n4a-cm05-clean-wheels`, respectively. The DAG wheel is a release build
with `extension-module,methods-optimizer` and stripped symbols. IO is unchanged
from the preceding milestone. The Python wheel's **599 package files match the
sources byte for byte**; all wheels exclude interpreter caches. Proof:
`/tmp/n4a-late-hpo-wheel-proof.json`.

The Python source distribution also builds successfully. Rebuilding its wheel
reproduces all 599 package files byte for byte; neither interpreter caches nor
private/generated workspaces are included. The sdist is
`/tmp/n4a-late-hpo-sdist/nirs4all-1.0.1.tar.gz` (2,499,703 bytes), SHA256
`9962ecadc158f383ec1dbd8bef8cf580cd506602a5e1595179a9a80e0d9635be`.
Proof: `/tmp/n4a-late-hpo-sdist-proof.json`.

Version numbers alone do not identify these development capabilities: older
published packages with the same numbers do not include this implementation.
Archives remain Python host artifacts, with trusted joblib contents and declared
dependency versions; this is not a portable Core, ONNX or cross-language release.
