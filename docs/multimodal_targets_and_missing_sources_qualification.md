# Synthetic targets and missing-source qualification

Development qualification executed on 18 September 2026, using synthetic inputs
only. Real measurements and scientific-paper work are deferred. This record
extends the [first demonstration](multimodal_qualification.md) and its
[installation proof](multimodal_installation_qualification.md).

## Implemented behavior

- `MultimodalClassifier` preserves string and numeric labels, including int64
  labels near their maximum value. Encoding and task inference use training
  targets only; held-out unknown classes are rejected before fitting. Task
  inference precedes the historical float32 numeric-storage conversion.
- Regression accepts named output columns. `target_policy="per_target"` fits
  independent encoder/model chains using observed target cells only. Native
  DAG scoring respects target validity masks and emits metrics per output.
- IO aligns incomplete sources with `source_alignment="left"` and records
  boolean presence masks. `missing_source_policy="zero_with_indicator"` fits
  and transforms each encoder on present rows only, fills absent embeddings
  with zero, and appends a weighted presence column for each source.
- Source and target masks compose: each target's encoder sees the intersection
  of present source rows and observed target rows within its training fold.
  A source absent from that entire training scope is refused. An entire source
  may be absent at prediction time, without calling its encoder.
- Durable native HPO includes training masks in its fingerprint and ignores
  hidden cells and held-out data. R² defaults to maximization; explicit tuning
  directions remain effective. Classification validation folds must not overlap,
  since the existing native resampling reducer averages numeric predictions.
- Captured archives preserve all fitted components and accept new identities,
  source mapping order and presence patterns with the same declared source schema.

The [user guide](source/user_guide/data/multimodal.md) describes these APIs and
their explicit modelling choices. These are Python host implementations using
native DAG scheduling and scoring; no portable Core implementation is implied.

## Executable demonstrations

```bash
.venv/bin/python examples/user/02_data_handling/U08_multimodal_targets.py \
  --output /tmp/mm-targets
.venv/bin/python examples/user/02_data_handling/U09_multimodal_missing_sources.py \
  --output /tmp/mm-missing --stop-after 1
.venv/bin/python examples/user/02_data_handling/U09_multimodal_missing_sources.py \
  --output /tmp/mm-missing --resume
.venv/bin/python examples/user/02_data_handling/U09_multimodal_missing_sources.py \
  --fusion intermediate --output /tmp/mm-missing-mbpls
```

All examples use four raw modalities and three grouped folds. U08 covers string
classification, two named regression targets and partially observed targets.
U09 combines missing sources with partial targets, using early or intermediate
MBPLS fusion. Its new cohort has 12 observations and no images; NIR, series and
metadata are present for 9, 10 and 10 observations respectively.

The scripts write JSON dataset/pipeline declarations, native score reports,
durable trial histories, prediction inputs and complete archives. U09's search
stopped after one trial and resumed to two with exactly the same trial records,
native scores and predictions as uninterrupted execution. U07/U08/U09 also pass
the official runner with `-c user -n 'U0[789]_multimodal*.py' -p -s` when the
development virtual environment is on `PATH`.

## Installed training and replay

Three source-built wheels were installed in
`/tmp/n4a-cm04-final-clean`, without editable installations. Standalone copies
of U07/U08/U09 live in `/tmp/n4a-cm04-final-example` and run with `python -I`.
All five installed training cases reproduced the source executions' predictions,
native reports and trial records exactly.

| Case | Native pooled CV metric | New prediction shape | Replay comparison |
| --- | ---: | --- | --- |
| U08 classification | balanced accuracy 0.717948717948718 | `(12,)` | Exact labels |
| U08 complete regression | RMSE 0.5425889101263331 | `(12, 2)` | Maximum absolute error 0 |
| U08 partial targets | RMSE 0.5304851188376025 | `(12, 2)` | Maximum absolute error 0 |
| U09 missing sources, early | RMSE 1.2044297382199076 | `(12, 2)` | Maximum absolute error 0 |
| U09 missing sources, MBPLS | RMSE 0.9106490266139335 | `(12, 2)` | Maximum absolute error 0 |

These measurements verify fixture behavior. They do not rank methods on real
measurements. The native tuning objective averages fold metrics; the table uses
pooled OOF scores, so those two reported quantities need not coincide.

Each replay received only its archive, new dataset declaration, expected report
and [verification script](../scripts/verify_multimodal_archive.py). An audit hook
forbids access to `/home/delete/nirs4all` and the original training directory
`/tmp/n4a-cm04-final-source`. All relevant `fit`, `fit_transform` and `partial_fit`
methods, plus the legacy runner, are forbidden. All five replays pass with zero
fit calls, verified artifact integrity and no training workspace access.

Source artifacts: `/tmp/n4a-cm04-final-source`; installed training:
`/tmp/n4a-cm04-verified-installed`; isolated replay inputs:
`/tmp/n4a-cm04-final-replay`. Exact installed/source comparisons are recorded in
`/tmp/n4a-cm04-verified-installed-parity.json`; replay attestations are
`/tmp/n4a-cm04-verified-{classification,regression,masked,missing-early,missing-intermediate}-proof.json`.

Environment: Python 3.11.15, dag-ml-data 0.2.11, Methods 1.0.19, NumPy 2.4.6,
SciPy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0 and Optuna 5.0.0.
The installed environment passes `pip check`. The DAG wheel includes the
`extension-module,methods-optimizer` features; IO includes its native binding.
Wheel versions identify development distributions and do not assert that older
published packages with those numbers contain these changes.

### Initial runtime-qualified wheel identities

The later packaging check below supersedes the IO and DAG distribution files
with cache-free wheels; their Python sources and native binaries are unchanged.

| Wheel | Bytes | SHA256 |
| --- | ---: | --- |
| `nirs4all-1.0.1-py3-none-any.whl` | 2851275 | `4222a1c770ff8f8cd69875adf4c6a4d461b99fa6f0769e6912f8490b1d48ac16` |
| `nirs4all_io-0.1.18-cp311-abi3-manylinux_2_34_x86_64.whl` | 3643634 | `91ee0950e3672eab11dd4952503a9696d51b6a872d76904eddddccc2374b6fdf` |
| `dag_ml-0.3.25-cp311-abi3-manylinux_2_34_x86_64.whl` | 9299089 | `273077f4947dae6e2fc02d15e942460fa949bb4f4ed171a464f4982ca1e80935` |

These files are in `/tmp/n4a-cm04-verified-wheels`,
`/tmp/io-source-presence-wheels` and `/tmp/dagml-masked-regression-wheels`
respectively. All installed training and replay checks above used these exact
wheels. They are local qualification artifacts, not a published release.

For example, one of the five isolated replay commands was:

```bash
/tmp/n4a-cm04-final-clean/bin/python -I \
  /tmp/n4a-cm04-final-replay/verify.py \
  --archive /tmp/n4a-cm04-final-replay/missing-intermediate/multimodal.n4a \
  --dataset /tmp/n4a-cm04-final-replay/missing-intermediate/prediction_dataset.json \
  --expected /tmp/n4a-cm04-final-replay/missing-intermediate/report.json \
  --deny-root /home/delete/nirs4all --deny-root /tmp/n4a-cm04-final-source
```

## Verification

Dedicated gates passed: IO 485 tests with two optional-format skips; extracted
native IO binding 279 tests; DAG Rust workspace 847 tests with three ignored,
formatting, Clippy and W1/D4/criteria contracts; 89 operator tests with both
scikit-learn 1.9.1 and 1.5.2; all 79 multimodal integration tests with
scikit-learn 1.5.2, including 15 missing-source cases; target scope/precision,
masks, archive, classification-fold and tuning regressions. The minimum-version
run includes the wrapper classifier-identification correction. Ruff passes;
package mypy with `--no-incremental` passes on 528 source files.

The final full suite passed **10,563 tests, with 113 skipped and 6,279 warnings,
in 683.54 seconds**. Log: `/tmp/n4a-cm04-verified-full.log`. The full-suite
environment uses an isolated Methods 1.0.18 installation for the existing
version-pinned historical witness; installed demonstrations use Methods 1.0.19.
The correction forwarding prediction options into native voting is included.
The wheel's 591 packaged Python modules match the current sources byte for byte.

Commands run from the Python repository, through the local `rtk proxy` wrapper:

```bash
env PYTHONPATH=/tmp/n4m018-multimodal-qualification \
  .venv/bin/pytest tests/unit tests/integration -n 8 -q --disable-warnings
.venv/bin/ruff check .
.venv/bin/mypy --no-incremental nirs4all
.venv/bin/python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /tmp/n4a-cm04-verified-wheels .
```

The minimum-version integration log is
`/tmp/n4a-cm04-sklearn152-integrations-fixed.log`; final typing and build logs are
`/tmp/n4a-cm04-verified-mypy.log` and `/tmp/n4a-cm04-verified-build.log`.

Sphinx also built the complete HTML documentation without warnings, including
the multimodal guide and generated model API pages. Documentation dependencies
were installed separately under `/tmp/n4a-cm04-docs-qualification/deps`; a copy
of `docs/source` kept generated API files outside the checkout. Output is in
`/tmp/n4a-cm04-docs-qualification/html`, with the build log at
`/tmp/n4a-cm04-docs-build.log` and an empty warnings log at
`/tmp/n4a-cm04-docs-warnings.log`. All 131 local links checked across the current
planning documents, qualification report and user guide resolve.

An isolated source-distribution build also passed. Rebuilding a wheel from
`/tmp/n4a-cm04-verified-sdist/nirs4all-1.0.1.tar.gz` reproduced all **598 package
files byte for byte**, including all new modules and package data, compared with
the installed qualified wheel. The source archive is 2,497,579 bytes, SHA256
`ff1fb9a5ded3b44e7bde931283be65a81c47460aae2a2a1f2718a0e8d680ab3a`.
Proof: `/tmp/n4a-cm04-sdist-proof.json`; build logs:
`/tmp/n4a-cm04-sdist-build.log` and `/tmp/n4a-cm04-sdist-wheel-build.log`.
The archive contains no virtual environment, Python cache, private documentation
directory or trained `.n4a` model. No additional runtime test was needed for the
rebuilt wheel's identical package payload.

## Native source distributions and cache exclusion

Both native packages rebuilt successfully from their source distributions using
`pip wheel --no-deps` in isolated build environments outside the workspace. The
archives included all current runtime sources: 74 Python/Rust files for IO and
71 for DAG matched the checkout byte for byte, with no precompiled libraries.
The resulting local wheels carry `linux_x86_64` tags; this source-build check
does not establish compatibility with other platforms.

All three packages rebuilt from source distributions were installed into the
new `/tmp/n4a-cm05-source-install` environment, which passes `pip check`.
The five U08/U09 cases again produced exactly the same predictions, native
scores and trial records as the source runs. All five isolated archive replays
passed with zero fit calls and workspace access forbidden. Evidence is in
`/tmp/n4a-cm05-runtime-proof.json`, `/tmp/n4a-cm05-installed` and
`/tmp/n4a-cm05-{classification,regression,masked,missing-early,missing-intermediate}-proof.json`.

Comparing these wheels with direct builds exposed local bytecode caches in the
original direct wheels: 12 files in IO and 3 in DAG. Their binding pyprojects now
exclude `**/__pycache__/**`, `**/*.pyc` and `**/*.pyo`, using the documented
[Maturin exclude configuration](https://www.maturin.rs/config.html).
Direct release builds with `--strip` and regenerated source archives contain
zero cache files even though those caches remain in the development checkout.
All other package files, including the native libraries, are byte-identical to
the initially qualified direct wheels. The regenerated source archives differ
from the successfully compiled archives only by this exclusion configuration.

| Corrected native artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `nirs4all_io-0.1.18-cp311-abi3-manylinux_2_34_x86_64.whl` | 3565069 | `7f2aa94a88d276ab5ec147e074cdd1a43a02da99455a297e046f0feea0b5ea39` |
| `dag_ml-0.3.25-cp311-abi3-manylinux_2_34_x86_64.whl` | 9240879 | `35768947ee9a3c3a34d8cc91299dc8d5d341ae3cf7f2e2fcf9034663ab3dfc46` |
| `nirs4all_io-0.1.18.tar.gz` | 390857 | `b72bc29d0c11a6581108e6214155a2c181aefb1274861655903135f930a42b15` |
| `dag_ml-0.3.25.tar.gz` | 897921 | `20edba9de16828f581e571edad5d4976bcb4acadd07f19b3b86b7da503ad237b` |

Corrected wheels are in `/tmp/n4a-cm05-clean-wheels`; corrected source archives
are in `/tmp/n4a-cm05-clean-sdists`. Their inventory and digests are recorded in
`/tmp/n4a-cm05-clean-distributions-proof.json`. This packaging-only correction
does not change the runtime covered by the full non-regression gate above.

## Remaining boundaries

Partial-target scoring currently covers sample-level regression. Partial source
or target masks in late fusion, masked classification targets, mixed tasks per
output, ragged tensors and cross-language replay remain outside the qualified
profile. Partial source pipelines keep their encoders inside the multimodal
operator. Archives retain trusted Python objects and require their recorded
dependencies. No packages have been published as part of this work.
