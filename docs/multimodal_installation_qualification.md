# Multimodal installation and replay qualification

Executed on 2026-09-17 using synthetic inputs only, Linux x86_64 and Python
3.11.15. No real corpus or scientific-paper work is part of this qualification.

Three wheels built from the modified sources were installed in a new virtual
environment, without editable installations or sibling source paths:

| Wheel | Bytes | SHA256 |
| --- | ---: | --- |
| `nirs4all-1.0.1-py3-none-any.whl` | 2845073 | `ca038304022f6f4e1893bec4b2a6da78c19c885496ee26c8b5f69bd7d0c77c2e` |
| `nirs4all_io-0.1.18-cp311-abi3-manylinux_2_34_x86_64.whl` | 3640418 | `6d2902416912550953457096a97675c4df1a0b6a3b61441d6a117ce67f04ae36` |
| `dag_ml-0.3.25-cp311-abi3-manylinux_2_34_x86_64.whl` | 9315957 | `f3325be4f9b257aec44daf5c5b54f48661671680b5d080b45122aa45753dd4a6` |

The IO wheel includes its Rust `_native` binding. The DAG wheel was built with
`extension-module,methods-optimizer`, preserving the existing native Methods
training profile. These are development artifacts; the same version numbers
on older published wheels do not imply that they contain these changes.

Dependencies used for the numerical path: dag-ml-data 0.2.11,
nirs4all-methods 1.0.19, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1 and
joblib 1.6.0. The archive records its exact host dependencies.

## Installed training and durable resume

The source demonstration was stopped after two native trials and resumed to a
total budget of eight. A copied U07 script then ran all eight trials from the
installed wheels, using `python -I` from `/tmp`. Both executions produced
identical trial parameters, scores and new-input predictions.

- Selected parameters: `model.alpha=1.0`, `source_weights.image=0.5`,
  `transformers.image.n_components=2`.
- Native CV RMSE: `0.04527561590802293`.
- Held-out test RMSE: `0.033157834301111466`.
- Twelve predictions on a new, unlabelled synthetic cohort.

These scores verify execution and reproducibility on the fixture. They are not
evidence of performance on real measurements.

## Installed replay with training and source access forbidden

Only the archive, `prediction_dataset.json`, expected report and the copied
[verification script](../scripts/verify_multimodal_archive.py) were provided to
the replay process. The script installs an audit hook that rejects access to
the original repository tree and training output directory. Every relevant
encoder/model `fit`, `fit_transform` and `partial_fit`, plus the legacy runner,
is replaced by an assertion that fails if called.

```bash
/tmp/n4a-multimodal-clean/bin/python -I /tmp/n4a-multimodal-isolated-replay/verify.py \
  --archive /tmp/n4a-multimodal-isolated-replay/multimodal.n4a \
  --dataset /tmp/n4a-multimodal-isolated-replay/prediction_dataset.json \
  --expected /tmp/n4a-multimodal-isolated-replay/report.json \
  --deny-root /home/delete/nirs4all --deny-root /tmp/n4a-multimodal-final
```

Result: **passed**, 12 predictions, **maximum absolute error 0**, zero fit calls,
artifact integrity verified. The imported library was
`/tmp/n4a-multimodal-clean/lib/python3.11/site-packages/nirs4all/__init__.py`.
Tolerance was `atol=rtol=1e-9`.

Verified archive SHA256:
`2fd0acd2bd4ad408804c91f24c87a45c8703358a98c1556889f55adc68a9856b`.
The complete local replay report is `/tmp/n4a-mm-isolated-proof.json`.

## Reproduce with current sources

Build nirs4all with `pip wheel --no-deps`. Build IO from
`nirs4all-io/bindings/python` with `maturin build --release --strip`. Build
DAG from `dag-ml/crates/dag-ml-py` with
`maturin build --release --strip --features extension-module,methods-optimizer`.
Install the three resulting wheels together in a fresh Python 3.11 environment
with the numerical versions above. Wheel digests identify this particular
build; later source changes require a new qualification record.

Copy `U07_multimodal.py` outside the checkout and run it with that environment's
`python -I`, using a new output directory. Follow the
[user guide](source/user_guide/data/multimodal.md) for stop/resume and replay.
Copy the four replay files elsewhere and invoke the verification script with
the original repository and training directories listed under `--deny-root`.

The [14-case qualification](multimodal_qualification.md) separately covers
per-source baselines and early, intermediate and late fusion. This installation
proof covers the complete four-source early-fusion pipeline with durable tuning.
It does not establish cross-language or portable Core archive support.

## Final regression checks

The complete unit/integration gate passed: **10,417 passed, 113 skipped** in
693.52 seconds. It used eight pytest workers and the corrected native IO/DAG
wheels. One existing Methods witness pins 1.0.18 exactly, so this full gate
loaded an isolated installed copy of that wheel using
`PYTHONPATH=/tmp/n4m018-multimodal-qualification`; no sibling source checkout was
added. The installed demonstration above and targeted native/tuning tests also
passed with Methods 1.0.19.

Ruff over the Python repository and mypy over the package plus demonstration,
qualification and replay-verification scripts passed (531 files). IO's source
suite passed 375 tests with two optional-format skips. DAG's Rust workspace
tests, Clippy, formatting, W1/D4 contract gates and native Python binding tests
passed. Both the development and isolated runtime environments pass `pip check`.

Local full-gate log: `/tmp/n4a-mm-full-final.log`.
