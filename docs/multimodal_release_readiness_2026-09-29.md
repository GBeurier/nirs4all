# Multimodal release readiness — updated 30 September 2026

This is software qualification using synthetic test fixtures, not a real-corpus or paper result. The existing NIRS generator remains the product generator. No stable version has been bumped or published in this train.

The scientific source qualified below is nirs4all `22902524078669c99eb5b9070a0498c058fc3cc3`, with DAG-ML `9d7273a718aff7c1e511dc8708b13e7bfe8f5613` and IO `5a887788089568955e7011ad0036264dd3f6e979`. Later changes to this report are documentation only.

| Gate | Result | Scope or remaining limit |
| --- | --- | --- |
| All nirs4all examples | **85/85 passed**, zero skips, warnings or failures; strict runner `./run_ci_examples.sh -c all -j 2 -k`, `JAX_PLATFORMS=cpu`, 23m01s | Covers public multimodal examples U07–U14, archive/replay and the framework examples. JAX runs on CPU in this WSL. |
| Complete nirs4all integration suite | **2,692 passed, 20 skipped, zero failures**, 74m10s; `N4A_ENGINE=dag-ml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --timeout=300 -n 2 --dist worksteal -rs tests/integration` | Full suite on the exact source above. Optional native and AutoGluon cases also have separate executed evidence below. |
| Current nirs4all CI | **9,947 passed, 21 expected skips**, lint/type checks green | [Exact-source CI](https://github.com/GBeurier/nirs4all/actions/runs/36633344135). |
| Native opt-ins | **33/33 passed** in the dedicated candidate-wheel profile; two Methods V3 cases passed again with explicit `N4M_LIB_PATH` | Methods source runtime is not a substitute for qualifying newer Methods bindings as a separate release. |
| Real AutoGluon | **8/8 passed**, AutoGluon Tabular 1.6.3, with a JUnit gate refusing missing cases or skips | [Exact-source CI](https://github.com/GBeurier/nirs4all/actions/runs/36633344135): regression/classification, model/framework, in-process/subprocess and archive replay. |
| sklearn/PyTorch provider | U11 passed locally with workers=0; **workers=2 passed in CI**; IO CPU adapter job **70/70 passed** | [nirs4all CI](https://github.com/GBeurier/nirs4all/actions/runs/36633344135), [IO CI](https://github.com/GBeurier/nirs4all-io/actions/runs/36627802834). WSL blocks the multiprocessing socket locally. |
| Installed scientific provider | X/y assembly, CV/archive, HPO/archive, resume and **45 sampler/pruner compositions passed** from exact candidate wheels and packaged runtimes | The script rejects source-checkout imports; wheel hashes and native identities are checked on runners. |
| Linux Studio installed application | **Passed**: exact DEB/runtime, installed four-source import → run → archive → replay, four predictions without fit, populated upgrade from public 0.14.0 | [Candidate `583cdd72`](https://github.com/GBeurier/nirs4all-studio/actions/runs/36643081259). Nonpublishing proof, not the final stable artifact. |
| Windows x64 / macOS arm64 installed application | Exact packaged wheels/native identities and initial installed journeys passed; the final separate multimodal smoke failed to display the completed run within 5 seconds | API completion succeeded. Studio `8a1ffa77` allows 30 seconds for the renderer's 10-second refresh, retaining the visible UI assertion. [New candidate](https://github.com/GBeurier/nirs4all-studio/actions/runs/36649448536) must prove the correction. |
| CUDA hardware integration | One optional test remains unexecuted on this WSL | No usable CUDA device/NVML access. This train has CPU evidence; it must not claim a GPU qualification. |
| macOS Intel | Optional for this train, as requested by the maintainer | No claim of qualification without that platform's own completed gate. |

## Promotion conditions

The maintainer has withdrawn Astra/Claude reviews from the current gate. No further review session is required or running.

1. Confirm current Studio CI, including Windows Rust containment, and installed candidate journeys on Windows x64 and macOS arm64. The candidate remains nonpublishing while any required platform fails.
2. Align final DAG-ML, IO, nirs4all, Tools and Studio versions and immutable source/wheel/native identities. Build and validate the final distributions; candidate proofs do not automatically qualify changed release artifacts.
3. Publish only after the applicable technical gates pass. Record the CPU scope and optional hardware/platform omissions explicitly.
4. Keep newer Methods `main` (LVSE/GCU and optimizer bindings) in a separate qualification phase before raising Studio's native Methods pin, currently published 1.2.1/source `b8b942ae`.

Current planning is maintained in the workspace's consolidated backlog. Shared UI, licensed MATLAB and real-corpus/paper work are outside this release train.
