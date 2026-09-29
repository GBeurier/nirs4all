# Multimodal release readiness — 29 September 2026

This is a software qualification using synthetic fixtures, not a real-corpus or paper result. No stable version was bumped and nothing was published.

| Gate | Local result | Remaining limit |
| --- | --- | --- |
| All nirs4all examples | 85/85 passed, 0 skipped, 0 warnings, strict runner `./run_ci_examples.sh -c all -j 2 -k` with `JAX_PLATFORMS=cpu` (24m37s) | The CPU setting avoids the installed JAX CUDA plugin's startup traceback on this GPU-less WSL. |
| Full nirs4all integration | 2,675 passed, 25 skipped, 0 failed: `N4A_ENGINE=dag-ml JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --timeout=300 -n 2 --dist worksteal -rs tests/integration` (1h11m) | Six failures in the first pass were five stale preflight mocks and a Triton import crash; the mocks were updated and Triton removed locally, matching the release workflow's CPU configuration. |
| Opt-in native profiles | 33/33 passed with current installed candidate DAG-ML/IO/nirs4all wheels, local Methods 1.2.1/ABI 2.15 library and `NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2=1`, `NIRS4ALL_REQUIRE_NATIVE_METHODS_HPO=1`, `N4M_LIB_PATH` and `NIRS4ALL_CORE_LIVE_METHODS_LIBRARY` | This covers the 16 native tests skipped in the general run, including Archive V2, HPO, Studio installed, V3 refit, Methods witness and CLI export. Methods was installed from the current source, not its exact published wheel. |
| Installed multimodal provider | `python -I tests/qualification/installed_multimodal_provider.py` passed from outside the checkout with local installed wheels: X/y assembly, CV/archive, HPO/archive, resume and all 45 sampler/pruner combinations | The qualification guard now checks the three packages are inside `sys.prefix`; its negative source-checkout check also passes. |
| Optional integration surfaces | 8 AutoGluon tests and 1 GPU test were skipped | AutoGluon is absent and this WSL has no usable CUDA device. They were not run. |

The installed wheel SHA-256 values used for the final local provider check are:

- nirs4all `1.3.1.dev0`: `3be39014c8298cce3823e7e59490aebf7d3202787c5987301738372aab810f2c`
- DAG-ML `0.3.31.dev0`: `020213a9d87a9d85912c1b67ba92cf6a9731c2fc9a5f08087849af352c3bccf8`
- nirs4all-io `0.2.1.dev0`: `46bae08cda9497ad22af028fed360d946ee3bf91f3a971ea882d18a197d47461`

Publication gates remain open. The nonpublishing Studio candidate workflow now exercises the installed provider from both extracted and installed `.deb` runtimes, but its runner result cannot be read from this WSL. Local Electron UI smoke stops at sandbox `listen EPERM`, so installation/migration and Windows/macOS installers of the exact candidate are not proven here. GitHub Actions and Claude Opus 5.5 are unreachable from this environment; Astra's final static review found no P0/P1 in the publication gates. Local `pip check` reports that this venv's Torch 2.14 requires Triton, deliberately removed after its import crashed; the release workflow uses a CPU Torch wheel and removes Triton. These limits block a stable bump or publication from this qualification alone.
