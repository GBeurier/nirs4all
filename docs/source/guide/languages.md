# 10. Use your language's runtime

Keep the common dataset and pipeline contract stable while expressing file I/O, arrays, errors and resource ownership idiomatically in each host. Language selection is synchronized across the tabs in this guide during a browser session.

| Language | Data and model conventions | Runtime/resource considerations |
|---|---|---|
| Python SDK | NumPy/pandas and sklearn-compatible controllers; Torch/TF/JAX optional | Select supported engine explicitly when needed; close result/session/native objects |
| Python Core | IO public dataset and JSON-native task facade | Resolve Core CLI and Methods library; importing Core should not eagerly load optional providers |
| R | Matrices/data frames and dedicated product functions | Local R workflow differs from native Core workflow; retain CLI and model runtime across reload/resume |
| JS/TS/WASM | Typed arrays, JSON records and async initialization | Initialize each worker; separate browser storage from Node files; close/dispose native resources |
| MATLAB/Octave | Matrices, structs, namespace functions and MEX | Preserve input dtype/schema, UTF-8 and runtime paths; use the qualified package layout |
| Rust | Typed contracts and native bindings | Use dynamic Methods library with matching ABI; pass data identities and own resource lifetimes |
| CLI | JSON/YAML, paths, stdout or new output files | Read help for the exact group; avoid overlapping publication destinations |

## Add a host model

A host controller must describe accepted task/representations and fitting/prediction behavior. Keep fitting buffers and learned model binaries in that host. DAG owns phase scheduling and result identity. Prediction import must validate the saved input schema and artifact binding before invoking the host model.

Do not block a synchronous WASM callback on a promise. Asynchronous JavaScript estimators need an execution host that can await their callbacks. See the [Core classical-ML guide](https://github.com/GBeurier/nirs4all-core/blob/main/docs/CLASSIC_ML_JS.md) for the qualified libraries and serialization conventions.

The {doc}`Python practices </user_guide/python/index>`, {doc}`controller guide </developer/controllers>` and {doc}`runtime interfaces </reference/public_interfaces>` explain extension points. R documentation is distributed with the R package; Core documents Rust, JS and MATLAB surfaces. Always use the API name exported by that product rather than translating a Python name mechanically.
