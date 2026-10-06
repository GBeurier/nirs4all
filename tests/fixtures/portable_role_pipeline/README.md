# Five-model portable RolePipeline capture

This deterministic synthetic test fixture contains the actual five RAW
Methods RolePipeline/N4ME artifacts captured by DAG-ML CV/SELECT/REFIT:
four numeric NIR/image/series/metadata projections and an OOF-trained Ridge
meta-model. It is test data, not a multimodal generator or the canonical
U07 N-D encoder profile. No third-party dataset or native library is included.

Original source: DAG-ML `30d9050ea8d21b85ca63e032ae581a1d9a6336d0`,
`scripts/smoke_wasm_multimodal_methods_hpo.mjs`; recorded in workspace audit
`_audits/2026-10-01-wasm-complete-predictor-package/full-archive-node-receipt.json`.
Original receipt SHA-256:
`6191402f61ef5e0227ef1bd47ed44eac5f6721cbcc8d4509ed0614f0737aae35`.
Original fixture SHA-256: `af67c6ce0040253efca5cef7d67ca4e443f1b89eaffcce3df93a0bb47e0028c2`.
Current fixture SHA-256: `a7182661d36ddc2d777813c57648459a7b67c8a4735d4029c8ea038b3e1bbfd5`.

Requalified on 2026-10-05 with Python SDK 1.4.1, DAG-ML 0.3.37,
DAG-ML Data 0.2.13, Core 0.4.2 and Methods 1.3.2. The original DAG-ML
0.3.32 capture scored accuracy using absolute error below 0.5; current
accuracy and balanced accuracy compare rounded class identities, consistently
with F1. The original outcome therefore fails current native OOF validation.
`regenerate.py` repeats native CV/SELECT/REFIT on the exact original numeric
inputs and selected Ridge parameters. All ten OOF average prediction blocks,
truth, unit order, selected variant, signed training request, trusted manifest
and target-free heldout inputs are unchanged. DAG-ML generates new score
reports, outcome/package fingerprints and the signed replay request; Methods
generates new RAW N4ME states. No tolerance or semantic validator is changed.

To reproduce locally with the original receipt (its SHA-256 is checked):

```bash
.venv/bin/python tests/fixtures/portable_role_pipeline/regenerate.py \
  /path/to/full-archive-node-receipt.json \
  tests/fixtures/portable_role_pipeline/five_model_capture.json
```

The retained subset contains the signed training request, exact captured
package/outcome JSON, trusted controller manifest, signed target-free heldout
request/envelopes, four current numeric source rows and expected prediction.
HPO history, training inputs and large evidence logs are omitted. The expected
heldout prediction is retained from the original capture to check numerical
interoperability with the requalified states. The original
Methods WASM capture used ABI 2.15; Python replay of these exact N4ME states
was qualified with Methods 1.2.1/ABI 2.14 in the linked audit.

The SDK integration test exports this complete package as Core `.n4a`, reads
it using the public SDK facade, hydrates a fresh Methods controller and
replays without fit or targets. Set `NIRS4ALL_REQUIRE_PORTABLE_ARCHIVE_V2=1`
for a required native gate. Optionally point
`NIRS4ALL_ROLE_PIPELINE_CAPTURE_JSON` to a fresh complete qualification
receipt to repeat the same test with newly captured artifacts.
