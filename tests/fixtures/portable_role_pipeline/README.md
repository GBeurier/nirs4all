# Five-model portable RolePipeline capture

This deterministic synthetic test fixture contains the actual five RAW
Methods RolePipeline/N4ME artifacts captured by DAG-ML CV/SELECT/REFIT:
four numeric NIR/image/series/metadata projections and an OOF-trained Ridge
meta-model. It is test data, not a multimodal generator or the canonical
U07 N-D encoder profile. No third-party dataset or native library is included.

Source: DAG-ML `30d9050ea8d21b85ca63e032ae581a1d9a6336d0`,
`scripts/smoke_wasm_multimodal_methods_hpo.mjs`; recorded in workspace audit
`_audits/2026-10-01-wasm-complete-predictor-package/full-archive-node-receipt.json`.
Original receipt SHA-256:
`6191402f61ef5e0227ef1bd47ed44eac5f6721cbcc8d4509ed0614f0737aae35`.
Fixture SHA-256: `af67c6ce0040253efca5cef7d67ca4e443f1b89eaffcce3df93a0bb47e0028c2`.

The retained subset contains the signed training request, exact captured
package/outcome JSON, trusted controller manifest, signed target-free heldout
request/envelopes, four current numeric source rows and expected prediction.
HPO history, intermediate scores, training targets and large evidence logs
are omitted. The native contract fingerprints are unchanged. The original
Methods WASM capture used ABI 2.15; Python replay of these exact N4ME states
was qualified with Methods 1.2.1/ABI 2.14 in the linked audit.

The SDK integration test exports this complete package as Core `.n4a`, reads
it using the public SDK facade, hydrates a fresh Methods controller and
replays without fit or targets. Set `NIRS4ALL_REQUIRE_PORTABLE_ARCHIVE_V2=1`
for a required native gate. Optionally point
`NIRS4ALL_ROLE_PIPELINE_CAPTURE_JSON` to a fresh complete qualification
receipt to repeat the same test with newly captured artifacts.
