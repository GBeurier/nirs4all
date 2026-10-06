# Bug audit resolution — 2026-10-05

This ledger records the implementation of [the authoritative audit](bug_audit_2026-10-05.md), including its section 6 policy for the deprecated legacy engine. The original findings remain available separately from their implementation outcome.

**Release candidate: nirs4all 1.4.2. Local qualification is recorded below; SDK publication remains pending.** Artifact hashes and the final source commit are recorded outside this document to avoid a recursive source-archive digest.

## Integrated public cohort

| Distribution | Version |
|---|---|
| nirs4all | 1.4.2 |
| dag-ml | 0.3.37 |
| dag-ml-data | 0.2.13 |
| nirs4all-core | 0.4.2 |
| nirs4all-io | 0.2.5 |
| nirs4all-methods / pls4all | 1.3.2 / native ABI 2.17 |
| nirs4all-formats | 0.2.11 |

All seven upstream wheels were downloaded from PyPI and independently checked against the live registry's whole-file SHA-256 values. They are installed in the SDK source environment, separate installed-SDK child environments and Studio. Both Methods bindings contain the same public native binary, reporting `1.3.2+abi.2.17.0`; its SHA-256 is `08b284369f28edc6f261ca41a7865661917f835354d221d239f9cde8ac357e86`. The public tag's Python metadata helper is retained. The upstream publisher confirmed the completed cohort and no further upstream patch release was pending.

## Validation

The broad local evidence below was obtained against the 1.4.1 baseline. The final review then repaired CTD-13, CTD-16, CTD-18 and CFG-11 locally, with 116 passing affected cases including 21 new regressions. Restoring the immutable baseline reproduces fourteen expected failures while retaining seven negative controls. The 1.4.2 package has five changed runtime files: those four repairs and the version declaration. Its additional validation is recorded separately below; the earlier baseline runs are not relabelled as new 1.4.2 invocations.

The final unit/regression run passes **11,411 tests**, with one CUDA-only skip, in 268.85 seconds. That exact CUDA node subsequently passes independently on the actual GPU in 3.48 seconds. Production, upstream payloads and the DAG CLI remain unchanged throughout the unit run.

The installed-SDK API run covers **1,041 unique cases**: 1,040 pass and one native topology overlap witness exceeds its 300-second deadline under the concurrent broad-suite load. Both unchanged native overlap witnesses subsequently pass in an isolated run, retaining the same deadline, inputs, independent numerical oracles and serialized negative controls. Their concurrent peak activity is two, versus one for the serialized controls; every native call is recorded without overflow. The broad run's exit 1 remains diagnostic evidence. The combined evidence qualifies all 1,041 unique cases; it is not described as a single green broad run.

The frozen parity diagnostic records **1,563 passes, one stale fixture failure and nine skips** across 1,573 unchanged cases (1,681.13 seconds). The fixture previously expected an unusable archive to be created; it now checks the intended refusal before creating or replacing either archive format. The corrected node and its complete five-case file pass with unchanged production and all other test bodies. The original parity CUDA skip subsequently passes on the actual GPU in 3.78 seconds. Combined evidence covers **1,565 unique passing parity nodes**, with eight AutoGluon skips in that invocation. All eight original real-AutoGluon cases subsequently pass in an isolated official CPU profile against 1.4.1 in 30.30 seconds, completing 1,573 unique baseline parity cases; the original diagnostic remains retained. The original broad exit 1 remains recorded; the complete suite is not relabelled as a fresh green invocation.

The other-integration run passes **all 609 cases** in 106.98 seconds. The e2e run passes **all four cases** in 121.33 seconds. Both retain their original collections and unchanged production/cohort/CLI. Ruff passes across source, tests and scripts; mypy passes across 586 source files; Sphinx builds with restored network access. The actual public Methods binary passes the existing usability benchmark thresholds: wall-time ratios 1.025 and 1.103, RSS ratios 1.017 and 1.022, maximum held-out prediction difference zero and OOF score difference below 3.4e-9. No numerical tolerance or performance threshold was loosened.

The CPU qualification uses the official PyTorch `2.14.0+cpu` wheel in separate environments. The original GPU environment and its TensorFlow, JAX CUDA plugins and required Triton remain intact. Declared cold Torch models now prepare Torch-owned runtime imports before global framework seeding, without constructing user models or optimizers; canonical deserialization remains after seeding. The repair passes 209 unique focused cases and fourteen actual installed-SDK cold-process API cases. The unrelated long-lived GPU TensorFlow-before-Triton native import conflict remains an explicitly unqualified runtime profile; the independent CUDA scope pass does not qualify that mixed-framework broad profile.

Earlier source-changing, stale-cohort, GPU-crash and disk-exhaustion runs are retained as diagnostics, not release passes. Focused batches overlap and their counts must not be added together.

## Scientific and artifact contracts

Legacy supervised preprocessing remains fold-local for ordinary CV/refit, Optuna/n4m tuning, multi-source merges, existing augmented rows, OOF meta-features and fitted replay. Unsupported augmentation after the retained supervised prefix, supervised by-source routing and feature merges after supervised branch preprocessing refuse explicitly. X-only Mixup/LocalMixup remains directly supported; controller sample augmentation refuses it until joint X/y weights and multiple-parent identity are implemented.

Explicit legacy `allow_no_cv=True` yields warned in-sample training predictions with zero OOF folds. Declared CV still requires actual validation predictions and minimum coverage. Its 63-case gate retains independent native nested-OOF checks. Learned global MSC preprocessing is not qualified as fold-local by these tests.

Metadata separation now selects the same filtered physical sample IDs as X. Its 136-case gate passes, and restoring the original implementation makes 26 of the 40 new witnesses fail. Mixed text metadata is losslessly marshalled as object storage only when the captured schema declares object metadata; fixed-width saved schemas still refuse mismatches. All 30 metadata/U07 cases and nine installed-wheel replay cases pass against the public tag helper. Explicit source concatenation feeds each block once; five cases check unequal-width independent oracles, while the implicit early-fusion default and persisted historical layout flags are preserved.

Parallel legacy branches own separate runtime/artifact/trace coordinators and merge their records deterministically after joining. Seventy-four unique cases pass, including seventeen new concurrency, physical CV preprocessing, independent refit and archive witnesses. The strict parallel example passes all six scenarios. Prediction charts rank CV candidates and exclude final rows by default; all 207 visualization cases pass.

Legacy prediction joins refuse unsupported multi-target outputs before mutating context or arrays; classification probability columns retain their distinct contract. Its 211-case gate passes. The exact selected final MetaModel artifact now survives cleanup and cold store reopen, checked against an independent Ridge oracle. Legacy final MetaModel REFIT features remain full-TRAIN base predictions, explicitly in-sample rather than OOF. The two final legacy stacking archive profiles lack captured raw-input dependency closure and refuse before creating or overwriting either archive format; this does not claim their raw-input replay is supported. The 298-case export/store/controller gate also retains valid scalar base CV/refit exports.

Two Rust tests produced real XL03 Archive V3 train/refit captures. Fourteen installed-SDK capture-consumption cases pass in the current public-cohort API run. These checks preserve actual fitted-state replay and do not turn external helper development dependencies into a public-cohort claim.

The historical full 95-example diagnostic records 89 passes and six failures. The visualization example passes after repair; three JAX examples pass the unchanged strict validator in an explicit CPU subprocess profile, with the original GPU plugins restored; the parallel example passes its six original scenarios. Octave is unavailable locally. This evidence is not a fresh all-95 pass. The release workflow retains the full installed-example gate and prepares Octave.

## Additional 1.4.2 validation and release gates

The actual 1.4.2 wheel and source distribution pass Twine and complete archive review: 655 package payloads match current source, 670 RECORD entries verify, and all source-distribution payloads, dependency/extra metadata, licences and entry points pass. The wheel is installed in a separate interpreter with the seven actual public upstream wheels; all 655 installed SDK payloads and upstream RECORD/native origins verify.

The five mandatory cold replay witnesses that failed in the original release CI pass against 1.4.2 in 23.41 seconds when parent and child use the same actual public DAG binary. Their training, independent prediction oracles, source/extension/archives SHA-256 guards and no-FIT/no-HPO replay controls remain intact. The CI preparation explicitly aligns the fresh parent with the same seven selected public wheels. Actual isolated preparation verifies every upstream Python/native payload and matching parent/child DAG and Methods binding hashes; its five unchanged cold witnesses pass in 24.19 seconds. It retains the source SDK, CLI and Torch profile, with no dependency resolver mutation. All eight actual AutoGluon cases also pass against 1.4.2 in 30.40 seconds, and the unchanged CUDA scope case passes on the real GPU in 1.96 seconds.

The final frozen 1.4.2 unit/regression gate passes **11,449 cases**, with one CUDA-only skip, in 149.97 seconds. All production/test sources, public upstream payloads, Torch and the CLI remain byte-identical throughout the complete run. The exact CUDA node already passes independently on the real GPU as recorded above. The earlier 11,424-pass diagnostic remains retained: its wrapper correctly refused source-freeze qualification because one CI preparation test file changed during that invocation. Ruff passes across source/tests/scripts and mypy across 586 source files; both finalized CI preparation scripts also pass their focused type checks.

GitHub's 1.4.1 publication attempt remains historical failed evidence: the source gate records 12,501 passes, five cold replay hash failures and 35 skips, while the strict 95-example run records 93 passes, one declared TabPFN precondition skip and one Octave replay failure. The latter trained and exported five real Octave models, then failed when a Python bytes payload reached a JSON transport. The upstream tooling-only correction converts bytes to JSON octets. Its exact pinned commit changes only the helper and its test; every other Git blob/mode is identical to the qualified DAG runtime. SDK CI checks that proof before using the separate helper checkout. No numerical model, scientific input, timeout or hash assertion is weakened. The full strict external-runtime and integration gates remain mandatory before publication.

## Studio propagation

Studio's SDK constraints and seven-package runtime cohort are updated locally. Its registry-only Cargo lock resolves the actual public Core 0.4.2 crate, with all 334 registry checksums checked and no local source patches. The exact applied lock passes `cargo check --offline --locked` and all fourteen native archive tests.

Against the installed 1.4.1 candidate and public upstreams, the final frontend diagnostic records 4,240 passes and seven stale document-adapter attestation failures; after regenerating the actual changed member, all nineteen affected cases pass. This qualifies 4,247 unique cases without relabelling the failed full invocation. The backend passes 2,527 cases with three explicit profile skips; the two real Rust HTTP profiles subsequently pass, qualifying 2,529 unique cases. Native default tests pass 322 cases with four ignored profiles; the installed-SDK general request and live two-target conformal producer/store/renderer profiles separately pass, qualifying 324 unique native cases. The packaged-runtime benchmark and separate raw Archive V2 fixture remain unavailable profiles. Eight lint gates pass.

The curated Derivate schema now exposes the advanced integer axis parameter, default 1 for wavelengths, in both standard and extended editor modes. Its generator metadata and actual extended output agree. Two new registry witnesses and all sixty-seven affected registry/document-adapter/plugin cases pass after re-attestation. Final SDK 1.4.2 publication URL, whole-wheel SHA, source commit and installed/callable manifests remain pending actual PyPI acquisition and final consumer qualification. Studio's app version remains 0.15.0; this task propagates the SDK/cohort update.

## Audit dispositions

A corrected entry can represent a numerical repair, an explicit capability refusal or a documentation/contract correction. It does not mean every historical allegation described a scientific bug. Duplicate IDs identify their canonical finding. Minor deprecated-engine findings remain deferred when a complete verified repair exceeds section 6's small local-change exception.

| Disposition | Findings |
|---|---:|
| Corrected / contract enforced | 244 |
| Deferred by audit policy | 24 |
| Rejected by re-audit | 2 |
| Duplicate | 6 |
| Qualified contract | 3 |

### DGA

| ID | Outcome | Implementation / qualification |
|---|---|---|
| DGA-01 | Corrected / contract enforced | Deep-copy a caller-owned SpectroDataset at materialization; augmentation, metadata, folds and holdout partitions are isolated. Host mutation and public augmentation-then-CV regressions pass. |
| DGA-02 | Corrected / contract enforced | Merge expanded non-reserved sibling model parameters over serialized defaults. Catalog/winner recovery and public CV-only/refit sweeps pass; all variant fold RMSE values match independent sklearn fits. |
| DGA-03 | Corrected / contract enforced | Reject unsupported CV-only per-branch prediction aggregation before execution, and explicitly forward refit on the supported path. This prevents silently refitting; CV-only stacking lowering is still unsupported. |
| DGA-04 | Corrected / contract enforced | Parse dict splitters through _split_pipeline in both duplication checkpoint paths and the interleaved exclusion checkpoint path. Bare/dict/grouped fold tests and public dict-split checkpoint runs pass. |
| DGA-05 | Corrected / contract enforced | Exclude avg/w_avg reports and absent variant IDs from positional label recovery. Both fallback and recovered-winner mappings retain every real CV variant. |
| DGA-06 | Deferred by audit policy | Sibling keys remain in original configuration but are not model_params/force_params; the instantiated estimator is returned unchanged. Correct reserved/control-key discrimination, cloning and override precedence exceeds a complete two-line change. |
| DGA-07 | Corrected / contract enforced | Shell-quote interpreter and adapter paths; real launcher invocation in a directory with spaces passes. |

### DGB

| ID | Outcome | Implementation / qualification |
|---|---|---|
| DGB-01 | Corrected / contract enforced | Shared _CoordinateTransform converts cm-1 to nm only for strict _requires_wavelengths=True operators; Resampler retains cm-1. Both node_runner and run_paths callers use this wrapper, so no run_paths.py edit was needed. Physical-axis absence remains a refusal. |
| DGB-02 | Corrected / contract enforced | Contract decision: preserve the existing DAG early-fusion default (concatenate before plain SNV) and document its difference from legacy per-source preprocessing. A new unequal seven/three-channel public DAG run matches an independent sklearn fused-input refit and demonstrably differs from per-source SNV; existing CLI baseline wording corrected. No scientific default switched. |
| DGB-03 | Corrected / contract enforced | Stacking replay weighting delegates to core.metrics.is_higher_better, fixing F1 inverted weights and avoiding another metric-direction list. Regression is helper-level; full classification archive replay was not established. |
| DGB-04 | Rejected by re-audit | Missing physical wavelength refusal is intentional; no synthetic nanometer coordinates or fallback added. |

### DGC

| ID | Outcome | Implementation / qualification |
|---|---|---|
| DGC-01 | Corrected / contract enforced | MetadataFilter exclude/tag masks receive source metadata reindexed to the exact selected base-sample pool order, including fold-local exclusion. Public DAG runs pass. |
| DGC-02 | Corrected / contract enforced | Non-string JSON scalar mapping keys use the existing pair-preserving marker, recursively including serialized nested estimator payloads. Integer class-weighted direct/nested classifiers and a public DAG classifier run pass. Parent-owned component_serialization.py was not edited. |
| DGC-03 | Corrected / contract enforced | Construct a Polars DataFrame from separate metadata columns, preserving float/int/bool/text types and NaN values; keep row-count validation. |
| DGC-04 | Corrected / contract enforced | Publish conformal bundles through a sibling staging file and os.replace; failed writes preserve previous bytes. |
| DGC-05 | Corrected / contract enforced | Use exact rational rank from decimal coverage rather than binary floating ceil. |

### CTD

| ID | Outcome | Implementation / qualification |
|---|---|---|
| CTD-01 | Corrected / contract enforced | TrainingSetReconstructor now receives an explicit model-name/branch mapping, preserving independent OOF columns for same-named models. |
| CTD-02 | Corrected / contract enforced | Disjoint merge uses full stored-row dimensions and absolute IDs; membership includes test rows and averages test fold predictions. End-to-end excluded-row prediction identities depend on the separately owned CTM-02 repair. |
| CTD-03 | Corrected / contract enforced | Separation membership published to model controllers includes train and test universe IDs. Regression covers metadata, tag and fitted filter routing and the model consumer retaining four held-out rows. |
| CTD-04 | Corrected / contract enforced | Both source-merge syntaxes remove original sources after concat/stack and reset processing selection to the merged source. |
| CTD-05 | Corrected / contract enforced | Branch snapshots preserve indexer/exclusions/tags/augmentation state, targets, metadata, folds and flags together, reconnect accessors on restore, invalidate hashes and preserve updated snapshots for subsequent steps. Both deep-copy and CoW modes verified. |
| CTD-06 | Corrected / contract enforced | Tuple producers now register within their own active branch/substep trace. Persisted names and feature schemas replay through original controllers. Feature selection uses stable source/processing keys; transfer preprocessing loads fitted objects during predict instead of fitting the prediction cohort. Workspace/archive roundtrips cover concat, resampler, transfer (stacked/augmentation), and MC-UVE, including nested branches. Resampler artifact lookup now uses stable source/processing names instead of an operation counter; all23 selector/axis parity cases, including the four reported CARS/MC-UVE augmented-selector/resampler cases, pass. Branch trace follow-up: Filter artifacts are persisted while their parent separation trace is active, before child execution; both duplication and separation parents explicitly close their trace before child steps replace it. Nested branch trace ownership is preserved without suppressing recorder errors or changing EXE-01 scientific qualifications. |
| CTD-07 | Corrected / contract enforced | Verified two-line local minor correction: both source collection paths include excluded stored rows when replacing source arrays. |
| CTD-08 | Deferred by audit policy | Ordinary fitting still reads the train selector without applying custom sample_partition. Consistent branch-group X/y fitting across regular and augmentation transform modes exceeds two lines. Cache identity already includes sample_partition; a claim that it still lacks that key is obsolete. |
| CTD-09 | Corrected / contract enforced | Verified two-line local minor correction: duplication and separation paths publish the actual pre-branch feature snapshot for include_original; regression compares the original merged block to the raw input. |
| CTD-10 | Corrected / contract enforced | Skip an unselected source during final feature/header update. A regression verifies its original nm headers, unit, processing selection and feature rows remain unchanged. |
| CTD-11 | Deferred by audit policy | Source loop still uses all dataset processings and does not honor context.selector.processing. Complete repair needs both selected source/processing filtering and preserved output-context processing lists, beyond two lines. |
| CTD-12 | Deferred by audit policy | Sequential train branches still warn and continue after an exception; predict/explain raise. A one-line unconditional raise would change warning/continue execution semantics, so contract choice is required. Parallel branch coordinator repair does not repair this sequential contract. |
| CTD-13 | Corrected / contract enforced | Fit tagging filters on nonaugmented train X and matching y (including None); apply the fitted mask to the original selected population with explicit augmentation flags. Single/multi-target shape and mask alignment verified. |
| CTD-14 | Deferred by audit policy | by_filter still fits in every mode. Tag only uses loaded_binaries and refits otherwise. CTD-06 tuple replay repair does not provide the complete persisted-filter lookup and no-refit lifecycle for both entry points; that exceeds two lines. |
| CTD-15 | Deferred by audit policy | Dict output still lives in a throwaway context and lacks a downstream multi-input consumer. Full support exceeds two lines; a short rejection would change the documented output_as=dict contract and requires an explicit capability decision. |
| CTD-16 | Corrected / contract enforced | Use both absolute fold ID arrays directly for target histograms; remove obsolete base-ID remapping and IndexError fallback. Excluded samples and stacked/staggered layouts verified. |
| CTD-17 | Deferred by audit policy | Legacy default remains add, while class documentation/example and two reference pages say extend. Existing unit contract explicitly requires add. Correcting all inconsistent published locations exceeds two local lines; changing runtime default would be a behavior contract change. |
| CTD-18 | Corrected / contract enforced | Replace train/test diagnostic matrix CSV writes with shape-only debug messages; retain stored fitted transformers. CSV absence and existing-byte preservation, one/two sources, artifact on/off, and persisted transformer interpolation verified. |

### CTM

| ID | Outcome | Implementation / qualification |
|---|---|---|
| CTM-01 | Corrected / contract enforced | Stateful check-before-fit keys include deep constructor parameters, input content, fit_on_all and cohort selection; lookup and persistence use the same identity. |
| CTM-02 | Corrected / contract enforced | Forward active sample IDs through fold/single/average storage; keep slicing positional. Stored metadata follows IDs and augmented origins. |
| CTM-03 | Corrected / contract enforced | MetaModel REFIT trace fold identity now matches final prediction identity, preventing normal cleanup from deleting the genuine fitted final meta artifact. Exact selected artifacts and raw-input closure are checked before output mutation. No stacking-bundler/schema redesign. |
| CTM-04 | Corrected / contract enforced | All three aggregation paths use stored metric direction, inverse error weights, and finite-score handling; negative higher-is-better scores remain rankable. |
| CTM-05 | Corrected / contract enforced | Clone the wrapped estimator before applying optimized parameters; folds cannot share or mutate the user estimator. |
| CTM-06 | Corrected / contract enforced | Prediction/explanation creates placeholder folds from saved artifact count and avoids parsing/validating the original fold file. Public prediction succeeds after deletion of the file. |
| CTM-07 | Deferred by audit policy | Prediction still calls get_n_splits without X/groups. Passing new X alone would produce the wrong original LeaveOneOut fold count; saved fold-count and group-dependent replay metadata require a larger repair. |
| CTM-08 | Deferred by audit policy | Fold training still passes full launch_training arguments to default process-backed Parallel. The actual threaded branch fix explicitly qualifies this as a separate loky serialization issue, not repaired by D04 parallel example. Picklable payload or safe shared-runtime threading needs more than two lines/contract review. |
| CTM-09 | Deferred by audit policy | Fold adapter accepts candidate with >=2 folds without an exact validation-partition check. Correct auto fallback versus required-policy error needs partition validation and routing beyond two lines. |
| CTM-10 | Deferred by audit policy | Reconstructor returns complete_mask and zero-filled rows, while MetaModel returns unmasked X/y. Dropping requires matching scaled/unscaled targets, sample IDs and remapped fold coordinates; a one-line feature mask is incomplete. |
| CTM-11 | Deferred by audit policy | Selector still returns the first matching stored validation score. Average-row preference alone leaves missing-average cases; correct aggregation and fallback requires a defined CV ranking contract beyond two verified local lines. |
| CTM-12 | Deferred by audit policy | JAX wrapper persists classification role and JAX prediction argmax is now classification-only (OM1-14). TensorFlow still argmaxes every multi-column output, while task_type exists only in training scope. Complete task-role binding through TensorFlow prediction/artifact replay needs more than a local guard; no complete TensorFlow multi-target witness was found. Existing JAX test proves real binary/multiclass loss/probabilities, not the entire TensorFlow finding. |
| CTM-13 | Corrected / contract enforced | Verified one-line legacy correction: branch on metric direction before score sign; independent R2/accuracy/RMSE fold-weight references pass. |
| CTM-14 | Corrected / contract enforced | Verified one-line legacy correction: uniform weights for nonpositive total, including regression and class probabilities; 7 aggregation witnesses pass. |
| CTM-15 | Deferred by audit policy | CSV stores train IDs only and reconstructs validation complements within the train-union universe. Correct single/nonpartitioning splits need writer/loader schema or saved universe/validation metadata changes >2 lines. Explicit-val JSON is unaffected. |
| CTM-16 | Deferred by audit policy | Warm-start best and unknown selectors still return first artifact fold. Correct validation ranking requires score metadata lookup plus selector validation >2 lines; legacy fold_N versus DAG foldN remains a distinct syntax contract. |
| CTM-17 | Corrected / contract enforced | Verified local two-line exception: recognize exceeds/exceeded and raise on otherwise unrecognized invalid source validation. |

### API

| ID | Outcome | Implementation / qualification |
|---|---|---|
| API-01 | Corrected / contract enforced | Combine global and per-dataset refit evidence; final/final_score/export select the CV winner; eager models include all model names and pair finals with their own CV configuration. |
| API-02 | Deferred by audit policy | Each top-k selected config still reaches per-model best extraction for multi-model topology; extraction reselects best per class instead of respecting the selected config. Dispatch/config-selection repair exceeds two lines. |
| API-03 | Deferred by audit policy | Legacy Session still sends tuples/arrays/raw X-y dicts directly to DatasetConfigs. Existing orchestrator array wrapper could fix a subset, but documented dict/metadata materialization remains unresolved; complete normalization exceeds two lines. |
| API-04 | Deferred by audit policy | Legacy direct-run dict reaches DatasetConfigs; accepted key aliases are partitioned train_x/train_y etc., not public raw X/y/metadata mapping. Complete normalization includes X/y and metadata/cohort semantics beyond two lines. |
| API-05 | Deferred by audit policy | Legacy Session still drops refit/store lifecycle run kwargs and default plots_visible=False resets a configured runner. Supporting all runner.run arguments and distinguishing omitted versus explicit False entails argument/default contract work; a two-line refit-only patch would leave plotting behavior or public default contract unresolved. |
| API-06 | Deferred by audit policy | Legacy session reuse still applies only verbose when !=1 and does not apply other explicit runner settings. Correct distinction of omitted/explicit overrides and restoring shared runner state after success/error requires structured settings ownership beyond two lines. |
| API-07 | Deferred by audit policy | Session closes its runner, while attached legacy results retain that runner and export via its closed store. Correct detach/reopen ownership across all produced results requires lifecycle work beyond two lines. Current final API receipt explicitly retains API-07. |
| API-08 | Deferred by audit policy | Nested all-prediction dict still yields empty y_array. Bundle-only single-entry unwrapping is a subset; complete multi-dataset/fold result aggregation needs explicit cohort/axis semantics, not silently averaging unrelated arrays in two lines. |
| API-09 | Corrected / contract enforced | Template path retains physical wavelength axis in nm; explicit caller axis takes precedence. |
| API-10 | Duplicate | Duplicate; see canonical SYN-19 |
| API-11 | Corrected / contract enforced | JSON documents bypass YAML preflight while common byte/node/depth limits remain. |

### CFG

| ID | Outcome | Implementation / qualification |
|---|---|---|
| CFG-01 | Corrected / contract enforced | Default classification loss minimization and explicit metric direction; independent optimizer trials cover invalid/maximize paths. |
| CFG-02 | Corrected / contract enforced | Preserve literal nested sequences in eager/lazy/choices expansion; augmentation count alone is not a generator. |
| CFG-03 | Corrected / contract enforced | Classifier delegates to the shared DAG-aware result exporter without source selection. |
| CFG-04 | Corrected / contract enforced | Captured DAG preprocessing replays before probabilities and transform; base model access stays compatible with SHAP. |
| CFG-05 | Corrected / contract enforced | Nonfinite and failed maximizing objectives get the worst score. |
| CFG-06 | Corrected / contract enforced | Best-fold aggregation follows direction; robust reducers reject nonfinite scores. |
| CFG-07 | Corrected / contract enforced | Reentrant rotation lock verified by a bounded emitting thread. |
| CFG-08 | Corrected / contract enforced | Relative/path parameter strings round-trip without imports. |
| CFG-09 | Corrected / contract enforced | Expand nested primary choices before second selection; explicit second-order grammar documented. |
| CFG-10 | Corrected / contract enforced | Numeric tuple search ranges serialize with typed bounds; lists retain categorical meaning. |
| CFG-11 | Corrected / contract enforced | Reject zero generator configurations after both analytical-zero and constraint-pruned expansion. Empty nongenerator configuration, ordinary model, and nested branch bypass contracts retained. |
| CFG-12 | Corrected / contract enforced | Inline YAML mapping follows loader rules; scalar rejected; uppercase suffix recognized. |
| CFG-13 | Corrected / contract enforced | SQLite read-only URIs encode filesystem punctuation. |
| CFG-14 | Corrected / contract enforced | Preset siblings override dictionary defaults; unsupported scalar merges fail explicitly. |
| CFG-15 | Corrected / contract enforced | Floating ranges use matching integer-index cardinality, nested grid/weighted subset counts match expansion, empty zip emits zero; constraints intentionally retain safe upper-bound counts. |
| CFG-16 | Corrected / contract enforced | OR modifier keys come from the shared keyword definitions. |
| CFG-17 | Corrected / contract enforced | Lazy expansion normalizes sibling parameter ranges/grid/zip like eager expansion. |
| CFG-18 | Corrected / contract enforced | Range sampling respects local seed; tiny ranges retain magnitude. |
| CFG-19 | Corrected / contract enforced | Central higher-is-better table recognizes jaccard_score; f1_score already supported. |
| CFG-20 | Corrected / contract enforced | Validator accepts root list, steps and pipeline forms without reclassifying dataset documents. |
| CFG-21 | Corrected / contract enforced | Remove import-time process-wide warning suppression. |
| CFG-22 | Corrected / contract enforced | Clone round-trip for fold parameter, reject unknown parameters; owned TemporaryDirectory cleans on adapter disposal/failure. |
| CFG-23 | Corrected / contract enforced | Retention excludes the active log file. |
| CFG-24 | Corrected / contract enforced | Extra formatting clones LogRecord to avoid mutating shared messages. |
| CFG-25 | Corrected / contract enforced | Log/sample ranges retain tiny values with significant-digit precision. |

### STO

| ID | Outcome | Implementation / qualification |
|---|---|---|
| STO-01 | Corrected / contract enforced | Repair historical refcount undercounts against all chain references before pruning, run deletion and transient cleanup; preserve surviving sibling replay/export, release last references. |
| STO-02 | Corrected / contract enforced | Transaction-enclosed run and prediction cascades delete linked conformal/tuning/robustness results in FK order; failed cascades roll back metadata and refcounts before tombstones/GC. Physical GC waits outside open transactions. |
| STO-03 | Corrected / contract enforced | Portable resolver exports collect MapArtifactProvider steps without a trace, retain V3/canonical/UUID fold membership and final priority, and refuse empty artifact collections. Scope: Explicit legacy portable export; no claim of native portable script support. |
| STO-04 | Corrected / contract enforced | Strip canonical fold_ prefix, prefer final models, skip shared metadata keys and replay source-specific transforms using _source_map in portable scripts. Scope: Validated flat aligned multisource preprocessing/model chains, not general branching/stacking portability. |
| STO-05 | Corrected / contract enforced | Enumerate artifacts and append sub-index to archive member names; duplicate same-class shared transformers survive ZIP write and loader indexing. Scope: Exactly two local production-line changes under the legacy minor exception; does not establish general multisource ZIP prediction support. |
| STO-06 | Corrected / contract enforced | Stage workspace-chain, resolver, single-model and portable-script exports beside destination; replace atomically only after successful close; remove staging files on failure. |
| STO-07 | Corrected / contract enforced | Atomically publish complete artifact bytes on regular, chain and deferred registration; hash-verify in-memory dedup hits and repair truncated existing files. |
| STO-08 | Corrected / contract enforced | Rollback transactions on BaseException, including KeyboardInterrupt/SystemExit; verify another workspace connection can write afterward. |
| STO-09 | Corrected / contract enforced | Reject unsafe category/name components and resolved paths outside the library on save/import/load/delete/list; preserve library contents on dot-dot requests. |
| STO-10 | Corrected / contract enforced | Dispatch n4a.py through BundleGenerator; reject unsupported formats and relational manifest requests incompatible with scripts before writing. |
| STO-11 | Corrected / contract enforced | Both SQLite retry wrappers immediately propagate non-lock OperationalError instead of retrying permanent errors. |
| STO-12 | Corrected / contract enforced | Filter run datasets by exact JSON name equality; SQL wildcards/substrings/quoted names do not match unrelated datasets. |
| STO-13 | Qualified contract | Authoritative verdict is a low-severity diagnostic/fallback qualification: original format failure is retained in a warning. No format fallback redesign under this batch. Actual trained JAX controller wrapper was registered, persisted, loaded with a fresh registry, and verified for wrapper/model types, parameters, optimizer state, batch statistics, training steps and predictions. Scientific qualification: Measured full trained JAX artifact roundtrip:4training steps,77958cloudpickle bytes, maxabs prediction error0.0, weights/optimizer/batch statistics retained. Diagnostic fallback qualification, no serializer redesign. |
| STO-14 | Corrected / contract enforced | One-line precedence correction preserves explicit pipeline_id when pipeline_uid is empty. |
| STO-15 | Corrected / contract enforced | Forward and validate cv/all/final scope; aggregate raw prediction membership, score statistics, IDs and counts within scope before ranking/pagination. Exclude correlated aggregate fold rows. Final scope defaults to test-score ranking; historical cv_* and final_* summary fields remain available. |
| STO-16 | Corrected / contract enforced | Propagate artifact serialization failure with artifact identity and original cause; atomic export preserves prior valid bundle. Scope: One local production-line replacement under legacy minor exception. Regression injects a serialization failure; actual JAX serialization is not repaired or newly qualified. |

### DT1

| ID | Outcome | Implementation / qualification |
|---|---|---|
| DT1-01 | Corrected / contract enforced | Validate train/test source counts and feature widths before adding samples; ArrayStorage rejects mismatched 2D sample appends before mutation. Generated or inferred header strings need not match. Processing padding remains available. |
| DT1-02 | Corrected / contract enforced | Remove partition scores by each stored metric direction; trim independently within metric groups so incomparable units are not mixed. |
| DT1-03 | Corrected / contract enforced | Snapshot prior-source predictions; preserve source configuration/chain metadata in distinct merge contexts; avoid same-source self-conflicts and stale foreign keys. |
| DT1-04 | Corrected / contract enforced | Include dataset, configuration, pipeline, chain, branch and model identity when deduplicating train/test final twins. |
| DT1-05 | Corrected / contract enforced | Clear display arrays when the requested partition is absent instead of evaluating validation arrays as test metrics. |
| DT1-06 | Corrected / contract enforced | Skip zero-column/empty targets for optional unlabeled partitions; y selectors omit missing trailing target rows and preserve labeled sample order. Labeled train plus unlabeled test and cached reloads are covered. |
| DT1-07 | Corrected / contract enforced | Actually skip worse or equal source rows; compare by stored metric direction; replace inferior target rows and handle missing scores. |
| DT1-08 | Corrected / contract enforced | Empty metadata selections return zero rows for get/get_column/label/onehot, both cold and cached. Target half was repaired in initial receipt. |
| DT1-09 | Corrected / contract enforced | Binning digitizes only interior edges, giving exactly bins zero-based intervals and preventing a singleton minimum. |
| DT1-10 | Corrected / contract enforced | QUALIFIED finding repaired by rejecting incompatible requested refit metrics. Default refit ranking uses the stored selection metric and pre-test selection_score/val_score only. |
| DT1-11 | Corrected / contract enforced | Partition lookup matches pipeline/chain/preprocessing/branch/target-processing identity to prevent borrowing another variant arrays. |
| DT1-12 | Corrected / contract enforced | Metadata maps selected sample ids to origins, preserves order/duplicates via left joins, and repeats base metadata for augmented samples. All metadata views align with X/Y; empty and base-only selections covered. Integration expectations now require exact repeated origin values and encoded order, with augmentation explicitly excluded in separate assertions. Follow-up: Three obsolete base-only metadata expectations replaced with independent origin mapping assertions; two accidentally comment-concatenated test definitions restored. Follow-up: Metadata accessor already honors selection and origin mapping. Legacy metadata separation wrongly indexed its compact filtered result with physical sample IDs. Column existence now uses metadata schema, training metadata uses the exact X selector, and universe metadata selects actual universe sample IDs. IDs remain physical, excluded/augmented rows do not leak into grouping; branch minimum counts and context subsets are preserved. |
| DT1-13 | Corrected / contract enforced | Signal detection selects the individual source rather than concatenating every source. |
| DT1-14 | Corrected / contract enforced | Repetition reshapes copy per-source header metadata and the complete fitted target chain, subset all target processings, and retain raw class names/inverse transforms. Interleaved partition order is preserved. Partially unlabeled reshape is refused before mutation. |
| DT1-15 | Corrected / contract enforced | Indexer recognizes tag_filters in normalized mappings as well as typed selectors, preserving tag filtering across X/Y/metadata views. |
| DT1-16 | Corrected / contract enforced | Query validation uses live store columns, so new tag columns select and update only matching rows. |
| DT1-17 | Corrected / contract enforced | Raw public multi-source append rejects unequal source row counts before initializing feature storage or extending indexer. |
| DT1-18 | Corrected / contract enforced | Partial processing width changes are refused before mutation. Complete replacements retain all names and values. Concat controller collects every processing result from original inputs then replaces the complete source once, so valid full replacement no longer fails or corrupts subsequent inputs. |
| DT1-19 | Corrected / contract enforced | Transactional raw and batch appends stage independent source storage/header state and the entire index frame; publish only after every source succeeds. Existing spectral arrays are referenced read-only during staging, not deep-copied. Late second-source dtype/header-unit/dimensions/processing-count/row-count failures, failed first initialization, CoW snapshots, and corrected retries are tested. No late append inconsistency is deferred. |
| DT1-20 | Corrected / contract enforced | Public add_features accepts its documented bare ndarray input without an ambiguous truth-value error. |
| DT1-21 | Corrected / contract enforced | Indexer accepts validated per-row partition lists; mixed batch appends and augment_rows inherit each origin partition. |
| DT1-22 | Corrected / contract enforced | DataCache refuses nonpositive max_entries explicitly instead of hanging; clear removes each entry through its eviction callback, releasing resources once. |
| DT1-23 | Duplicate | Duplicate; see canonical CFG-19 |
| DT1-24 | Corrected / contract enforced | group_by_fold selects top N per fold and adds fold_id to explicit grouping, with clarified local docstring. |
| DT1-25 | Corrected / contract enforced | Validate finite fractions in [0,1], permit floor-rounded zero removals, and report actual remaining rows for dry runs. |

### DT2

| ID | Outcome | Implementation / qualification |
|---|---|---|
| DT2-01 | Corrected / contract enforced | Determine absent partitions from config keys, never from error-message substrings. Configured missing/NA/malformed train_x and test_x errors propagate. Preserve relation-config guard. |
| DT2-02 | Corrected / contract enforced | Validate original row counts, then intersect retained original positions across X/Y/metadata and all feature sources. Normalize stored DataFrame indices to positional identity; Parquet index labels never become accidental joins. Metadata remove_sample participates when requested; default missing metadata remains permitted. |
| DT2-03 | Corrected / contract enforced | All parser syntaxes and dict/JSON/YAML preserve explicit root settings, including False, grouping, signal, folds, and provenance. |
| DT2-04 | Corrected / contract enforced | CSV does not forward framework signal_type to pandas. Both global_params and train_params target/metadata loads are tested. |
| DT2-05 | Corrected / contract enforced | Factorization preserves categorical missing values for NA handling; blank rows retain positional identity. abort/remove_sample/ignore and object-array encoder missing values are covered without inventing classes. |
| DT2-06 | Corrected / contract enforced | Every feature path retains a parameter placeholder; source and per-file overrides merge without shifting positions. |
| DT2-07 | Corrected / contract enforced | Source and variation inference matches basename tokens, never parent directories. |
| DT2-08 | Corrected / contract enforced | Files inference uses whole tokens and rejects ambiguous names; explicit partitions override inference. |
| DT2-09 | Corrected / contract enforced | Common apply_na_policy coerces dictionary fill configs across all loaders; Parquet regression included. |
| DT2-10 | Qualified contract | Qualified contract: pandas NA defaults retained; typed keep_default_na/na_values/na_filter controls document and test literal category round-trips. |
| DT2-11 | Duplicate | Duplicate; see canonical DT1-06 |
| DT2-12 | Corrected / contract enforced | Explicitly refuses multiple separate/compare variations instead of silently selecting the first; single/concat/select supported. |
| DT2-13 | Corrected / contract enforced | Explicitly refuses all configured inline/file/folder folds at the load boundary; use dataset.set_folds or pipeline splitters. No silent CV replacement. |
| DT2-14 | Corrected / contract enforced | Refuses legacy files/source/variation/shared-reference columns/rows/link_by and duplicate shared partition references. Full relation staging and materialize_relation_table remain separate and intact. |
| DT2-15 | Corrected / contract enforced | Unit-aware CSV normalization converts numeric decimal-comma spectral headers to decimal points while preserving text/metadata headers. |
| DT2-16 | Corrected / contract enforced | Resolve per-dataset overrides by config identity plus name; duplicate dataset names do not reuse the first task setting. |
| DT2-17 | Corrected / contract enforced | Parser errors retain specifics, bare files load, and constructor refuses unsupported entries instead of skipping them. |
| DT2-18 | Corrected / contract enforced | Folder scanner rejects simultaneous val/test families with an explicit selection instruction. |
| DT2-19 | Corrected / contract enforced | Archive dispatch sends encoding only to CSV text loaders; real zipped Parquet round-trip tested. |
| DT2-20 | Corrected / contract enforced | Array/DataFrame/mixed multi-source inputs share positional loader alignment, honor scoped units, validate original row counts, and apply requested joint NA removal. Missing partition metadata reserves null rows; test-generated coordinates cannot overwrite training metadata. |
| DT2-21 | Duplicate | Duplicate; see canonical OPO-16 |
| DT2-22 | Corrected / contract enforced | Restore original categorical labels from loader reports; existing Targets converter fits training vocabulary once and reuses it for test. Original raw class names and inverse transformation remain available; mixed numerical target columns retain values. |

### OPO

| ID | Outcome | Implementation / qualification |
|---|---|---|
| OPO-01 | Corrected / contract enforced | Both production EMSC and copied AOM ExtendedMSC now use raw mean reference, normalized wavelength-index polynomial basis including constant, and reusable multiplicative/additive correction. Identifiability failures are explicit. |
| OPO-02 | Corrected / contract enforced | Qualified axis finding repaired by an explicit spectral axis=1 default in Derivate/derivate, used by existing presets; sample-axis differentiation remains explicitly selectable with axis=0. No input mutation. |
| OPO-03 | Corrected / contract enforced | EPO stores an orthogonal feature-space complement of fitted external-parameter loadings; fit_transform and transform apply the identical operator. Docs acknowledge removal of signal sharing interference directions. |
| OPO-04 | Corrected / contract enforced | Python MixupAugmenter/LocalMixupAugmenter and native Mixup/LocalMixup through legacy/DAG sample augmentation are explicitly unsupported until joint X/y weights and multiple-parent sample identity are implemented. Entire selection validates before sample materialization; nested sklearn pipelines and wrapped native roles are covered. Direct X-only operator APIs remain supported. |
| OPO-05 | Corrected / contract enforced | Sort wavelength coordinates and spectra together before interpolation/convolution and restore original order; reversal/permutation and analytic linear interpolation references. |
| OPO-06 | Corrected / contract enforced | Select exactly n_train unique nearest unused centroid representatives; test is the exact complement, including degenerate duplicate inputs. |
| OPO-07 | Corrected / contract enforced | Use a local seed for initial twin; validate documented reciprocal integer ratios rather than silently quantize arbitrary fractions. |
| OPO-08 | Corrected / contract enforced | CARS normalizes coefficients by their exact sum, with uniform fallback for zero total importance. Selection tested invariant to target unit scales down to 1e-9. |
| OPO-09 | Corrected / contract enforced | Build edge child operators from current flags/strength/model/seed at transform; set_params and cloned seeded diversity validated. |
| OPO-10 | Corrected / contract enforced | Reduce X only with PCA and retain original scalar/multioutput target distances. |
| OPO-11 | Corrected / contract enforced | Correct SNV/robust SNV docs to describe stateless within-batch axis=0 statistics; preserve legitimate batch-dependent behavior and defaults. |
| OPO-12 | Corrected / contract enforced | Functional norml respects custom-bound condition and subtracts matrix-wide minimum; preserve original global-bound functional and column-norm default contracts. |
| OPO-13 | Corrected / contract enforced | Normalize.user_defined is derived from current feature_range via property, so set_params followed by refit uses current bounds. |
| OPO-14 | Corrected / contract enforced | Guard zero calibration spans/norms in Normalize, SimpleScale, norml and spl_norml; inverse uses the same guarded scale, matching sklearn on constants. |
| OPO-15 | Corrected / contract enforced | Qualified API repair: MSC cannot infer discarded per-spectrum offset/slope. inverse_transform now raises explicit NotImplementedError instead of accessing removed attributes; no invented inverse or cached-last-batch semantics. |
| OPO-16 | Corrected / contract enforced | ToAbsorbance opt-out refuses non-positive input and leaves positive values unclipped; default epsilon clipping is preserved. |
| OPO-17 | Corrected / contract enforced | Qualified documentation repair; preserve chi-square root-Mahalanobis and empirical PCA 95th-percentile defaults. |
| OPO-18 | Corrected / contract enforced | Explicit pp_names lists now fail clearly at configuration construction; supported string templates retained. Full list support needs dataset caller indices outside this scope. |
| OPO-19 | Corrected / contract enforced | Self reference preserves a+bX; global mean adds scatter perturbation X+a+(b-1)*fitted_reference, documented and tested analytically. |
| OPO-20 | Corrected / contract enforced | Qualified sparse-bin input limitation documented; preserve legitimate defaults and sklearn stratification refusal. |
| OPO-21 | Corrected / contract enforced | Use local random.Random preserving seeded rotation behavior without changing global RNG. |

### OM1

| ID | Outcome | Implementation / qualification |
|---|---|---|
| OM1-01 | Corrected / contract enforced | 17 sklearn estimators place mixins before BaseEstimator, verified on original and cloned estimator tags. |
| OM1-02 | Corrected / contract enforced | Weighted location and variance are recomputed inside each IRLS fit and final NumPy/JAX fit; clean-test intercept/slope regression witnesses pass. Qualified audit scope preserved: weights previously improved slopes but left intercept bias. |
| OM1-03 | Corrected / contract enforced | KOPLS always centers targets; KernelPLS retains target offset whenever kernel centering is enabled while preserving explicitly uncentered scale_y=False behavior. |
| OM1-04 | Corrected / contract enforced | SparsePLS scale toggle is passed into both kernels; centering remains enabled. |
| OM1-05 | Corrected / contract enforced | SparsePLS selection uses fitted coefficients on both backends, eliminating the undefined wrapper. |
| OM1-06 | Corrected / contract enforced | MBPLS and SparsePLS project with deflation rotations; multiblock MBPLS preserves normalized super-score weights. |
| OM1-07 | Corrected / contract enforced | Short final JAX interval uses gather with width mask; no left-clamped dynamic slice. |
| OM1-08 | Corrected / contract enforced | FractionalConvFeaturizer stores constructor sequence objects unchanged and retains list-valued public filter info. |
| OM1-09 | Corrected / contract enforced | JAX fractional convolution flips kernels and honors same/valid modes in all model paths; feature-bank and final-model parity tested. |
| OM1-10 | Corrected / contract enforced | TF separable classifier uses binary sigmoid only for two classes. |
| OM1-11 | Corrected / contract enforced | Four TF factory callables declare tensorflow routing metadata. |
| OM1-12 | Corrected / contract enforced | Inception input length uses feature dimension; integral convolution filters permit Keras3 construction. |
| OM1-13 | Corrected / contract enforced | SE reduction uses positive integer Dense units on Keras3. |
| OM1-14 | Corrected / contract enforced | JAX classifiers return logits; binary train/validation use BCE; probabilities activate once and labels follow zero-logit threshold; task survives wrapper state. |
| OM1-15 | Corrected / contract enforced | Diversity selector uses stored metric direction. |
| OM1-16 | Corrected / contract enforced | Torch spectral transformer honors cls/mean pooling and validates modes. |
| OM1-17 | Corrected / contract enforced | Zero validation scores remain valid ranking values. |
| OM1-18 | Corrected / contract enforced | Validate scoring and integer CV counts up front; propagate unexpected CV errors and reject all-nonfinite interval scores. |
| OM1-19 | Corrected / contract enforced | Complete across touched branches: OPLS scale, MBPLS standardize, IntervalPLS scaled fitting. Non-r2 JAX IntervalPLS scoring and unsupported MBPLS method/max_tol values raise explicit errors. Default MBPLS closed-form updates do not claim iterative tolerance support. |
| OM1-20 | Corrected / contract enforced | Existing KOPLS tests now expect effective target rank, matching new regression. |
| OM1-21 | Corrected / contract enforced | Qualified intentional limitation: initial means remain fixed; corrected false EMA documentation and removed misleading dead EMA loop. No offset-drift algorithm introduced. |
| OM1-22 | Corrected / contract enforced | Both optimization backends honor random_state for non-warm initialization and re-solve regression/dynamics after final projection update. |
| OM1-23 | Corrected / contract enforced | MetaModel.set_params updates constructor name and rejects nonconstructor attributes. |
| OM1-24 | Corrected / contract enforced | JAX/Torch activation None is identity; unknown names fail explicitly. |
| OM1-25 | Corrected / contract enforced | All 16 audited custom estimator setters now use sklearn key/nesting validation across batches; DiPLS/LWPLS completed here. Listed predict-before-fit gaps now raise NotFittedError; OKLMPLS nested featurizer keys are included and clone-tested. PCR/TabPFNNIRS already inherit base validation and were not changed. |
| OM1-26 | Corrected / contract enforced | Qualified architecture constraints enforced before layer construction: UNET requires length>=25 divisible by25; Custom_VG_Residuals checks all configured valid convolution sizes/strides. Supported defaults and customized kernels remain usable. UNet_NIRS documented truthfully as legacy VGG11; historical public name retained, no architecture replacement. |

### AOM

| ID | Outcome | Implementation / qualification |
|---|---|---|
| AOM-01 | Corrected / contract enforced | Use conditional leave-one-out ridge loss on fitted latent scores, including intercept leverage; analytic hat correction matches explicit leave-one-out score-space refits. This is not full PLS refitting per held-out row. |
| AOM-02 | Duplicate | Duplicate; see canonical OPO-01 |
| AOM-03 | Corrected / contract enforced | Include training kernel diagonal in nearest-neighbor distance ranking; analytic positive/negative query distances and local Ridge predictions match Euclidean/sklearn references. |
| AOM-04 | Corrected / contract enforced | Center branch outputs after raw-spectrum preprocessing and replay fitted post-transform means in fold and held-out kernels. Identity branch matches sklearn Ridge, preserves center=False, and is invariant to offsets when center=True. |
| AOM-05 | Corrected / contract enforced | Use rotations Z @ pinv(P.T @ Z) for latent transform so training scores, transform and logistic calibration agree across five engines and nonidentity operators. |
| AOM-06 | Corrected / contract enforced | Use all residual target columns via the dominant response covariance combination shared across mixture operators; target permutations preserve coefficients and zero targets stop extraction. |
| AOM-07 | Corrected / contract enforced | Compute per-prefix rank-aware latent-score leverage with intercept instead of a full-X SVD including null directions. Conditional approximation documented; explicit fixed-score OLS LOO reference matches. |
| AOM-08 | Corrected / contract enforced | Guard auto alpha handling with isinstance(str); documented ndarray grid now fits and primal/dual predictions agree. |
| AOM-09 | Corrected / contract enforced | Compute standard error from fold scores at the best alpha, choose largest numeric eligible alpha, and apply the rule for uniform/manual/kta/softmax_cv strategies. Preserve per-fold evidence and mean scores. |
| AOM-10 | Corrected / contract enforced | Implement pooled trimmed residual scoring in active, MKL and branch-global paths, retaining ordinary per-fold RMSE evidence for selection rules. All five modes match independent pooled sklearn fold predictions. |
| AOM-11 | Corrected / contract enforced | Select automatic prefixes in soft/superblock/active_superblock with fold-local mode extraction and reusable prefix coefficients; preserve score curves in diagnostics. CV references and zero-component termination validated. |
| AOM-12 | Corrected / contract enforced | Accept numbers.Integral component counts and apply fitted feature standard deviations consistently to training, transform, probability prediction, and original-unit coefficients. |
| AOM-13 | Corrected / contract enforced | Clear stale constant-y prediction before a subsequent fit; compare constant-to-varying refit with a fresh model. |
| AOM-14 | Corrected / contract enforced | Include zero_trace_policy and zero_trace_threshold in clone parameter dictionary; validate preserved drop behavior. |
| AOM-15 | Corrected / contract enforced | Skip adaptive repetition for explicit grids and stop before an unused final expansion; diagnostics count only evaluated additional grids. All five modes validated. |
| AOM-16 | Corrected / contract enforced | Retain fitted inner estimator in serialized wrapper state; pickle prediction roundtrip validated with local Ridge substitute. Actual TabPFN weights/GPU/network portability remains unverified; serialization failures stay explicit. |
| AOM-17 | Corrected / contract enforced | Preserve original labels in balanced accuracy; compare labels directly rather than integer conversion/truncation. Same local defect in macro_f1 repaired. |

### SYN

| ID | Outcome | Implementation / qualification |
|---|---|---|
| SYN-01 | Corrected / contract enforced | Supply generated X before spectral targets; nonrecursive concentration fallback. |
| SYN-02 | Duplicate | Duplicate; see canonical API-09 |
| SYN-03 | Corrected / contract enforced | Attach requested sample metadata in split order and implement classification stratification; include per-sample generation metadata when requested. Raw build_arrays remains the documented two-array contract. |
| SYN-04 | Corrected / contract enforced | Keep separate dataset sources and wavelength headers. |
| SYN-05 | Corrected / contract enforced | Fit stochastic operators once so streams advance. |
| SYN-06 | Corrected / contract enforced | Independent spawned child seeds in generator/builder; independent source/product children. |
| SYN-07 | Corrected / contract enforced | Apply exact per-sample temperatures, including configured mean, tracked in metadata. |
| SYN-08 | Corrected / contract enforced | Construct requested simulator and apply both generation routes. |
| SYN-09 | Corrected / contract enforced | Standardize source before correlated Gaussian mixing; before bounds/closure correlation contract measured. Scientific qualification: Product rho describes standardized Gaussian mixing before bounds/closure; clipping/closure can change realized Pearson correlation. Measured positive/negative rho and low clipping, no universal exact-rho assertion. |
| SYN-10 | Corrected / contract enforced | Signed logistic-normal latent correlations; PSD validation and explicit simplex closure limitation. Scientific qualification: Signed PSD latent Gaussian log-abundance correlation with logistic-normal closure. Do not assert arbitrary final simplex Pearson matrices. |
| SYN-11 | Corrected / contract enforced | Use a genuine principal concentration direction for compositional regime quantiles. |
| SYN-12 | Corrected / contract enforced | Clamp threshold-noise scale at zero for separation >=3. |
| SYN-13 | Corrected / contract enforced | All69 aggregate compositions and variability references resolve to existing spectra. Generic/loratadine-like templates explicitly label paracetamol proxies; generic metformin/amoxicillin parent spectra explicitly disclaim salt/hydrate modeling. No invented compound-specific spectra. |
| SYN-14 | Corrected / contract enforced | Short public prior labels resolve canonical registry domains. Fallback starch is valid; biomedical domain uses existing oxy/deoxy hemoglobin components. Blood/lubricant labels use broader tissue/fuels priors, documented. |
| SYN-15 | Corrected / contract enforced | Replay fitted physics, mode, environment, scattering, edge configuration and detected components through generator and builder. Domain inference remains descriptive metadata. Scientific qualification: Fitted replay propagates supported physics/configuration and detected components; inferred domain remains descriptive metadata, not a generation parameter. |
| SYN-16 | Corrected / contract enforced | Persist preprocessing_type and is_preprocessed through save/load. |
| SYN-17 | Corrected / contract enforced | Preserve selected averaging method with and without rejected scans. |
| SYN-18 | Corrected / contract enforced | NumPy seed forwarded and correct per-spectrum RMS noise scale. |
| SYN-19 | Corrected / contract enforced | Unsupported generate() options fail before builder construction; distribution/batch_effects and engine/plugin/allow_fallback retain support. Document actual 350–2500 nm, 2 nm default; 74 API tests pass including 5 new witnesses. |
| SYN-20 | Corrected / contract enforced | Round-trip nonlinear, confounder, multi-regime, group-name, component-index and batch amplitude fields; consume batch amplitudes during generation. |
| SYN-21 | Corrected / contract enforced | Ascending zone ranges and narrow visible subzone precedence. |
| SYN-22 | Corrected / contract enforced | Exact registry channel counts via explicit linspace axis; registry measurement mode applied; executable doc example. |
| SYN-23 | Corrected / contract enforced | Exact decimal headers across all named authoring/export routes; integer-count bounded grids. |
| SYN-24 | Corrected / contract enforced | Limit spike count and dead-band ranges to available channels. |
| SYN-25 | Qualified contract | Range scales base targets before separately configured noise. Docstrings clarify base bounds and mixing-control correlation; clipping would silently change specified additive-noise distribution. Regression verifies base-range versus noisy-range distinction. |
| SYN-26 | Corrected / contract enforced | Each multi-sensor channel uses a wavelength-covering detector with maximal response; genuine uncovered channels retain primary detector fallback. No universal physical/noise ratio assertion. |
| SYN-27 | Corrected / contract enforced | Every failed metric retained as failed gate with error detail; transfer exceptions surfaced and fail overall gate. |
| SYN-28 | Corrected / contract enforced | Merge fluent physics options; stratification (SYN03); CSV headers/compression; meaningful SNR region; honor explicit procedural fields. |
| SYN-29 | Corrected / contract enforced | Local optional seed contract added to both stochastic fit paths; measured global NumPy state unchanged and repeatable local seeded fits/restarts. Scientific qualification: Original audit showed global RNG consumption, not nondeterminism of every fit. New optional local seed and global RNG isolation are measured; no universal identical-fit claim without explicit seed. |

### VIZ

| ID | Outcome | Implementation / qualification |
|---|---|---|
| VIZ-01 | Corrected / contract enforced | Fixed by concurrent Predictions.top order repair, consolidated at shared source. Verified both TopKComparisonChart and ConfusionMatrixChart, cached and direct, with opposing validation/test rankings. No redundant chart sort introduced. |
| VIZ-02 | Corrected / contract enforced | Aggregated heatmap groups by both non-partition cell axes and selects one best entry per cell; fills complete grids and both partition orientations. |
| VIZ-03 | Corrected / contract enforced | Three charts use Polars JSON path lookup rather than incomplete numeric regex; negative and scientific-notation secondary metrics retained. Non-finite JSON values normalized to null before lookup so NaN/Infinity siblings do not suppress finite metrics. |
| VIZ-04 | Corrected / contract enforced | Infer direction from stored evaluation metric, expose descending/partition/evaluation-metric selection, reject mixed metric populations, and rank nulls last. |
| VIZ-05 | Corrected / contract enforced | Both transfer metric implementations project source/target into one pooled PCA frame for coordinate comparisons; independent subspace loadings/EVR retained. Centroid offset and reduction tests verify real units. No universal scientific equivalence claim for position-paired Procrustes on unrelated samples. Scientific qualification: Shared PCA coordinate frame repaired; no scientific equivalence claim for position-paired Procrustes on unrelated samples. |
| VIZ-06 | Corrected / contract enforced | One-line parenthesis fix makes relative spread improvement reachable when raw baseline is positive. |
| VIZ-07 | Corrected / contract enforced | Augmented stacked names resolve into cloned sklearn pipelines, preserving component chains; stage4 retains stored transforms. Verified usable non-None transform output after both stages. |
| VIZ-08 | Corrected / contract enforced | Two local logical-line corrections per duplicate implementation: kneighbors already excludes self, and trustworthiness normalization has no extra factor two. Verified against sklearn on three distinct-distance noise fixtures; no tied-distance equivalence claim. Scientific qualification: Sklearn agreement measured on distinct-distance fixtures; no tied-distance equivalence claim. |
| VIZ-09 | Corrected / contract enforced | Inspect the shared sharded artifacts tree once, protect SQLite artifact rows and all dataset manifests during registry cleanup/purge and workspace GC. Reject dataset-scoped orphan cleanup because orphan ownership cannot be inferred. Dataset purge only considers known-owned unpublished registry records; CLI does not infer ownership or delete live/shared files. |
| VIZ-10 | Corrected / contract enforced | Required sklearn integration success controls overall result. Optional missing dependencies produce SKIP, not PASS, and cannot mask required failure. Six combinations cover success/failure/unavailability. |
| VIZ-11 | Corrected / contract enforced | Preserve validator warning code, message and field in text and JSON diagnostics. Valid-with-warnings remains the validator contract; missing files are not relabeled as encoding failures. |
| VIZ-12 | Corrected / contract enforced | Apply all three minimum score thresholds, including zero thresholds and combined filters. Stats computes count/mean/min/max for the selected score column, separately by evaluation metric. |
| VIZ-13 | Corrected / contract enforced | Both heatmap paths apply display_agg best/worst/mean/median to the full cell population independently of rank_agg; analyzer mean default agrees with chart and documentation. Same-partition secondary display metrics no longer alias the rank metric. |
| VIZ-14 | Corrected / contract enforced | Both heatmap paths normalize the comparison value with the internally lowercased model column; verified original mixed-case PLS8 filter in fast and aggregated paths. |
| VIZ-15 | Corrected / contract enforced | Fast chart dataframes use a shared public-scope filter: refit/final selects final rows, folds/cv selects CV observations excluding synthetic averages and refit-context rows, all preserves non-companion entries. Persisted _agg twins excluded. Verified scope populations for all three charts and empty refit data. Empty dataframes preserve chart messages. Refit rows lacking requested validation scores return explicit scope/rank/display missing-data message rather than mixing in CV rows or raising; both example plotting smoke flags pass. |
| VIZ-16 | Corrected / contract enforced | Confusion matrix uses explicit union of true/predicted class labels for construction and ticks. Verified noncontiguous integer and string classes. |
| VIZ-17 | Corrected / contract enforced | Branch summary/count, score collection, comparison and ranking select only the requested raw partition; merged partition dictionaries remain one unit. Exclude avg/w_avg and aggregate fold rows; prefer CV folds per branch, or refit rows if no CV folds exist. Refits no longer augment CV sample counts; insufficient refit-only samples reject statistical tests. |
| VIZ-18 | Corrected / contract enforced | Pool unbiased sample variances with ddof=1 for Cohen d; regression uses unequal group sizes and explicit sums of squared deviations. |
| VIZ-19 | Corrected / contract enforced | One-line branch_summary correction supplies sample standard deviation (ddof=1) to t confidence intervals; a singleton has undefined sample std (NaN). Plot methods left to rendering owner. |
| VIZ-20 | Corrected / contract enforced | One-line compatibility correction removes tick_labels= from boxplot; existing set_xticklabels controls labels. Test emulates older supported signature without downgrading dependencies. |
| VIZ-21 | Rejected by re-audit | Documented preset precedence is intentional; preset=None permits manual options. No visualization changes made. |
| VIZ-22 | Corrected / contract enforced | Require an existing store.sqlite before workspace inspection; open store-backed inspections read-only; list-library does not initialize a missing library. Three existing mocked workspace tests now initialize valid stores. |
| VIZ-23 | Corrected / contract enforced | Analyzer forwards metric/partition/show_metrics independently from shapes. Prediction-summary diagram displays requested metric and partition score with explicit label. |
| VIZ-24 | Corrected / contract enforced | One-line conditional chooses reversed colormap only for distance metrics. Verified RGB direction independently of plotting alpha. |
| VIZ-25 | Corrected / contract enforced | Two local logical-line corrections replace global RNG with deterministic local sampling and replace position-paired minimum distance with symmetric mean of directed nearest distances. Primary spread finding repaired; unrelated SHAP background RNG reference is not part of this verified spread contract. Scientific qualification: Directed nearest-distance spread repaired; unrelated SHAP background RNG is outside verified spread contract. |
| VIZ-26 | Corrected / contract enforced | Unified metadata/tag/filter separation dictionaries preserve group steps instead of treating reserved keys as duplication branches; feature merge dictionary correctly labelled. |
| VIZ-27 | Corrected / contract enforced | Node dimensions computed by drawing nodes before edges; edge endpoints use final padded bounds. Numeric endpoint regressions for multiline boxes, no pixel QA claimed. |
| VIZ-28 | Corrected / contract enforced | One-line EVR guard checks raw baseline first; positive raw variance and zero retained variance receives zero EVR credit. |

### EXE

| ID | Outcome | Implementation / qualification |
|---|---|---|
| EXE-01 | Corrected / contract enforced | Feature stages retain raw input, fit on fold training IDs only, and persist their fold replay state with each model. CV OOF and held-out test arrays match independent sklearn SelectKBest/MC-UVE pipelines; full-train refit is separate. Workspace, trace/index bundles, store replay, no-test CV and OOF meta-features pass. Root implemented Optuna/n4m evaluation hooks; eight independent tuning witnesses rerun passed. Extended contract now covers source merges, exact augmentation child IDs, all four layouts and actual TensorFlow archive replay. Unsupported feature/augmentation routing has explicit tested refusals. Target replay follow-up repairs all eleven reload/reuse/stacking/SHAP failures from the 16-failure other-integration log. Follow-up: WorkspaceStore.replay_chain loads model wrappers first and skips the feature stages they own, while preserving y-processing inverse transforms. Tuple step replay uses its controller schema instead of sequential transformer calls. Scope: Explicit legacy/controller feature-preprocessing repair. Does not establish equivalent defects in native/DAG. Unsup-only legacy global preprocessing remains unchanged; arbitrary neural architecture training quality is not claimed. Unsupported compositions refuse instead of producing scientifically misleading CV scores. Explicit refusals: Python MixupAugmenter/LocalMixupAugmenter and native Mixup/LocalMixup through legacy/DAG sample augmentation are explicitly unsupported until joint X/y weights and multiple-parent sample identity are implemented. Entire selection validates before sample materialization; nested sklearn pipelines and wrapped native roles are covered. Direct X-only operator APIs remain supported.; Sample augmentation introduced after the retained learned-feature snapshot when supervised CV preprocessing is required: refuses with instruction to place augmentation before feature stages.; Supervised feature selection routed by by_source branches: refuses with instruction to use merge_sources before selection or separate per-source pipelines.; Branch feature merges after supervised branch preprocessing: refuses with instruction to merge before supervised selection or train inside ordinary branches. OOF prediction merges/MetaModel are not refused. |
| EXE-02 | Deferred by audit policy | Multi-dataset exceptions still continue regardless of continue_on_error and the managed run is completed afterward. Complete false/error and true/partial lifecycle reporting requires multi-path control/status changes >2 lines; public single-dataset runs are separate. |
| EXE-03 | Deferred by audit policy | Transfer/finetune path begins a store run and returns without completion/failure. Exceptions also occur before the narrow executor try block, so adding one success and one inner failure call is incomplete; structured whole-run lifecycle exceeds two lines. |
| EXE-04 | Corrected / contract enforced | Diagnostic KFold ratio matches balanced train/validation counts. |
| EXE-05 | Corrected / contract enforced | Shape inference handles PCA forms, crop slices and canonical resample controls. |
| EXE-06 | Corrected / contract enforced | Unwrap canonical splitter configurations and preserve grouping. |
| EXE-07 | Corrected / contract enforced | Verified two-line legacy exception: deserialize bare canonical class strings for topology model/splitter detection. |

### QA

| ID | Outcome | Implementation / qualification |
|---|---|---|
| QA-01 | Corrected / contract enforced | Public signature/export contract explicitly advances for already-local supported API additions while preserving old exports. |

## Reproducible gates

Use Python 3.11+ and the exact public cohort above. The broad CPU profile uses the official PyTorch CPU distribution; a CUDA-only test requires a separate actual GPU profile. Set the DAG CLI and installed-child interpreter variables through the repository's installed-runtime preparation helpers. Keep mandatory cold-replay flags enabled; source imports do not replace an installed-wheel replay.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 CUDA_VISIBLE_DEVICES= \
  python -m pytest tests/unit tests/regression -n 6 --dist worksteal --timeout=300
python -m pytest tests/integration/api -n 2 --dist worksteal --timeout=300
python -m pytest tests/integration/parity -n 2 --dist worksteal --timeout=300
python -m ruff check nirs4all tests scripts
python -m mypy nirs4all
python -m sphinx -b html docs/source /tmp/nirs4all-release-docs --keep-going
python -m build --no-isolation --sdist --wheel --outdir /tmp/nirs4all-release-artifacts
python -m twine check /tmp/nirs4all-release-artifacts/nirs4all-1.4.2*
```

The release workflow additionally runs every integration test and all installed examples, including the prepared external-runtime lanes. Optional missing runtimes and capability skips remain explicit. New audit witnesses live beside their modules under `tests/unit/**/test_audit_*.py`, with API, controller, workflow and native integration witnesses. They verify independent calculations, actual fitted replay, atomic failure behavior and public contracts.
