# Runtime identity foundation: isolated candidate only

This phase supplies an in-memory codec and original-source envelopes. It is
not connected to `run`, Session, PipelineRunner, dispatch, storage or export.
It neither expands recipes nor claims execution support, numerical correctness,
runtime variant binding, safety attestation, replay or ecosystem migration.

The candidate follows `closed-plan-root-decisions.md` (SHA-256
`d0a1fa8147533ba0d63ea7c68be019b9e1a5fbfb0ad347b9906b8d15e5659e00`).
Its restrictive interim profile must not be rolled out as a replacement for
previously supported representations or requests.

## Ordered recipe bytes

`encode_recipe` snapshots exact built-in JSON values. It refuses non-string
mapping keys, tuples, subclasses, unsupported objects, cycles, invalid UTF-8
strings and nonfinite floats. It never imports dotted references, evaluates
callables, calls constructors or stringifies unknown objects.

`RecipeEncoding.payload` is compact UTF-8 JSON, without key sorting or Unicode
normalization. `canonical` is the ASCII profile name
`nirs4all.realized-recipe.ordered-json.v1`, a NUL byte, then compact JSON of a
typed ordered tree. Tags distinguish null, boolean, decimal integer, hexadecimal
binary64 float, string, list and ordered mapping entries. Integer `1`, float
`1.0`, boolean `true`, and positive/negative floating zero remain distinct.
The digest is `sha256:` followed by lowercase SHA-256 of those canonical bytes.

`verify_recipe` reconstructs that tree from stored payload bytes and checks the
profile, exact compact payload and digest. Duplicate JSON keys, nonfinite
tokens/overflow and alternate payload formatting refuse. A future qualified
writer must insert these exact validated bytes, not reserialize a mutable value
through `_to_json` or `default=str`. Hashes detect inconsistency, not malicious
producers or scientific validity. Existing `get_hash`, names and old lineage
are not changed or rewritten.

Limits are depth 128 and 8 MiB for both payload and domain-prefixed canonical
bytes. Smaller caller limits are allowed; larger limits refuse. Unsupported
integer conversions fail with a typed JSON error rather than stringification.
Snapshot traversal charges the expanded canonical byte cost before retaining
each node. Completed shared subtrees reuse their validated copy and measured
cost, charging that cost at every occurrence and checking depth at that location.
Shared acyclic inputs remain valid when their full representation fits; cycles
and oversized shared graphs refuse without truncation. The byte format, digest
domain and caps are unchanged. Literal golden bytes and a literal SHA-256 pin
the profile, integer/float encodings and type tags.
Constants also declare the accepted later-stage 10,000-occurrence and 64 MiB
prepared-request limits; this phase does not implement a prepared-request
catalog or enforce aggregate limits for a caller-owned collection.

## Original source and pre-serialization checks

`capture_source` preserves parsed/Python list or dictionary roots, all keys and
their order, and independent containers. It does not collapse `steps` and
`pipeline`, choose a root alias, strip metadata or parse executable references.
Its versioned envelope contains representation kind, normalized document,
original text when available, and declared conversion provenance.

`capture_json_source` accepts supplied strict UTF-8 JSON bytes and retains their
exact whitespace and spelling. It performs no filesystem IO. Original Python
source bytes are explicitly unavailable. YAML parsing, live-object inference
and prebuilt PipelineConfigs recovery are not implemented here; an owner must
supply a verified representation in a later authorized phase.

Checks occur before the serializer can erase tuple types or coerce keys. The
only declared tuple conversion is the existing `_preserve_finetune_ranges`
behavior: a numeric two-tuple in a finetune mapping becomes
`{"type":"int"|"float","min":...,"max":...}`. Boolean endpoints refuse.
Original endpoint types, values and path are retained, and `document()` restores
the original tuple. Categorical lists keep their meaning; tuple-valued choices,
ordinary parameter tuples and documented three-tuples such as `('int',1,30)`
refuse this interim profile. Boolean-keyed separation branches also refuse;
their scientific semantics are not redefined.

`project_constructor` requires explicitly exposed raw parameter dictionaries.
It captures and validates them before calling an owner-supplied projector with
a separate copy. It records constructor configuration, not object identity or
learned state. It does not obtain parameters by guessing or provide a numerical
safety guarantee for arbitrary projectors. Tests use only synthetic projectors
and a nonnumeric dummy constructor.

Source envelopes are bounded by 1 MiB, including their encoded metadata and
canonical validation overhead. Their `sha256:` digest covers exact envelope
payload bytes. `read_source` checks schema, content hash, original-text/document
agreement, and exact conversion validity. Accessors verify again and return
fresh containers. Capture charges normalized document and conversion-record
bytes during traversal, including constructor preflight before its projector.
Conversion-free shared subtrees can reuse measured costs; tuple conversions
are revisited at each actual source location. Final envelope validation also
charges its wrapper, original text and metadata, so the declared cap is not a
promise to accept 1 MiB of raw input. Envelope wrapping consumes one depth level.
Invalid finetune ranges report the original tuple path, and invalid mapping keys
report their original containing dictionary rather than a generated envelope path.
Bytes remain in memory: content-addressed file storage,
copy/verification before cross-workspace merge, and SQLite publication belong
to later phases. A source digest never proves runtime binding.

## Mandatory completion before rollout

Complete the accepted representation profile for reconstructible typed keys,
tuples and three-tuple finetune semantics, with exact round-trip and semantic
tests and updated consumers. Do not ship a parallel old-profile path. Obtain
owner exposure for genuinely inaccessible constructor parameters or refuse
honestly.

Owner effective-seed/located-choice exposure, complete native correlation and
selection transport, DMD input identity, qualified lanes, one-time CV-pool
preservation and per-occurrence lowering remain mandatory. Seedless/unproven
binding refuses during interim preparation; supported unseeded requests must
not be disabled at rollout. No seed or choice is invented by this codec.

The shared admission gate and active-caller migration are not implemented.
Legacy durable entry points have no NULL-manifest workaround. First-qualified
schema ensure under a write lock, same-INSERT manifest publication, one-workspace
batch atomicity and source-envelope transfer must be implemented and tested
later. Files, charts, Parquet and artifacts remain outside SQLite guarantees.

An accepted pinned Arena implementation/adapter is required. Owner IDs remain
separate from Arena-derived IDs; unknown safety is unranked and unsafe never
ranks. Fold, partition and `avg`/`w_avg`/`final`/`final_agg` observations remain
distinct. No score, historical identifier or scientific reference is recomputed.
Deleted historical SQLite/array artifacts remain unavailable and their engine
unknown. Studio-facing preflight, frontend, private sources and frozen references
are unchanged. WORK review and root authorization precede any live integration.
