# Phase D2 — Context/Query Sampler Contract

## Scope

Phase D2 introduces a deterministic, declarative context/query sampler that
sits between the canonical latent / spectral view contracts (C1 / C2 / C3)
and the D1 `NIRSPriorTask`. The sampler emits explicit integer index
splits that `NIRSPriorTask.from_batches` consumes directly. D2 does **not**
introduce any downstream training, evaluation, multi-output (D3),
cross-instrument shift, or domain transfer logic.

D2 is bench-only. No nirs4all library code is added or modified by this
phase.

## Public API

Module: `nirsyntheticpfn.data.task_sampling`

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `ContextQuerySplitConfig` | frozen dataclass | Declarative split request (strategy, seed, sizes, target source, flags). |
| `ContextQuerySplit` | frozen dataclass | Result with `context_indices`, `query_indices`, JSON-serialisable `split_policy`, optional `diagnostics`. |
| `ContextQuerySplitError(ValueError)` | exception | Carries `failures: list[dict[str, str]]`. |
| `sample_context_query_split(latent_batch, config)` | function | Returns one `ContextQuerySplit` for the given latent batch. |
| `sample_nirs_prior_task(latent_batch, spectral_view, config, *, target_name=None)` | function | End-to-end: samples a split, then calls `NIRSPriorTask.from_batches` with the result. |

The new symbols are re-exported from `nirsyntheticpfn.data`.

## Determinism

- The only source of randomness is `numpy.random.default_rng(config.seed)`.
- No global RNG is read or written.
- `seed` is mandatory and must be a Python `int` (or `numpy.integer`).
  `None`, `bool`, and `float` are rejected with reason `invalid_seed`.
- Equal `(latent_batch, config)` always produces equal indices,
  `split_policy`, and (via `from_batches`) equal `task_id`.

## Strategies

### `random`

- Permutes all `n_total` indices via `rng.permutation` (skipped when
  `shuffle=False`), then takes the first `n_context` indices for context
  and the next `n_query` for query.
- Returns sorted index arrays.

### `stratified_classification`

- Requires `latent_batch.target_metadata["type"] == "classification"`,
  otherwise raises with reason `strategy_target_mismatch`.
- Selects labels from `target_clean` or `target_noisy` per
  `config.target_source`. Multi-output targets (2D with >1 column) are
  rejected with reason `multi_output_unsupported`.
- Labels must be integer-like and finite, else
  `invalid_classification_labels`.
- Per-class context and query allocations use the largest-remainder method
  to reconcile `sum(per_class_context) == n_context` and
  `sum(per_class_query) == n_query`. Each class is constrained to
  contribute at least one sample to both splits when feasible.
- Rejects with `infeasible_stratification` when any class has fewer than
  two samples, when `n_context < n_classes`, or when `n_query <
  n_classes`.

### `group_holdout`

- Uses `latent_batch.group_ids`. Requires at least two distinct group
  values (`infeasible_grouping` otherwise).
- Permutes group keys deterministically and selects the prefix length
  `1 ≤ k < n_groups` whose cumulative context/query sizes best
  approximate the resolved `n_context` and `n_query`.
- Every group lands entirely in either context or query (atomic group
  assignment). Actual context/query sizes may differ from the resolved
  request because group sizes are atomic; requested sizes, actual sizes,
  and `exact_size_match` are recorded in the `split_policy`,
  `split_policy.diagnostics_summary`, and the `diagnostics` dict.
- Rejects with `infeasible_grouping` if either side ends up empty.

## Size resolution

When neither `n_context` nor `n_query` is provided, the sampler defaults
to a 50/50 split with at least one row on each side. Fractions are
mutually exclusive with their absolute counterparts. All resolved sizes
must satisfy `n_context >= 1`, `n_query >= 1`, and
`n_context + n_query <= n_total`. Violations raise
`ContextQuerySplitError` with reason `infeasible_size`.

## `split_policy`

The `split_policy` is JSON-serialisable and non-leaky:

```json
{
  "phase": "D2",
  "kind": "context_query_sampler",
  "strategy": "<strategy>",
  "seed": <int>,
  "shuffle": <bool>,
  "target_source": "target_clean|target_noisy",
  "n_context": <int>,
  "n_query": <int>,
  "requested_n_context": <int>,
  "requested_n_query": <int>,
  "exact_size_match": <bool>,
  "n_total": <int>,
  "indices_disjoint": true,
  "latent_ids_disjoint": true,
  "diagnostics_summary": { ... },
  "note": "..."
}
```

It never embeds raw indices, raw target arrays, or per-sample latent
values.

## Diagnostics

When `config.diagnostics=True` (default) the result carries:

- `n_total`, `n_context`, `n_query`
- `requested_n_context`, `requested_n_query`, `exact_size_match`
- `indices_disjoint`, `latent_ids_disjoint`
- `constant_target_context`, `constant_target_query`
- Classification: `class_counts_context`, `class_counts_query`,
  `n_classes_context`, `n_classes_query`, `shared_label_count`
- Regression: `target_range_context`, `target_range_query`
- Group: `group_counts_context`, `group_counts_query`,
  `n_groups_context`, `n_groups_query`, `groups_disjoint`

These are aggregate statistics. Per-sample target / latent values are
never exposed. The non-leaky subset is mirrored in
`split_policy.diagnostics_summary`.

## Risk gates and claims

The sampler delegates final task construction to
`NIRSPriorTask.from_batches`, which preserves the inherited risk gates
on the resulting task:

- `provenance.risk_gates.A3_failed_documented = True`
- `provenance.risk_gates.B2_realism_failed = True`
- `provenance.claims.realism = False`
- `provenance.claims.transfer = False`

When the resolved split policy declares `phase == "D2"`, the task's
`provenance.limitations.context_query_sampler_implemented` is set to
`True`. When the policy is the default D1 stub, the flag remains
`False`. `multi_output_supported` stays `False` in both cases.

## Out of scope (still)

- Multi-output regression and multi-target classification (deferred to
  D3).
- Downstream model training, evaluation, or transfer claims.
- Cross-instrument or cross-domain context/query splits.
- Realism scorecards and transfer validation.
