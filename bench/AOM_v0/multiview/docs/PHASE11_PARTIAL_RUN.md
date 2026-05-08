# `adaptive-super-learner` Phase-11 — Partial Run Documentation

**Owner**: Agent A (Classical Production Pack)
**Date**: 2026-05-05 (revised 17:30 CEST after Codex round 1)
**Status**: `DECISION_LOCKED` (Codex round 1: REVISE A) — atom-set guard
reframed as a separately named guarded config; per-atom wall-clock aborts
replace the flat 4 h timeout. Original D-A-002 logged at 14:04 SYNC.
**Scope**: completion plan or formal exclusion memo for the Phase-11
"Adaptive Super Learner" full-57 run that was killed at 1h30 with 38 / 57
datasets completed.

This document closes the partial-run question raised in
`bench/PLAN_REPRISE_2026-05.md` §6 A3 ("AdaptiveSuperLearner Phase-11 full-57 |
35/61, kill 1h30 | relancer avec timeout > 4h, logs wall-clock, exclusions
documentées").

---

## 1. Observed coverage

Source: `bench/benchmark_master_results.csv` (post-P0 freeze, 2026-05-05 14:30).
All 54 rows for `variant == "adaptive-super-learner"` carry
`protocol_maturity=exploratory`.

| `source_run` | Datasets | Status counts |
|---|---:|---|
| `bench/AOM_v0/multiview/results/full57.csv` | 38 | 38 OK + 2 ERROR (40 rows) |
| `bench/AOM_v0/multiview/results/smoke10.csv` | 10 | 10 OK |
| `bench/AOM_v0/multiview/results/smoke4_baseline.csv` | 4 | 4 OK |

The plan's "35 / 61" prose came from the multiview-internal cohort definition
(61 datasets). Aligned with the canonical 57-dataset cohort, the partial
coverage is **38 / 57** (with 2 of those 38 in error status). Effective
clean coverage: **36 / 57** (~63%).

The two error rows were inspected via the master CSV `error_message` column;
they are upstream PLS-baseline grid bugs (same family as the r20 single
ERROR row Agent B documented), not failures of the ASL stack itself.

## 2. Failure mechanism

Per `bench/AOM_v0/multiview/docs/SUMMARY.md §16`:

```text
The Phase 11 full-57 run was killed at 1.5 hours (35 of 61 datasets done)
because the larger n × p datasets ran into NNLS-stacker overhead (5-fold OOF
× 4 atom bases × heavy fits like AOM-Ridge component).
```

The atom set used in Phase-11 is fixed at:

```
{multiK-3-5-7, moe-preproc-soft, lazy-V2-AOM, AOM-PLS-compact}
```

(4 atoms after the Codex-reviewed split between atoms and recipes — see
`PHASE11_STRATEGY.md` and `SUMMARY.md` §16). The phrase "AOM-Ridge component"
in the kill log refers to the **kernel-style cost** carried by `AOM-PLS-compact`
and `lazy-V2-AOM` on big-n inputs (both have an internal AOM block that
behaves like a kernel-ridge sub-fit), **not** to a separate `AOM-Ridge`
atom. This matters for the guard rule below: the expensive atoms to drop at
big-n are `AOM-PLS-compact` and `lazy-V2-AOM`, leaving the cheaper
`{multiK-3-5-7, moe-preproc-soft}` pair.

Dominant per-dataset cost:

- 4 atoms × 5-fold OOF = 20 atom fits per dataset for the NNLS meta-stacker.
- `AOM-PLS-compact` and `lazy-V2-AOM` each fit in O(n²) on their internal
  block solves; for `n_train > 3 000` they dominate wall-clock.
- The `min_margin=0.005` recipe-vs-NNLS circuit-breaker fires only after the
  full OOF pass, so it does not save wall-clock on big-n.

The 19 missing datasets in `full57.csv` cluster on `n_train > 1 500`; many
overlap with the Big-n class enumerated in `AOMRIDGE_BIGN_OOM.md` §3.

## 3. Locked remediation — `adaptive-super-learner-bigN-guarded` config

**Codex round 1 verdict (D-A-002)**: REVISE A. The original draft proposed
silently merging an `n>3 000` atom-drop guard into the existing
`adaptive-super-learner` row. Codex requires that the guarded variant be
exposed as a **separately named** registry config so consumers can tell the
two rows apart, and replaces the flat 4 h dataset timeout with a
**per-atom wall-clock abort** because the original 4 h × 19 = 76 h
worst-case is unrealistic for a single-seed budget.

### 3.1 New config: `adaptive-super-learner-bigN-guarded`

Proposed registry entry (Agent A drafts; Agent C owns commit):

```yaml
- canonical_name: adaptive-super-learner-bigN-guarded
  aliases:
    - adaptive-super-learner-bigN-guarded
  model_class: AdaptiveSuperLearner
  module: bench.AOM_v0.multiview.adaptive_super_learner
  config_template: bench/scenarios/configs/adaptive_super_learner_bigN_guarded.yaml
  task_types: [regression]
  input_constraints: {min_n: 3001}
  supports_predefined_test_split: true
  inner_cv_nested: true
  runtime_tier: slow
  maturity: exploratory
  notes: |
    Phase-11 ASL with atom-set guard active. Drops AOM-PLS-compact and
    lazy-V2-AOM from the NNLS atom pool; keeps {multiK-3-5-7, moe-preproc-soft}.
    Use only for n_train > 3000 datasets. Result rows are tagged
    extras.atom_guard=true so the synthesis can keep them out of the
    apples-to-apples comparison with the unguarded ASL.
```

The original `adaptive-super-learner` entry keeps its full atom set and
applies to `n_train ≤ 3 000` (or whichever cap the guarded config's
`min_n: 3001` mirrors). Two rows per big-n dataset only happen if a future
seed run produces them; the synthesis keeps them in disjoint groups for §10.2
tests.

### 3.2 Per-atom wall-clock abort (replaces flat 4 h)

Each atom inside the NNLS / recipe-select scan gets a budget; if exceeded,
the atom is recorded as `status=atom_timeout` for that fold and the
meta-stacker proceeds with the remaining atoms.

| Atom | Per-fold budget | Per-dataset budget (5-fold) |
|---|---:|---:|
| `multiK-3-5-7` | 60 s | 5 min |
| `moe-preproc-soft` | 60 s | 5 min |
| `lazy-V2-AOM` | 300 s | 25 min |
| `AOM-PLS-compact` | 300 s | 25 min |
| **Total per dataset** | — | **60 min** (with all 4 atoms) |

The guarded config drops the two heavy atoms, so its per-dataset budget is
**~10 min**. Any individual atom that times out leaves a structured row in
`extras.atom_timeouts` so the synthesis knows the NNLS solve is partial.

### 3.3 Completion run

Run the guarded config on the 19 missing datasets:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/multiview \
python bench/AOM_v0/multiview/benchmarks/run_full57.py \
    --variants adaptive-super-learner-bigN-guarded \
    --datasets phase11_completion_cohort.csv \
    --workspace bench/AOM_v0/multiview/results/full57_phase11_completion \
    --seed 0 \
    --per-atom-budget-config bench/scenarios/configs/atom_budgets.yaml
```

The unguarded `adaptive-super-learner` row stays as-is on the 38 datasets
where it completed; tag those rows `extras.atom_guard=false` for symmetry.

### 3.4 Master CSV tagging

| Subset | Variant | `protocol_maturity` | `extras.atom_guard` |
|---|---|---|---|
| 38 datasets where ASL completed | `adaptive-super-learner` | `exploratory` (until multi-seed) | `false` |
| 19 big-n datasets after completion | `adaptive-super-learner-bigN-guarded` | `exploratory` (different atom set) | `true` |

Both subsets stay `exploratory` until §5's multi-seed gate is satisfied. The
two registry entries are NOT pooled in §10.2 tests because they use
different atom pools.

## 4. Alternative (fallback)

If even the guarded config exceeds the 10 min per-dataset budget on any
specific big-n dataset:

- Keep that dataset's row absent from the master.
- Add a row in
  `bench/AOM_v0/multiview/docs/PHASE11_EXCLUDED_DATASETS.csv` with columns:
  `dataset_group, dataset, n_train, p,
  reason ∈ {timeout_unguarded, timeout_guarded, oom},
  planned_action ∈ {rerun_with_smaller_atoms, drop_permanently}`.
- The synthesis surfaces the exclusion via Agent C's exporter `low_coverage`
  penalty.

## 5. Promotion gate (proposal)

| Tier | Required evidence | Today |
|---|---|---|
| `exploratory` | Variant runs end-to-end on at least the smoke cohort | **Met** — current state |
| `audit` (`audit20_transfer_core`) | Coverage of the 20-dataset audit cohort + ≥3 seeds | **Not met** — single seed=0, partial coverage |
| `strong_practical` | Full-57 with completion or exclusion memo + ≥3 seeds | **Blocked** by §3 / §4 |
| `best_current` | Strong_practical + Friedman-Nemenyi + Nadeau-Bengio per §10.2 | **Blocked** |

Even with §3 completion, ASL stays out of `strong_practical` until the
multi-seed gate (≥3 seeds) is satisfied. Multi-seed for ASL is expensive
because each seed re-runs the entire OOF stack; budget estimate: 5 seeds ×
57 datasets × 1.5 h average = ~430 h sequentially, ~100 h with `n_jobs=-1`.
Not in scope for this round; deferred to the post-A2 timeline (§5 of
`bench/AOM/ROADMAP.md` would absorb it via M7's seed discipline).

## 6. Acceptance criteria

The Phase-11 partial-run question is closed when **all** of the following
hold:

- 38 / 57 unguarded `adaptive-super-learner` rows stay as-is, tagged
  `extras.atom_guard=false`, `protocol_maturity=exploratory`.
- 19 / 57 big-n `adaptive-super-learner-bigN-guarded` rows land in the
  master, tagged `extras.atom_guard=true`, `protocol_maturity=exploratory`.
- Any dataset still exceeding the §3.2 per-atom budgets is recorded in
  `PHASE11_EXCLUDED_DATASETS.csv` rather than as a placeholder row.
- A two-paragraph note in `bench/AOM_v0/multiview/docs/IMPLEMENTATION_LOG.md`
  records the locked decision and the date.

## 7. Decision history

- 2026-05-05 — D-A-002 posted with status `DECISION_PENDING_CODEX_REVIEW`,
  default remediation A (full completion + flat 4 h timeout).
- 2026-05-05 17:00 CEST — Codex round 1: REVISE A. Two corrections:
  (i) atom-set guard must be a separately named registry config
  (`adaptive-super-learner-bigN-guarded`), not silently merged into the
  unguarded ASL row; (ii) flat 4 h dataset timeout replaced with per-atom
  wall-clock budgets (§3.2). Codex also flagged an internal inconsistency
  in §2 between "AOM-Ridge atom" wording and the actual atom set; the
  rewrite above clarifies that the kernel-style cost lives inside
  `AOM-PLS-compact` and `lazy-V2-AOM`, not in a separate `AOM-Ridge` atom.
- 2026-05-05 17:30 CEST — D-A-002 status flipped to `DECISION_LOCKED`.
  Implementation = the completion run §3.3 once the guarded config lands
  in the registry (Agent C edit + Codex sanity).
