# Master CSV Freeze Report — 2026-05-05

**Owner**: Agent C-bootstrap (per `bench/PLAN_REPRISE_2026-05.md` section 5)
**Scope**: P0 freeze of `bench/benchmark_master_results.csv`, schema introduction of `protocol_maturity`, counter reconciliation, and audit of three plan-flagged runs.
**Sibling artefacts**:
- `bench/MASTER_CSV_HASH.txt` — date / command / SHA256 / row count / size.
- `bench/SYNC.md` — append-only multi-agent log; this freeze writes a `DECISION_PENDING_CODEX_REVIEW` entry.

---

## 1. Snapshot

| Field | Value |
|---|---|
| Date (snapshot) | 2026-05-05 |
| Builder | `bench/build_benchmark_synthesis.py` (single canonical edit point) |
| Output CSV | `bench/benchmark_master_results.csv` |
| SHA256 | `b27ea6f52b45e2568fb0c6912f535565f678d8b3e4f28af70dc2b86ae201ab5d` |
| Size | 17 285 004 bytes |
| Header rows | 1 |
| Data rows | 21 769 |
| Pre-freeze SHA256 | `3f1d596b835346f7a66fada968d8193a037dfd5fb8ebdbdd67ea7fc1e839390b` (master_pivot of 17 124 448 bytes) |
| Builder reproducibility | Pre-freeze: builder reproduced the existing master byte-identical. Post-freeze: re-running the builder is idempotent (verified 2026-05-05). |

The pre-freeze master and the post-freeze master differ only by the new `protocol_maturity` column appended at the end of every record. No score values, ratios, oracle assignments, or row counts moved.

## 2. Counter reconciliation

The plan flagged a 21 769 vs 20 964 vs 83 discrepancy. Resolution:

| Counter | Value | Definition |
|---|---:|---|
| Master CSV total rows | 21 769 | Every row produced by the builder. |
| Source rows (`record_type ∈ {observed, reference_paper, source_oracle}`) | 20 964 | Authentic source/reference rows ingested from `nicon_v2`, `AOM_v0`, `AOM_v0/Ridge`, `AOM_v0/Multi-kernel`, `AOM_v0/multiview`, `tabpfn_paper`, `fck_pls`, `paper_master_pivot`. |
| Derived oracle rows (`record_type ∈ {oracle_by_model_class, oracle_global_dataset}`) | 805 | Synthesised by `enrich_with_oracles`: 719 per (dataset, task, metric, model_class) winners + 86 per (dataset, task, metric) global winners. |
| Eligible source rows | 19 783 | Rows passing `eligible(...)`: `record_type ∈ {observed, reference_paper}`, status OK, score metric present, evaluation split not in {train, cv}. |
| Distinct datasets in eligible rows | 83 | Count of distinct `dataset` values among eligible source rows. |
| (dataset, task) pairs in eligible rows | 86 | Some datasets have both regression and classification entries. |

**Diagnosis of the previous prose**: `bench/benchmark_synthesis.md` previously read "covering {dataset_count} eligible dataset/task pairs", but `dataset_count` in the builder counted distinct datasets, not pairs. The freeze rewrites the prose to "covering 83 distinct eligible datasets across 86 (dataset, task) pairs" and computes both numbers separately in `write_md`. No data change; only language alignment.

## 3. Builder change summary (`bench/build_benchmark_synthesis.py`)

| Change | Location | Effect |
|---|---|---|
| Append `protocol_maturity` to `FIELDNAMES` | top of file | New column at the end of every CSV row. |
| Add `EXPLORATORY_PHASE11_ATOMS`, `EXPLORATORY_RUN_NAMES`, smoke regex constants | top of file | Single-source data for exploratory tagging rules. |
| Add `assign_maturity(record)` | above `main()` | Tag function. PROVISIONAL rules; pending Codex review. |
| Add `maturity_summary(records)` helper | above `main()` | Distribution counter used by `main()` and `write_md`. |
| `main()` now calls `assign_maturity` for every record before `write_csv` | `main()` | All rows tagged; idempotent. |
| `write_md` now computes `pair_count` separately and emits a "Protocol maturity distribution" section + an updated CSV-notes bullet for the new column | `write_md` | Synthesis MD reflects the new schema and corrects the dataset-count prose. |

No other logic changed. Pre-existing ruff issues (`I001`, `C420`, `SIM103`) on this file were not touched in P0 (out of scope for the freeze).

## 4. `protocol_maturity` schema

Allowed values:

| Tag | Definition (this freeze) | Rows |
|---|---|---:|
| `locked` | Stable, production-eligible source row from a non-smoke, non-diagnostic source run. | 19 392 |
| `exploratory` | Partial coverage / smoke / single-dataset diagnostic / awaiting nested audit. Excluded from production presets until the owner promotes the row via SYNC.md. | 1 513 |
| `legacy` | Explicitly superseded run. Reserved for owner-driven downgrades through SYNC.md; **not auto-applied** in this freeze (count 0). | 0 |
| `oracle` | Derived oracle row (`record_type ∈ {oracle_by_model_class, oracle_global_dataset}`). | 805 |
| `local_not_master` | `source_oracle` row already present in a source table (e.g. `meta_selector_full57.csv` `oracle_rmsep` column). Kept for audit; excluded from derived oracle calculations. | 59 |
| **total** | | **21 769** |

### 4.1 Tagging rules in `assign_maturity` (PROVISIONAL)

Order of precedence (first match wins):

1. `record_type ∈ {oracle_by_model_class, oracle_global_dataset}` → `oracle`.
2. `record_type == "source_oracle"` → `local_not_master`.
3. `model_name`/`variant` matches `headline-spxy3` AND family is AOM-Ridge (`aomridge`/`aom-ridge` substring) → `exploratory`. Reason: pending nested audit per plan section 6 task A2.
4. `source_path` contains `multiview` AND `source_run == "full57.csv"` AND `model_name ∈ {adaptive-super-learner, nnls-stack-atoms, nnls-stack-calibrated, trimmed-mean-4}` → `exploratory`. Reason: Phase-11 super-learner / atoms partial coverage (38–39 of the 61-dataset cohort) per plan section 6 task A3.
5. `source_run` ∈ explicit exploratory whitelist (multiview smoke files; nicon_v2 single-dataset / partial multiseed diagnostics; AOM_v0 / Ridge known smokes) → `exploratory`.
6. `source_run` matches `^smoke...`, `_smoke$`, or `^phase1[a-z]_smoke$` → `exploratory`.
7. Default → `locked`.

### 4.2 Distribution by source_run (highlights)

| source_family | source_run | rows | tag mix |
|---|---|---:|---|
| AOM_v0 | full | 7 888 | locked: 7 888 |
| AOM_v0 | smoke | 80 | exploratory: 80 |
| AOM_v0 | smoke_old_11ds | 36 | exploratory: 36 |
| AOM_v0_Ridge | all54_combined | 882 | locked: 776, exploratory: 106 (spxy3) |
| AOM_v0_Ridge | all54_headline | 534 | locked: 424, exploratory: 110 (spxy3) |
| AOM_v0_Ridge | smoke_cv5 | 51 | exploratory: 51 |
| AOM_v0_Ridge | smoke6 | 42 | exploratory: 42 |
| AOM_v0_Ridge | v5b_diverse | 29 | locked: 19, exploratory: 10 (spxy3) |
| AOM_v0_multiview | full57.csv | 939 | locked: 786, exploratory: 153 (38 ASL + 38 nnls-stack-atoms + 38 nnls-stack-calibrated + 39 trimmed-mean-4) |
| AOM_v0_multiview | meta_selector_full57 | 697 | locked: 638, local_not_master: 59 |
| AOM_v0_multiview | smoke10.csv | 318 | exploratory: 318 |
| AOM_v0_multiview | smoke4_baseline.csv | 113 | exploratory: 113 |
| AOM_v0_multiview | smoke_classification.csv | 22 | exploratory: 22 |
| nicon_v2 | r20_curated_oof | 195 | locked: 195 |
| nicon_v2 | r20_curated_oof_multiseed | 3 | exploratory: 3 |
| nicon_v2 | phase1*_smoke (a/b/c) | 174 | exploratory: 174 |
| paper_master_pivot | master_pivot | 335 | locked: 335 |
| tabpfn_paper | table_results_tabpfn_final_light | 60 | locked: 60 |

The smoke/legacy regex coverage is intentionally conservative. Owners A and B may downgrade additional source_runs to `exploratory` or `legacy` via SYNC.md; the builder is the single edit point.

## 5. Audit of plan-flagged runs

### 5.1 r20_curated_oof — TAGGED `locked` (provisional, awaiting B1 audit)

| Aspect | Finding |
|---|---|
| Source path | `bench/nicon_v2/benchmark_runs/r20_curated_oof/results.csv` |
| Source rows in master | 195 |
| Distinct datasets | 39 (matches plan section 7 B1 description) |
| Variants | `Ridge-baseline`, `PLS-baseline`, `V2L-learnableRMS`, `V2L-Residual-AOMPLS`, `V2L-Boost-AOMPLS` (5 × 39 = 195) |
| Seed | single seed = 0 across all rows |
| `cv_protocol` | `predefined` (predefined train/test holdout split) |
| `cv_fold` | -1 (aggregate row, not per-fold) |
| Status | 194 OK, 1 ERROR |
| Tag | `locked` |

Caveat: per plan section 7 task B1, Agent B is to **confirm OOF cleanness** (no leakage of test data into selection within `V2L-Residual-AOMPLS` and `V2L-Boost-AOMPLS`). The freeze tags these rows `locked` because the protocol structure is consistent with do-no-harm OOF (single seed, predefined holdout, OOF used internally for residual selection only). If B1 surfaces a leakage path, the rule in `assign_maturity` will be tightened in a follow-up commit and SYNC.md will be updated.

The companion run `r20_curated_oof_multiseed` (3 rows) is tagged `exploratory` because of its partial coverage; it is not the production multiseed run announced in plan section 7 task B2.

### 5.2 AdaptiveSuperLearner Phase-11 (and atoms) — TAGGED `exploratory`

| Aspect | Finding |
|---|---|
| Source path | `bench/AOM_v0/multiview/results/full57.csv` |
| Variants flagged | `adaptive-super-learner` (38 rows), `nnls-stack-atoms` (38), `nnls-stack-calibrated` (38), `trimmed-mean-4` (39) |
| Cohort | 61-dataset multiview cohort |
| Coverage | 38/61 ≈ 62 % for the super-learner itself; status counts 36 OK + 2 ERROR; the plan's "35/61, kill 1h30" reflects the run snapshot at the time of writing (recent Phase 11 commits 18f69973 / d7a8965a / 3df12ebb finalised at 38). |
| Tag | `exploratory` |

The 12 fully-covered (61/61) multiview variants — `PLS-standard-numpy`, `AOM-PLS-compact-numpy`, `lazy-V1-POP-blocks3-holdout`, `lazy-V2-AOM-combined-compact-holdout`, `moe-view-soft-pls`, `moe-preproc-soft-pls-compact`, `moe-view-soft-K5`, `moe-view-multiK-3-5-7`, `moe-view-multiK-3-5`, `mean-ensemble-3-fixed`, `mean-ensemble-4-fixed`, `moe-view-multiK-wide-2-10` — remain `locked`.

### 5.3 AOMRidge-Blender / AutoSelect headline-spxy3 — TAGGED `exploratory`

| Aspect | Finding |
|---|---|
| Variants | `AOMRidge-Blender-headline-spxy3` (128 rows), `AOMRidge-AutoSelect-headline-spxy3` (117 rows) |
| Total source rows | 245 |
| Distinct datasets | 53 |
| Source runs touched | `all54_headline` (110), `all54_combined` (106), `diverse_iter3_headline` (14), `v5b_diverse` (10), `diverse_iter3_wrappers_giants` (4), `v5b_smoke_alpine` (1) |
| Tag | `exploratory` (provisional) |

Reason: plan section 6 task A2 requires a nested OOF audit before either variant can be promoted. The freeze tags both `exploratory` so they are excluded from `strong_practical` / `best_current` presets until A2 ships a verdict. Promotion will be done by editing `assign_maturity` (single edit point) once Agent A delivers the audit and the decision is Codex-reviewed.

## 6. Counter consistency check (post-freeze)

| Check | Expected | Observed | Status |
|---|---:|---:|---|
| Total rows | 21 769 | 21 769 | OK |
| `protocol_maturity` non-empty | every row | 21 769 / 21 769 | OK |
| Sum of maturity counts | 21 769 | 19 392 + 1 513 + 0 + 805 + 59 = 21 769 | OK |
| `record_type=oracle_*` count | 805 | 805 | OK |
| `record_type=source_oracle` count | 59 | 59 | OK |
| `record_type=observed` count | 20 570 | 20 570 | OK |
| `record_type=reference_paper` count | 335 | 335 | OK |
| Eligible distinct datasets | 83 | 83 | OK |
| Eligible (dataset, task) pairs | 86 | 86 | OK |

## 7. Bloqueurs

None.

The three explicit blockers in plan section 5 are addressed:

1. **SHA256 + size + line count snapshotted** → `bench/MASTER_CSV_HASH.txt`.
2. **Builder reproduces master deterministically and is idempotent** → verified.
3. **`protocol_maturity` populated for every row via the builder** → verified via `maturity_summary`.

The Codex-review obligation on the schema and tagging rules is recorded in SYNC.md as `DECISION_PENDING_CODEX_REVIEW`, but it does not block downstream P0_DONE because the rules are conservative, single-edit-point, and easy to revise without rebuilding the master.

## 8. Conclusion

**P0_DONE** as of 2026-05-05.

Downstream agents A and B are unblocked. Agent C continues with section 8 of the plan (registry, exporter, harness, dashboard) under the standing Codex-review rule.
