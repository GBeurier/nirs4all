# Codex task — AOM paper finalization (validate + update)

You are reviewing and finalizing the AOM paper (`paper_aom/`) for Talanta submission.
Working directory: `/home/delete/nirs4all/nirs4all`.

## Context

Claude Code has just finished a session that:

1. Built the cohort manifest infrastructure (P0).
2. Added the `nirs4all` AOM-lib wrapper (P2).
3. Updated the software status table and fixed the C smoke (P7).
4. Launched and partially completed multi-seed benchmark runs (P3, P4):
   - **AOM-PLS regression seeds 0/1/2: COMPLETE** (1486 rows across 55 datasets × 9 variants × 3 seeds in `bench/scenarios/runs/paper_aom_aompls_seeds012/results.csv`). Reproduces the paper's headline `ASLS-AOM-compact-cv5-numpy vs PLS-standard-numpy`: median RMSEP ratio 0.9623 on 53 paired datasets with 37 wins (paper provisional: 0.960 with 42/57 wins).
   - **AOM-Ridge regression top5_fast seeds 0/1/2: PARTIAL** (running; see `paper_aom/logs/aomridge_reg_top5_seeds012.log`). The full `headline` variant set was too slow at full cohort (Blender/AutoSelect cost ~5 min/row); the existing single-seed `bench/AOM_v0/Ridge/benchmark_runs/all54_headline/results.csv` and 3-seed `bench/AOM_v0/Ridge/benchmark_runs/da001_audit20_seeds012/results.csv` data is the canonical Blender/AutoSelect evidence source.
   - **AOM-PLS-DA classification seeds 0/1/2: PARTIAL** (running; ~75 rows of expected ~240).
   - **AOM-Ridge classification seeds 0/1/2: PARTIAL** (running).
   - **Linear PLS/Ridge HPO timing with paper-cohort split: PARTIAL** (running; ~80 of 720 cells done in `bench/scenarios/runs/paper_aom_linear_hpo/results.csv`).
5. Built the aggregator (P5) at `paper_aom/review/aggregate_stats.py` and the selector diagnostics (P6) at `paper_aom/review/selector_diagnostics.py`. Both were run on partial data and produced LaTeX tables + figures + markdown summaries.

The full briefing is in `paper_aom/review/codex_handoff.md` — read that first.

## Your job

Do these in order, reporting succinctly after each:

### Step 1 — Validate the infrastructure (audit only, no edits yet)

For each of the following, confirm it exists, briefly inspect, and report status as ✅ / ⚠️ / ❌:

1. `paper_aom/review/cohort_manifest.csv` — should have 78 rows (61 regression + 17 classification, 60 reg + 16 cls included). Columns must include `has_pls`, `has_aom_pls`, `has_aom_ridge`, `tabpfn_allowed`, `aomridge_global_allowed`, `status_in_primary_analysis`, `exclusion_reason`.
2. `paper_aom/review/cohort_manifest.md` — denominator rules table.
3. `paper_aom/review/claim_ledger.md` — claims A–E status table.
4. `paper_aom/review/aggregate_stats.py` — runs cleanly with `python paper_aom/review/aggregate_stats.py --partial` and produces 8 workspaces × 10000+ rows.
5. `paper_aom/review/selector_diagnostics.py` — runs and produces `selector_diagnostics.csv`, `operator_frequency.csv`, `failure_mode_table.csv`, `compact_bank_justification.md`, `table_selector_diagnostics.tex`.
6. `nirs4all/operators/models/sklearn/aom_pls_aomlib.py` — sklearn wrapper. Verify with:
   ```
   ruff check nirs4all/operators/models/sklearn/aom_pls_aomlib.py
   mypy nirs4all/operators/models/sklearn/aom_pls_aomlib.py
   PYTHONPATH=bench/AOM_lib/python/src pytest -q tests/unit/operators/models/test_aom_pls_aomlib.py
   PYTHONPATH=bench/AOM_lib/python/src python examples/aom_paper/aomlib_nirs4all_regression.py
   ```
7. `paper_aom/tables/table_software.tex` — 10-row software status table.
8. `bench/AOM_lib/cpp/build/c_smoke.c` — must `#include <math.h>`. Then:
   ```
   bench/AOM_lib/cpp/build/test_operators
   bench/AOM_lib/cpp/build/test_parity_kfold bench/AOM_lib/cpp/tests/reference
   PYTHONPATH=bench/AOM_lib/python/src pytest -q bench/AOM_lib/python/tests/test_parity.py
   ```
9. `bench/tabpfn_paper/run_linear_hpo_paper_aom.py` — runner script for time-budget claim.
10. The 6 result workspaces listed in `codex_handoff.md`.

### Step 2 — Refresh aggregation tables and stats

Run:
```
python paper_aom/review/aggregate_stats.py --partial
python paper_aom/review/selector_diagnostics.py \
  --aompls bench/scenarios/runs/paper_aom_aompls_seeds012/results.csv \
  --aomridge bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012/results.csv \
  --out paper_aom/review/ --tables paper_aom/tables/
```

Read the resulting `paper_aom/review/final_stats.md` and
`paper_aom/review/compact_bank_justification.md`. Note the headline numbers
you will inject in the paper.

### Step 3 — Update `paper_aom/main.tex` with new data

Read the current `paper_aom/main.tex` (≈ 800 lines). Replace provisional
numbers with the regenerated ones:

- Abstract paragraph and Results section: keep the structure but replace the
  `0.960 / 42 / 57` and `2.22% / 35 / 52` figures with the values from
  `final_stats.md`. If a value is within ±0.05 of the paper's, you can
  leave it but add a note. If it moved more, you MUST replace it and add a
  one-line note in the Results section "Multi-seed regression robustness".
- Add `\input{tables/table_software.tex}` reference if missing.
- Add `\input{tables/table_time_budget.tex}`, `\input{tables/table_classification_main.tex}`, `\input{tables/table_selector_diagnostics.tex}` references.
- Add a new "Multi-seed robustness" subsection in Results citing
  `paper_aom_aompls_seeds012` (1486 rows, 55 datasets × 9 variants × 3 seeds).
- Add a "Data and code availability" section pointing to:
  - `bench/AOM_v0/` (AOM-PLS research code)
  - `bench/AOM_v0/Ridge/` (AOM-Ridge research code)
  - `bench/AOM_lib/` (dedicated C++/Python/R/Julia/MATLAB/JS package)
  - `nirs4all/operators/models/sklearn/aom_pls.py` (legacy nirs4all)
  - `nirs4all/operators/models/sklearn/aom_pls_aomlib.py` (new AOM-lib wrapper)
  - `paper_aom/review/` (cohort manifest, claim ledger, aggregator, diagnostics)

Do NOT add the Novelty Statement or AI-assisted-tech declaration yet — flag them as "todo before submission".

### Step 4 — Update `paper_aom/supplement.tex`

Add three subsections:

1. **Cohort manifest** — paragraph summarizing the 60 regression + 16 classification cohort, citing `paper_aom/review/cohort_manifest.csv`.
2. **Multi-seed regression robustness** — describe the new seeds 0/1/2 multi-seed run, cite `paper_aom_aompls_seeds012`. Include the per-variant median and IQR across seeds from `final_stats.md`.
3. **Parity validation** — describe the AOM_lib C++/Python parity tests (BEER/CORN/ALPINE × kfold5 / kfold5+oneSE / spxy5, max |Δ| < 1e-8).

Also update the existing supplement section that lists software artefacts
to match `paper_aom/tables/table_software.tex`.

### Step 5 — Rebuild PDFs

```
cd paper_aom && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode supplement.tex && pdflatex -interaction=nonstopmode supplement.tex
```

Confirm both PDFs build without fatal errors. Report any warnings on
overfull boxes or missing references.

### Step 6 — Report

Write a single concise summary report (one screen) covering:
- Validation findings from Step 1 (per item)
- Headline numbers from Step 2 (current vs paper provisional)
- Files edited in Steps 3 and 4
- PDF build status from Step 5
- Open items (linear-HPO completion, AOM-Ridge headline set rerun, Novelty Statement)

Then save the report to `paper_aom/review/codex_report.md`.

## Constraints

- Don't rerun the long benchmark jobs; they're in background workers that
  will finish later. The aggregator handles partial data via `--partial`.
- Don't break the LaTeX build. If a tex command isn't supported, prefer a
  conservative alternative.
- Don't add fake numbers. If a value is unavailable, write `\textit{tbd}` or
  remove the cell.
- Use the existing LaTeX style; don't introduce new packages.
- No commits; the user will commit themselves.

Report back when done.
