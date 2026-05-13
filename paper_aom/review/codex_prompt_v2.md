# Codex task v2 — Final paper update after Linear HPO + AOM-Ridge runs complete

Working directory: `/home/delete/nirs4all/nirs4all`.

## What's new since v1 (codex_prompt.md)

Since the v1 pass that produced `paper_aom/review/codex_report.md`, the
following has happened:

1. **AOM-PLS-DA classification seeds 0/1/2 COMPLETED** (was running during v1).
   Final stats now in `paper_aom/review/classification_stats.md`:
   - AOM-PLS-DA-global-simpls-covariance vs PLS-DA: Δ=+0.159, 12/13 wins, $p_\mathrm{Holm}=0.007$ (significant).
   - POP-PLS-DA-simpls-covariance vs PLS-DA: Δ=+0.052, 11/13, $p_\mathrm{Holm}=0.018$ (significant).
   - AOM-PLS-DA-global-nipals-adjoint vs PLS-DA: Δ=+0.030, 8/13, $p=0.211$ (n.s.).
   - POP-PLS-DA-nipals-adjoint vs PLS-DA: Δ=−0.037, 5/13, $p=0.685$ (n.s.).

   Claude already updated the §"Classification as secondary validation"
   section in `paper_aom/main.tex` (formerly "Classification orientation")
   with this data. Please confirm the wording is appropriate for Talanta
   and update the abstract sentence about classification if needed.

2. **Linear PLS/Ridge HPO timing seeds 0/1/2 should now be complete** (was
   ~42% during v1). Confirm completion by checking
   `bench/scenarios/runs/paper_aom_linear_hpo/results.csv` row count == 720.
   If still running, use whatever rows exist and mark the table preliminary.

3. **AOM-Ridge classification seeds 0/1/2** is still trickling in. Use
   `bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_cls_seeds012/results.csv`
   for whatever rows exist; combine with the AOM-PLS-DA data for the
   classification table.

4. **AOM-Ridge top5_fast seeds 0/1/2 multi-seed regression** is still
   running (slow). Whatever rows exist (`paper_aom_aomridge_seeds012/results.csv`)
   add to the AOM-Ridge regression evidence as a supplementary multi-seed
   robustness check on the base candidates.

5. **Cohort manifest MD** has a post-Codex QUARTZ clarification appended.
   Please re-read `paper_aom/review/cohort_manifest.md` and confirm the
   wording is satisfactory.

## Required steps

### Step 1 — Refresh aggregation

```
python paper_aom/review/aggregate_stats.py --partial
python paper_aom/review/selector_diagnostics.py \
  --aompls bench/scenarios/runs/paper_aom_aompls_seeds012/results.csv \
  --aomridge bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012/results.csv \
  --out paper_aom/review/ --tables paper_aom/tables/
```

Read the refreshed `paper_aom/review/final_stats.md` and
`paper_aom/review/classification_stats.md`. Note the headline numbers,
especially:
- the linear-HPO time-budget median/q90/max numbers per variant (this is
  the crucial new claim the paper has been waiting on);
- whether the paired comparisons against `pls-tabpfn-hpo-25trials` and
  `ridge-tabpfn-hpo-60trials` now have enough N to be conclusive.

### Step 2 — Time-budget table

Now that linear-HPO data is more complete, the `table_time_budget.tex`
should show concrete median fit + search + total times per variant.
Confirm that the paper's main-text time-budget paragraph (search
`time_budget` or `time budget` in main.tex) cites real numbers, not "tbd"
placeholders. If anything still says "tbd" or "preliminary", replace with
the regenerated value.

### Step 3 — AOM-Ridge headline numbers

The AOM-Ridge headline (claim C) for Blender vs Ridge-HPO needs a final
choice on which evidence to lead with:
- Option A: lead with the full-cohort single-seed
  `bench/AOM_v0/Ridge/benchmark_runs/all54_headline/results.csv` (Blender:
  35/52 wins, median -2.22% vs tuned Ridge). The audit20 × 3-seed data
  becomes the multi-seed robustness check.
- Option B: lead with the refreshed multi-seed paired overlap (N=4 in
  v1, hopefully higher now if linear-HPO produced more Ridge-HPO rows).

Recommendation: lead with **Option A** (full-cohort single seed), and add
the multi-seed paired overlap as a supplementary cross-check. This matches
the paper's original framing and gives a clean N=52 paired denominator.

Update the main.tex AOM-Ridge headline accordingly. The abstract should
still read "Blender selector improved over tuned Ridge by a median 2.22%
with 35 wins on 52 datasets" (Option A), with a sentence in Results noting
the multi-seed cross-check.

### Step 4 — Selector / operator diagnostics

Read the new `paper_aom/review/operator_frequency.csv` and
`compact_bank_justification.md`. The top operators across the multi-seed
compact-bank variants are:

- sg_smooth_w21_p3 (23.4% of selections, 22 datasets)
- sg_d1_w21_p3 (18.1%, 24 datasets)
- sg_d1_w11_p2 (13.3%, 16 datasets)
- fd_d1 (11.1%, 14 datasets)
- detrend_d2 (10.3%, 16 datasets)

Compact-vs-default: 34/53 datasets favour compact, geometric-mean
ratio 0.9661.

If the supplement section "Compact bank justification" doesn't already
cite these numbers, add them. Reference `table_selector_diagnostics.tex`.

### Step 5 — Final PDF rebuild

```
cd paper_aom && pdflatex -interaction=nonstopmode main.tex && \
  pdflatex -interaction=nonstopmode main.tex && \
  pdflatex -interaction=nonstopmode supplement.tex && \
  pdflatex -interaction=nonstopmode supplement.tex
```

Both PDFs must build without fatal errors. Overfull-box warnings are OK.

### Step 6 — Update codex_report

Append a v2 section to `paper_aom/review/codex_report.md`:

```
## v2 update (date)
- final linear-HPO row count: ...
- new headline numbers (post linear-HPO)
- changes to main.tex / supplement.tex
- known remaining open items
```

## Constraints

- Don't kill any benchmark jobs.
- Don't commit; the user will commit.
- If a data source is empty (e.g. AOM-Ridge top5_fast has only ~88 rows),
  mark the relevant claim "supplementary" not "primary".
- Don't introduce new LaTeX packages.
- Verify all `\input{}` targets exist before adding new ones.

Report back when done.
