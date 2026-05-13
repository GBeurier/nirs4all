# Codex finalization report

Date: 2026-05-13.

## Step 1 validation

| Item | Status | Finding |
| --- | --- | --- |
| Cohort manifest CSV | ⚠️ | Exists with 78 rows and required columns. It has 61 regression + 17 classification rows, but `status_in_primary_analysis=include` is 61 regression + 16 classification because `Quartz_spxy70` remains `include` with `exclusion_reason=denominator_near_zero_pairwise`; requested primary count was 60 + 16. |
| Cohort manifest MD | ⚠️ | Exists with denominator rules, but documents 61 included regression rows / paired AOM-PLS vs PLS denominator 61, not the requested 60 primary-regression denominator. |
| Claim ledger | ✅ | Claims A-E and evidence paths are present; statuses remain blocked/provisional. |
| Aggregator | ✅ | `python paper_aom/review/aggregate_stats.py --partial` completed; Step 2 refresh loaded 10,749 rows from 8 workspaces. |
| Selector diagnostics | ✅ | Regenerated CSV/Markdown/TeX outputs. AOM-Ridge diagnostics parsed 0 rows from the current partial top5_fast file. |
| AOM-lib wrapper | ✅ | `ruff`, `mypy`, 7 unit tests, and the nirs4all example all passed. |
| Software table | ✅ | 10 data rows plus status legend. |
| AOM_lib C/Python parity | ✅ | `c_smoke`, `test_operators`, `test_parity_kfold`, and Python parity tests passed; `c_smoke.c` includes `<math.h>`. |
| Linear HPO runner | ✅ | `bench/tabpfn_paper/run_linear_hpo_paper_aom.py` exists. |
| Result workspaces | ✅ | All six listed workspaces/files exist. Final row counts had grown while background jobs continued: AOM-PLS 1485 data rows, AOM-Ridge top5_fast 76, partial headline 11, AOM-PLS-DA 232, AOM-Ridge classification 39, linear-HPO 94. |

## Headline numbers

- AOM-PLS paper provisional: median RMSEP/PLS `0.960`, `42/57` wins.
- Refreshed AOM-PLS multi-seed robustness: median RMSEP/PLS `0.9623`, `37/53` wins after averaging RMSEP across seeds.
- `final_stats.md` partial paired overlap for ASLS-AOM vs PLS-default: ratio `0.964`, `6/7` wins.
- AOM-Ridge paper provisional: `2.22%` median improvement, `35/52` wins vs tuned Ridge.
- Refreshed `final_stats.md` AOM-Ridge overlap: Blender vs Ridge-HPO ratio `0.905`, `4/4` wins; denominator is still partial and not a full-cohort promotion.

## Files edited

- `paper_aom/main.tex`
- `paper_aom/supplement.tex`
- `paper_aom/tables/table_main_results.tex` (escaped `paper_aom` in generated table source so LaTeX builds)
- `paper_aom/review/aggregate_stats.py` (escapes plain-text LaTeX table cells so the same issue does not recur on refresh)

Aggregation and diagnostic commands also refreshed generated review/table artifacts under `paper_aom/review/` and `paper_aom/tables/`.

## PDF build

- `paper_aom/main.pdf`: built without fatal errors after two clean final runs.
- `paper_aom/supplement.pdf`: built without fatal errors; one extra run was used to settle references.
- Remaining warnings: overfull boxes from long code paths in `table_software.tex` and one long result path in the main text; several supplement `[h]` floats changed to `ht`. No unresolved references remained in the final logs.

## Open items

- Linear-HPO timing run is still incomplete; refresh aggregation/table once finished.
- Full AOM-Ridge Blender/AutoSelect headline rerun remains pending; current full-cohort run is top5_fast only.
- Add Talanta Novelty Statement and AI-assisted-technology declaration before submission.
- Resolve the manifest/documentation mismatch for QUARTZ if primary analyses must report 60 regression includes rather than 61 absolute-error rows.
