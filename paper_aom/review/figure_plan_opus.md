# Figure Plan From Opus Review

Scope: scientific figure guidance for the AOM/NIRS manuscript. The Opus session
could not write to the final repository path, so this file records the usable
design recommendations incorporated or deferred by Codex.

## Priority Figures

1. **Conceptual AOM schematic.** Contrast external preprocessing grid search
   with an operator bank selected inside the model. This is implemented as
   `figures/fig_concept.pdf`.
2. **AOM algebra schematic.** Show the same operator bank feeding PLS
   covariance selection and Ridge dual geometry. This is implemented as
   `figures/fig_math.pdf`.
3. **Benchmark-effect summary.** Show median error reductions and win counts,
   while clearly separating deployable Ridge variants from oracle-envelope
   numbers. This is implemented as `figures/fig_results.pdf`.
4. **Search-budget contrast.** Show that the reference PLS/Ridge protocols are
   selected outcomes of large preprocessing-HPO searches. This is implemented
   as `figures/fig_budget.pdf`.

## Visual Conventions

- Use restrained colors suitable for chemometrics journals: blue for PLS,
  green for Ridge, teal for AOM, muted orange for CatBoost, purple for CNN.
- Keep figures readable in grayscale through labels and direct annotation, not
  color alone.
- Avoid decorative gradients, shadows, 3D effects, and dense marketing-style
  visuals.
- Use vector PDF output with embedded fonts where possible.

## Deferred Figure Improvements

- Replace schematic result bars with forest plots once a single final cohort is
  frozen.
- Add operator-frequency plots once selection logs are regenerated.
- Add representative coefficient/weight plots from real datasets after final
  prediction artifacts are exported.
- Add single-column and double-column figure variants for the target journal.
