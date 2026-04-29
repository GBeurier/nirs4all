# Claude Code Driver Prompt: AOM-Ridge

```text
You are working in the repository at /home/delete/nirs4all/nirs4all.

Role:
- Claude Code pilots the AOM-Ridge implementation.
- Codex acts as an independent reviewer after each phase.
- Do not advance past a phase gate until tests pass and Codex high-severity
  findings are fixed.

Scope:
- Work only under bench/AOM_v0/Ridge for new Ridge code, tests, benchmarks, and
  docs.
- You may import and read bench/AOM_v0/aompls operators and bank presets.
- Do not modify production nirs4all.
- Do not refactor bench/AOM_v0/aompls unless explicitly instructed.

Read first:
- bench/AOM_v0/Ridge/README.md
- bench/AOM_v0/Ridge/docs/PLAN_REVIEW_CORRECTIONS.md
- bench/AOM_v0/Ridge/docs/AOM_RIDGE_MATH_SPEC.md
- bench/AOM_v0/Ridge/docs/IMPLEMENTATION_PLAN.md
- bench/AOM_v0/Ridge/docs/AOM_RIDGE_API.md
- bench/AOM_v0/Ridge/docs/TEST_PLAN.md
- bench/AOM_v0/docs/AOMPLS_MATH_SPEC.md
- bench/AOM_v0/aompls/operators.py
- bench/AOM_v0/aompls/banks.py

Primary goal:
Implement AOM-Ridge as a self-contained sklearn-like package under:

  bench/AOM_v0/Ridge/aomridge

with tests under:

  bench/AOM_v0/Ridge/tests

Phase order:
1. Implement strict-linear kernel utilities.
2. Implement dual Ridge solvers.
3. Implement AOMRidgeRegressor(selection="superblock").
4. Implement fold-local CV for alpha selection.
5. Implement AOMRidgeRegressor(selection="global").
6. Implement active_superblock.
7. Add smoke benchmark runner.
8. Only after all above pass, consider nonlinear branch kernels.

Mathematical invariants:
- Operators act on row spectra as X_b = X A_b^T.
- Single-operator kernel:
    K_b = Xc A_b^T A_b Xc^T
- Superblock kernel:
    K = sum_b s_b^2 Xc A_b^T A_b Xc^T
- Define:
    U = sum_b s_b^2 A_b^T A_b Xc^T
    K = Xc U
    C = (K + alpha I)^-1 Yc
    beta = U C
- The estimator predicts with:
    Y_hat = (X - x_mean_) @ coef_ + y_mean_
- Therefore coef_ must have shape (p, q), never (B*p, q).

Leakage rules:
- Do not compute one globally centered kernel and slice it for CV.
- In every fold, compute x_mean, y_mean, operator fit, and block scales from
  training fold only.
- Validation spectra are centered with the training fold mean.
- Clone or freshly create operator instances for each fold and final refit.
- MSC/EMSC/SNV branch work is out of phase-1 scope.

Block scaling:
- Default block_scaling="rms":
    s_b = 1 / (||Xc A_b^T||_F / sqrt(n*p) + eps)
- Implement block_scaling="none" for diagnostics.
- Apply s_b^2 in the metric M.

Alpha grid:
- If alphas="auto":
    base = max(trace(K) / n, eps)
    alphas = base * logspace(-6, 6, 50)
- alpha must be positive.

Tests before benchmarks:
- identity-only equals sklearn Ridge;
- single operator dual equals materialized Ridge;
- superblock dual equals explicit concatenated Ridge;
- Xc @ coef_ equals K @ dual_coef_;
- CV is fold-local and does not slice a globally centered kernel;
- global selection exposes selected operator and alpha;
- active_superblock prunes duplicate operators and keeps identity;
- multi-output Y works.

Codex review loop:
After each phase, run the matching prompt from:

  bench/AOM_v0/Ridge/prompts/codex_review_prompts

Suggested command:

  codex exec --skip-git-repo-check \
    --output-last-message /tmp/aomridge_codex_review.md \
    "$(cat bench/AOM_v0/Ridge/prompts/codex_review_prompts/math_review.md)" \
    </dev/null

Append review summaries to:

  bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md

Acceptance:
- PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest bench/AOM_v0/Ridge/tests -q
  passes.
- No production files are modified.
- Strict-linear superblock Ridge matches explicit concatenated Ridge.
- CV is leakage-safe.
```

