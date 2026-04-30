# Codex Review Workflow For BLUP

Codex should review Claude Code's work after each gate:

1. **Roadmap review** — before code starts.
2. **Math review** — after `decomposition.py` and `estimator.py`.
3. **Code review** — after `diagnostics.py` is implemented.
4. **Test review** — before benchmark runs.
5. **Benchmark review** — after smoke benchmark.
6. **Publication review** — after manuscript draft.

Suggested command:

```bash
codex exec --skip-git-repo-check \
  --output-last-message /tmp/blup_codex_math.md \
  "$(cat bench/aom_v0/Multi-kernel/Blup/docs/codex_review_prompts/math_review.md)" \
  </dev/null
```

After every review:
- summarise findings in `docs/IMPLEMENTATION_LOG.md`;
- fix high-severity findings before continuing;
- fix or explicitly defer medium findings;
- record tests run.

Codex should verify:
- BLUP formula `u_b = sigma_b^2 K_b alpha_dual` is correctly applied;
- `alpha_dual = V^-1 (y - X_f beta)` is precomputed at fit time;
- decomposition sum identity `sum_b u_b + X_f beta == predict` to fp tolerance;
- `predict_components` keys match block names from operator bank;
- no leakage of training info into test-time predictions beyond the
  fitted `alpha_dual` and `sigma_b^2`.
