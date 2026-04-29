# Codex Review Workflow For AOM-Ridge

Codex should review Claude Code's work after each gate:

1. Roadmap review before code starts.
2. Math review after kernel/solver implementation.
3. Code review after estimator/selection implementation.
4. Test review before benchmark runs.
5. Benchmark review after smoke benchmark.

Suggested command from repository root:

```bash
codex exec --skip-git-repo-check \
  --output-last-message /tmp/aomridge_codex_review.md \
  "$(cat bench/AOM_v0/Ridge/prompts/codex_review_prompts/math_review.md)" \
  </dev/null
```

After every review:

- summarize findings in `docs/IMPLEMENTATION_LOG.md`;
- fix high-severity findings before continuing;
- fix or explicitly defer medium findings;
- record tests run.

Codex should verify:

- no global centering or global block scaling inside CV;
- strict-linear formulas match explicit superblock Ridge;
- `coef_` is original-space;
- nonlinear branches are not presented as coefficient models;
- benchmark labels distinguish strict-linear, branch, and OOF expert models.

