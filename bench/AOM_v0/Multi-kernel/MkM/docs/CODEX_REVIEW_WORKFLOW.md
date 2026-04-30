# Codex Review Workflow For MKM

Codex should review Claude Code's work after each gate:

1. **Roadmap review** — before code starts (this file + IMPLEMENTATION_PLAN +
   MKM_MATH_SPEC).
2. **Math review** — after `kernelizer.py`, `likelihood.py` are implemented.
3. **Code review** — after `optimisation.py`, `estimator.py` are implemented.
4. **Test review** — before benchmark runs.
5. **Benchmark review** — after smoke benchmark.
6. **Publication review** — after manuscript draft.

Suggested command from repository root:

```bash
codex exec --skip-git-repo-check \
  --output-last-message /tmp/mkm_codex_math.md \
  "$(cat bench/aom_v0/Multi-kernel/MkM/docs/codex_review_prompts/math_review.md)" \
  </dev/null
```

After every review:
- summarise findings in `docs/IMPLEMENTATION_LOG.md`;
- fix high-severity findings before continuing;
- fix or explicitly defer medium findings (track in `docs/CODEX_BACKLOG_<date>.md`);
- record tests run.

Codex should verify:
- centred + trace-normalised kernels match the MKM_MATH_SPEC;
- REML formula (logdet `V` + logdet `X^T V^-1 X` + GLS quadratic) matches the spec;
- gradient analytic vs finite-difference agreement (`< 1e-4`);
- L-BFGS-B uses log-variances (not raw variances);
- multi-restart logic detects multimodality;
- no global centring or kernel reuse leaks into CV;
- `predict` agrees with mkR `predict` at fixed `(sigma_b^2, sigma_e^2)`.
