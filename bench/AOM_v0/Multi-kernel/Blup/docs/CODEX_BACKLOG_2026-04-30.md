# BLUP Codex Backlog — Round 2 (2026-04-30)

Source: `/tmp/codex_blup_math.md` (Codex math/code review).

## Verified OK

- E-BLUP mean prediction uses cached `alpha_dual_`.
- Random block contributions = `sigma_b^2 * K_b_cross @ alpha_dual_`.
- No predict-time `V` re-factorisation.
- Sklearn `clone`/`fit`/`predict`/`score` smoke tests pass.
- `pytest -q tests/` → 10 passed.

## High-severity finding (applied)

### H1 — Duplicate block names break the decomposition identity

**Bug**: Random components stored in `OrderedDict` keyed only by
`block_name`. With duplicate names (custom bank with two "dup" operators),
`random[name] = ...` overwrites the earlier entry; the "total" reconstructed
from the dict misses one contribution.

**Reproducer (Codex)**: with two operators renamed `"dup"`, `B_=3`,
`len(random)=2`, `max abs(components["total"] - predict) ≈ 2.96`. Identity
violated.

**Fix applied** (`blup/estimator.py:148`):
1. Accumulate `total` **inside the loop**, independently of dict keys.
2. Disambiguate dict keys with `__k` suffix when `block_name` repeats:
   `dup`, `dup__2`, `dup__3`, …

This guarantees the decomposition identity even with duplicate names, and
keeps a stable user-facing key naming.

## Medium-severity backlog

### M1 — Test coverage: held-out vs random unseen

Existing tests cover training-data identity (`test_decomposition_sum_equals_predict_train`)
and one random unseen matrix (`test_decomposition_sum_equals_predict_test`).
Add an additional held-out split from `_smooth_X_y` to differentiate
"random unseen" from "held-out under same data-generation".

### M2 — Duplicate block name regression test

Add a unit test using a custom bank with two operators sharing the same
`name` attribute and assert that
`predict_components(X)["total"] ≈ predict(X)` to fp tolerance.

### M3 — Shared `_predict_parts(X, X_fixed_test)`

`predict` delegates to MKM, while `predict_components` rebuilds the
intercept locally. Extract a single `_predict_parts(X, X_fixed_test=None)`
helper used by both. Important when non-intercept fixed effects are added
in a future round.
