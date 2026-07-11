No Blockers found.

## Findings

1. **Major — [n4m_engine.py:280](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:280): every `sorted_tuple` crashes during resolution.**  
   Trigger: `{"alphas": {"type": "sorted_tuple", "length": 3, "min": 0, "max": 1}}`. Native trials expose `alphas#0`, etc., not a parent `alphas`, so `trial.is_active("alphas")` fails after `ask()`, leaving the trial running. Fix: gate sorted tuples on `alphas#0` (or each element); use the parent name only for ordinary slots.

2. **Major — [n4m_engine.py:293](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:293): nested static model parameters remain flattened.**  
   Trigger: `{"cfg": {"depth": ("int", 1, 3), "mode": "fast"}}` resolves as `{"cfg": {"depth": 2}, "cfg__mode": "fast"}`. This also corrupts final `best_params` and may make every model construction fail. Fix: combine sampled and static flat dictionaries first, then unflatten once.

3. **Major — [n4m_engine.py:136](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:136): valid `approach="single"` is treated as grouped whenever folds exist.**  
   Trigger: folds supplied with `{"approach": "single"}`; no holdout split occurs. Fix: make this branch `if folds and approach == "grouped"` and validate the approach against `single/grouped/individual`.

4. **Major — [n4m_engine.py:389](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:389): mean-mode pruning can report a non-finite intermediate and abort.**  
   Trigger: `eval_mode="mean"`, a pruner enabled, first fold throws, second scores `0.8`. The latest score is finite, but the aggregate is `inf`, so `tell_intermediate(..., inf)` is called; a finite-validating native boundary throws before terminal state is reported. Fix: compute the aggregate once and call `tell_intermediate` only if that aggregate is finite.

5. **Major — [n4m_engine.py:330](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:330): post-`ask()` preprocessing failures leave trials untold and abort the run.**  
   Trigger: `_resolve()` throws, or `controller.process_hyperparameters()` rejects a sampled combination. Neither is inside the fold exception handling. Fix: catch resolution/processing failures after `ask()`, call `tell_result(..., FAILED, error=...)`, record a failed summary, and continue.

6. **Major — [n4m_engine.py:403](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:403): fold aggregation ignores maximize direction.**  
   Trigger: grouped tuning with `metric="r2"` or `"accuracy"` and default `eval_mode="best"`; scores `[0.2, 0.9]` aggregate to `0.2`, although higher is better. `robust_best` has the same problem. Fix: pass direction into `_aggregate()` and use `max(valid)` for maximize, including intermediate reports.

7. **Major — [n4m_engine.py:173](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:173): classification without a named metric optimizes legacy scores backwards.**  
   Trigger: classification with no `metric`. The controller contract returns minimize-oriented legacy values such as `-balanced_accuracy`, but the engine selects `MAXIMIZE`; `-0.70` beats `-0.95`, choosing the worse model. Fix: when `metric is None`, default to `minimize`; only infer maximize/minimize for named raw metrics.

8. **Major — [n4m_engine.py:263](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:263): `sorted_tuple` implements a lossy subset of the peer DSL.**  
   Triggers:

   - Omitted `length` should default to 3 but raises.
   - Dynamic lengths such as `("int", 3, 5)` raise.
   - `low/high` silently become `0/1`.
   - `step` and `log` are ignored.
   - `element_type="int_log"` becomes linear floating-point sampling.

   Fix: default missing length to 3, honor aliases, and compile dynamic/step/log forms as a length axis plus element axes, sorting/slicing during resolution. At minimum, reject unsupported options instead of silently changing the space.

9. **Major — [base_model.py:918](/home/delete/nirs4all/nirs4all/nirs4all/controllers/models/base_model.py:918): an unknown engine silently runs Optuna.**  
   Trigger: `{"engine": "natve"}` or `"n4m "`. This can run a different optimizer—or silently skip tuning if Optuna is unavailable. Because the key is new, unknown explicit values should error. Fix: `.strip().lower()`, explicitly branch on native aliases and `"optuna"`, otherwise raise `ValueError`.

10. **Minor — [n4m_engine.py:148](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:148): `eval_mode` and direction are not normalized or validated.**  
    Trigger: `eval_mode="means"` silently becomes best-mode behavior; `direction="MAXIMIZE"` runs native minimization while the result still reports `"MAXIMIZE"`. Fix: normalize and validate against the supported values.

11. **Minor — [n4m_engine.py:312](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:312): `seed=None` crashes.**  
    Trigger: `{"seed": None}`, accepted by the peer API, reaches `int(None)`. Fix: map `None` to native default `0`, otherwise convert to `int`.

12. **Minor — [n4m_engine.py:233](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:233): explicit range options can be reinterpreted.**  
    Trigger: `{"type":"float_log", ..., "log":False}` still enables log sampling; `step=0` is treated as omitted instead of rejected. Fix: distinguish omitted `log` from explicit `False`, and validate a supplied step as positive.

## Correct behavior

List-versus-range disambiguation, tuple inference, empty categorical rejection, and categorical reconstruction through `choices[idx]` are correct. Ordinary `is_active` handling and the `train.<name>` split are also correct; train parameters do not leak into `best_params`.

Normal evaluation paths terminate correctly: pruning returns immediately, all-nonfinite trials become `FAILED`, and finite trials use `tell()`. Best-trial reconstruction re-resolves parameters and reapplies `process_hyperparameters`.

The native manager is constructed lazily, missing `engine` preserves Optuna, unavailable `n4m` is guarded, enqueue failures are contained, integer seeds are deterministic, `n_startup_trials=10` is sensible, and Hyperband receives `max_resource=len(folds)`.

Targeted pytest could not start because the read-only environment has no writable temporary directory; the principal failures above were reproduced with in-memory test doubles.
---

## Resolution (all findings applied)

- **#1/#8 sorted_tuple** — removed native sorted_tuple support; `_add_dict` now
  raises `NotImplementedError` pointing to the Optuna engine (rather than shipping a
  lossy/crashing subset). Test `test_sorted_tuple_rejected`.
- **#2 nested static params** — `_resolve` seeds the flat dict with the (flat)
  static params and unflattens ONCE, so a nested static value lands in its group.
  Test `test_nested_static_unflatten`.
- **#3 approach routing** — `if folds and approach == "grouped"`; `single` always
  uses a holdout split even with folds. `approach` validated in `_normalize`.
- **#4 non-finite pruning** — prune only when the aggregate is finite (compute it
  once), so a failed fold under mean-mode cannot trip the native finite guard.
- **#5 untold trials** — the ask → resolve → process_hyperparameters block is wrapped;
  a failure calls `tell_result(FAILED)`, records a FAIL summary, and continues.
- **#6 direction-blind aggregation** — `_aggregate` takes `direction`; best/robust_best
  use `max` for maximize, and the failed-fold sentinel / empty-valid sentinel are
  direction-aware. Tests `test_aggregate_direction`, `test_maximize_*`.
- **#9 unknown engine** — `base_model.finetune` normalizes (`strip().lower()`), branches
  on native aliases + `optuna`, and RAISES `ValueError` on anything else.
- **#10 eval_mode/direction validation**, **#11 seed=None → 0**, **#12 explicit `log`
  wins over the type suffix + positive-step validation** — all applied. Test
  `test_seed_none_ok`.
- **#7 classification default direction** — intentionally kept IDENTICAL to
  `OptunaManager._resolve_metric_direction` (the drop-in must pick the same model as
  Optuna for the same config; diverging would break parity — if the shared controller
  contract is off, that is a separate cross-engine issue).

Plus the new **`when`/`when_not` conditional clause** (object__attribute conditional
finetuning), compiled into native conditional-activation constraints. Test
`test_conditional_when_clause`. 22 engine unit tests pass; 218 finetune tests green.
