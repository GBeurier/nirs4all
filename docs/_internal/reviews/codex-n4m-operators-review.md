No Blockers.

- **Major — [n4m_engine.py:365](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:365): flat-head detection depends on trial activity.** If conditional `est` is inactive while static or differently conditioned `est__alpha` remains active, resolution produces `{"est": {"alpha": ...}}` instead of flat `{"est__alpha": ...}`; `Pipeline.set_params` then replaces `est` with a dict. Compute flat heads from all declared model slots/static keys, not resolved active values.

- **Major — [n4m_engine.py:319](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:319): option names are not validated before stringification.** `{1: PLS(), "1": Ridge()}` creates duplicate native label `"1"`; `when: {"est": 1}` matches both operators and leaks conditional parameters. An empty-string name cannot be conditioned on, and tuple names fail naturally because [line 254](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:254) expands `(1, 2)` into two labels. Require non-empty, NUL-free string keys, or fully validate canonical-label uniqueness and tuple semantics.

- **Minor — [n4m_engine.py:504](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:504), [line 510](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:510): `options` is not recognized as a parameter-spec marker.** `{"est": {"options": {...}}}`—consistent with `_add_dict`’s default categorical type—is flattened as static data instead of sampled. Add `"options"` to both recognizers, or require explicit `"type": "categorical"`.

- **Minor — [n4m_engine.py:254](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:254): malformed clauses are incompletely guarded.** `when: {"est": []}` emits zero constraints, making the child unconditional; `options=None` raises an uncontextual `TypeError` at line 316. Reject empty `when` label sets and require `options` to be a mapping.

Correct: explicit empty mappings are rejected; named parents are native STRING categoricals; matching int/float/bool/`None` values use the same `str()` representation; native indices are range-clamped and preserve `choices[idx]` alignment. With a present standalone head, flat addressing—including three-level keys—is order-independent. Properly conditioned inactive children are skipped in trials and `best_params`, so no stale parameter remains.
---
## Resolution (all findings applied)
- Major (flat-head from activity): `flat_heads` is now computed from the DECLARED
  slots/static keys in `_optimize` and threaded into `_resolve`, so an inactive
  operator choice still keeps its `op__param` addressing flat.
- Major (option-name validation): `options` names must be non-empty NUL-free
  strings (they ARE the native labels `when` matches); non-string/duplicate names
  are rejected. Test `test_bad_option_names_raise`.
- Minor (`options` marker): added to `_is_param_spec` + `_is_sampable`, so
  `{"options": {...}}` without an explicit `type` is sampled, not treated as static.
  Test `test_options_without_explicit_type`.
- Minor (empty-`when` / non-mapping `options`): both rejected with a clear error.
  Test `test_empty_when_raises`.
26 engine unit tests pass; operator search runs end-to-end.
