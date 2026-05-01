# DESIGN_MOE — AOM-MoE algorithms (Phase 3)

**Status**: design draft, awaiting Codex review of math.
**Phase**: 3 (after AOM-MBPLS Phase 2).
**Prereqs**: Phase 1 ViewBuilder, Phase 2 block-sparse / classic MB-PLS-AOM.

---

## 1. Goal

Test whether **mixture-of-experts (MoE)** routing — multiple specialised PLS
models combined by a learned gate — beats the single-PLS-model approach of
Phases 1-2. Three routing styles × two expert layouts = six base variants.

Per the user constraint, Phase 3 starts with **PLS-only experts**. Phase 4
allows heterogeneous experts (AOM-PLS, AOM-Ridge, AOM-MkM) when justified by
Phase 3 signal. Ridge / multi-kernel parallel sessions own those heterogeneous
estimators; we consume their best variants without re-implementing.

---

## 2. Six base variants

| # | Routing | Experts                | Gate                              |
|---|---------|------------------------|-----------------------------------|
| 1 | Hard MoE         | per-view-PLS (K)         | Argmax of gate scores → 1-hot     |
| 2 | Soft MoE         | per-view-PLS (K)         | Softmax-weighted average           |
| 3 | AOM-per-LV MoE   | per-view-PLS (K)         | Per-LV PRESS routing within each expert |
| 4 | Hard MoE         | per-preproc-PLS (≈9)     | Argmax                             |
| 5 | Soft MoE         | per-preproc-PLS (≈9)     | Softmax-weighted                   |
| 6 | AOM-per-LV MoE   | per-preproc-PLS (≈9)     | Per-LV PRESS routing               |

Notation:

- "per-view" = one PLS expert per wavelength block (K=3 default).
- "per-preproc" = one PLS expert per operator in the `compact` bank (8
  non-identity ops + identity → 9 experts).

---

## 3. Algorithm specifications

### 3.1 Hard / soft MoE with PLS experts

```
# Training
for each expert e:
    fit PLS(n_components=k_e) on the e-th view of training data
gate = train_gate(predictions of all experts on a held-out fold, y_val)
    # gate: (predictions matrix) → weights vector

# Prediction
pred_e = expert_e.predict(X_view_e)         # (n_test, q)
gate_weights = gate.weights(X_test or other features)
y_pred = Σ_e gate_weights[e] · pred_e        # soft
y_pred = pred_{argmax(gate_weights)}         # hard
```

### 3.2 Gate design

**v1 — constant gate (no per-sample routing)**: train weights once on a
held-out fold using non-negative least squares (NNLS) to minimise OOF MSE
of the mixture. Same gate weights for every test sample.

**v2 — per-sample gate (Phase 3.5)**: train a small classifier (logistic
regression or 1-LV PLS) on the held-out fold predicting which expert
performs best per sample. Used only if v1 plateau-wins across smoke-4.

For Phase 3 we ship **v1**; v2 is a stub.

### 3.3 AOM-per-LV MoE

This is conceptually the same as POP-PLS on a multi-view bank, where each
"expert" is the LV-1 weight of one view's PLS. Per LV `a`:

1. Each expert e has a current "next weight" `w_{e,a}` (the LV-`a` weight
   from PLS on view e).
2. Score each expert's `t_{e,a}` against `y_res` via PRESS.
3. Pick winner e*; commit `t_a = t_{e*,a}`, deflate.

This is **Phase 2 classic MB-PLS-AOM with per-LV super-score = winner's
score** — i.e. block-sparse if K=number of blocks. Hence variant 3 is
mathematically identical to Phase 2 block-sparse V1, and variant 6 is
identical to Phase 2 block-sparse V2. They are excluded from Phase 3 (no
new algorithm) and the comparison row is reused from Phase 2 results.

Net new code for Phase 3 = variants 1, 2, 4, 5 (hard/soft MoE).

---

## 4. Implementation plan

```
bench/AOM_v0/multiview/multiview/moe.py
    AOMMoERegressor              # sklearn-compat
        - expert_layout: "per_view" | "per_preproc"
        - routing: "hard" | "soft" | "aom_per_lv"
        - K: int (per_view only)
        - bank_name: str (per_preproc only)
        - per_expert_components: int (default 10)
        - cv_splitter: optional
    _train_constant_gate(predictions: (n, K, q), y: (n, q)) -> weights
        # NNLS or simplex projection

bench/AOM_v0/multiview/tests/test_moe.py
    - K=1 reduces to single PLS
    - hard MoE picks the best single expert when one is dominant
    - soft MoE matches hard within tolerance when one expert dominates
    - per_preproc expert wiring uses the right bank

bench/AOM_v0/multiview/benchmarks/run_smoke4_phase3_moe.py
    - Append Phase 3 variants to smoke-4 CSV.
    - Variants: moe-{hard,soft}-{view,preproc}-holdout.
```

For each smoke-4 dataset, the workflow is:

1. **Train experts** with hyperparameters fixed by Phase 2 winners (e.g. if
   block-sparse V1 picks `n_components_per_block ≈ 3`, use 3 for per-view
   experts).
2. **Train gate** on a held-out fold of the training data using NNLS.
3. **Predict** on the test set using the gate-weighted (or argmax) mixture.

---

## 5. Stop / decision criteria

| Outcome on smoke-4 | Action |
|---------------------|--------|
| Hard or soft MoE wins ≥ 2/4 vs Phase 2 block-sparse winner | Escalate to smoke-10 |
| All MoE variants tie or lose to Phase 2 winner | Document negative result; skip Phase 3.5 (per-sample gate) |
| Per-preproc beats per-view consistently | Investigate why preproc views complement each other better than block views |
| One specific dataset has clear MoE advantage | Investigate the structure (e.g. multi-modal noise, multi-source) |

---

## 6. Codex review focus

1. **§3.2 v1 constant gate** — NNLS on OOF predictions vs unconstrained LS:
   does the non-negativity constraint matter for fair mixture, or should
   we allow negative weights (which would amount to a meta-Ridge with a
   Lasso prior)?

2. **§3.3 AOM-per-LV ≡ block-sparse?** — verify this claim: AOM-per-LV with
   PLS experts per view and per-LV PRESS routing reduces to selecting one
   block at each LV with global super-score. Is this exactly what Phase 2
   block-sparse does, modulo deflation strategy?

3. **§3.1 expert hyperparameters** — should each expert get its own
   `n_components` (cv-tuned per expert), or share the same `n_components`
   from a global search? Per-expert tuning is more expressive but more CV
   cost.

4. **OOF data for gate training** — if the gate is fit on held-out fold
   predictions, we lose those samples for expert training. Cross-fitting
   (CV-OOF for gate input) is the correct approach but adds significant
   compute. Acceptable for smoke-4, prohibitive for full-57?

5. **Multi-target generalisation** — the MoE math here assumes q=1; for
   q>1 the gate weights become per-target. Phase 3 ships q=1 only.
