**Findings**

1. **Train/test leakage:** outer split looks correct. `kernel_top_k_active` is passed into `AOMKernelizer`, and screening happens inside `fit(X_train, y_train)` only; test data is only used at prediction time. See [mkr_estimator.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/MKR/aomridge/mkr_estimator.py:248) and [kernelizer.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/MKR/aomridge/kernelizer.py:260). Caveat: the inner CV used by `softmax_cv` sees kernels whose active set was chosen using all outer-training `y`, so inner-CV RMSE is optimistic. Test-set results remain valid.

2. **Important mismatch:** Iter 1 did **not** use KTA by default. The runner defaults `screen_score_method` to `"norm"` despite comments saying active screening is KTA-based. See [run_multikernel_smoke.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py:180) and [selection.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Multi-kernel/MKR/aomridge/selection.py:512). Recommendation: relabel Iter 1 as `norm` screening, or rerun `active15-kta` / `active15-blend`.

3. **Screen before trace normalisation:** mathematically acceptable here. With centered `X`, linear-operator kernels are already centered up to numerical noise, and trace scaling does not change KTA ranking. For `"norm"`, the RMS block scaling makes the score proportional to `y.T @ K_trace_norm @ y`, so the pre-materialization screen is aligned with final trace-normalized kernels. This would need revisiting for non-linear/data-adaptive operators.

4. **Small-n fallback:** do **not** fall back to compact solely because `n` is small. BEER `n=40` and TIC `n=43` are the strongest active15 gains. The better fallback rule is: keep `Ridge-raw` and compact `MKM-reml-asls` in the candidate set, especially for low-signal/simple datasets like An_NeoSpectra and ASLS-sensitive targets.

5. **MANURE_Total_N regression:** likely not a general active-screen failure. mkR and MKM-reml improved there, while `MKM-reml-asls` regressed from `0.868` to `0.884` rel-PLS. The active ASLS set was derivative/detrend-composition heavy and the REML fit still had many boundary components, so this looks like variance-component identifiability/over-selection after ASLS, not lack of capacity. Keep compact `MKM-reml-asls` as a challenger.

**Iter 2 Recommendation**

Prioritize **tighter hyperparams on active15** over active30/active50: `n_restarts=5`, `max_iter=40`, larger alpha grid/CV for mkR; more REML restarts for MKM. Going larger is less promising because `kernel_alignment_max` is already near 1.0 on most datasets, so active30/50 mostly adds collinear blocks and selection variance.

Run a small ablation only after that:

- `active15-norm-tuned`
- `active15-kta-tuned`
- `active15-blend-tuned`
- optionally `active30-fast` as a capacity check

Also log `block_names_`, screen scores, and actual selected `B`; “active15” is a cap, not always exactly 15 after diversity pruning.