"""Controller for the residual model-combination operator (``ResidualModel`` / ``{"residual": {...}}``).

Fits a base model (leakage-safe OOF), trains a learner on the OOF residuals, and stores the combined
prediction ``base + lambda*gate*learner``. Both sub-models are routed through their own framework
controllers, so the learner may be any nirs4all model (sklearn, PyTorch ``nicon``, TabPFN, ...).

The two sub-models are fit by delegating to the standard model controllers via the router; the residual
target is delivered to the learner through a numeric-rooted processed-target series (``add_processed_targets``
+ ``context.with_y``), so no learner-side change is needed.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.operators.models.residual import ResidualModel


@register_controller
class ResidualModelController(OperatorController):
    """Fit base + residual-learner combination. Priority 5 (above framework model controllers)."""

    priority = 5

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        if isinstance(operator, ResidualModel):
            return True
        if keyword == "residual":
            return True
        return isinstance(step, dict) and isinstance(step.get("model"), ResidualModel)

    @classmethod
    def use_multi_source(cls) -> bool:
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    # ----- helpers -------------------------------------------------------------------------------
    def _resolve(self, operator: Any, keyword: Any) -> ResidualModel:
        if isinstance(operator, ResidualModel):
            return operator
        if isinstance(operator, dict):
            model = operator.get("model")
            if isinstance(model, ResidualModel):
                return model
            if keyword == "residual" or {"base", "learner"} <= set(operator):
                return ResidualModel(**operator)
        raise ValueError(f"ResidualModelController: could not resolve a ResidualModel "
                         f"(keyword={keyword!r}, operator={type(operator).__name__}).")

    def _fit_submodel(self, model_op, dataset, context, runtime_context, source, mode,
                      prediction_store, train_params=None, finetune_space=None):
        """Fit one sub-model by routing a synthetic ``{"model": ...}`` step through its framework controller.

        Returns the model_name under which its predictions were stored.
        """
        from nirs4all.pipeline.steps.parser import StepParser
        from nirs4all.pipeline.steps.router import ControllerRouter

        step = {"model": model_op}
        if train_params:
            step["train_params"] = train_params
        if finetune_space:
            step["finetune_params"] = finetune_space
        parsed = StepParser().parse(step)
        controller = ControllerRouter().route(parsed, step)
        if isinstance(controller, type):
            controller = controller()
        before = set(self._all_model_names(prediction_store))
        controller.execute(parsed, dataset, context, runtime_context,
                            source=source, mode=mode, prediction_store=prediction_store)
        after = self._all_model_names(prediction_store)
        new = [n for n in after if n not in before]
        return new[-1] if new else (after[-1] if after else None)

    @staticmethod
    def _all_model_names(prediction_store) -> list[str]:
        try:
            preds = prediction_store.filter_predictions(load_arrays=False)
            return [p.get("model_name") for p in preds]
        except Exception:  # noqa: BLE001
            return []

    def _vec(self, prediction_store, model_name, partition, fold_id=None):
        """Sample-indexed (idx, pred, y_true) for one model+partition.

        ``fold_id=None`` averages all *numeric* folds (the OOF set, for val). ``fold_id='final'`` selects
        the refit (full-train) prediction — used for the test partition so the combination is composed from
        the same refit predictions the baselines are scored on. Robust to fold layout."""
        from collections import defaultdict
        def _keep(r):
            if r.get("partition") != partition or r.get("y_pred") is None:
                return False
            fid = str(r.get("fold_id"))
            if fold_id is not None:
                return fid == str(fold_id)
            return fid not in ("avg", "w_avg", "final")  # numeric folds only = OOF
        rows = [r for r in prediction_store.filter_predictions(model_name=model_name, load_arrays=True)
                if _keep(r)]
        psum: dict = defaultdict(float)
        pcnt: dict = defaultdict(int)
        ytrue: dict = {}
        for r in rows:
            si = np.asarray(r.get("sample_indices")).ravel()
            yp = np.asarray(r["y_pred"], float).ravel()
            yy = np.asarray(r.get("y_true"), float).ravel() if r.get("y_true") is not None else None
            for k in range(min(len(si), len(yp))):
                i = int(si[k])
                psum[i] += float(yp[k])
                pcnt[i] += 1
                if yy is not None and k < len(yy):
                    ytrue[i] = float(yy[k])
        if not psum:
            return np.array([]), np.array([]), np.array([])
        idx = np.array(sorted(psum))
        pred = np.array([psum[i] / pcnt[i] for i in idx])
        y = np.array([ytrue.get(i, np.nan) for i in idx])
        return idx, pred, y

    @staticmethod
    def _resolve_gate(gate, r_oof, learner_oof, rli_threshold):
        """Return the effective scalar g (and rli) per the gate spec, estimated on OOF only."""
        denom = float(np.dot(learner_oof, learner_oof))
        sd = float(np.std(r_oof)) or 1.0
        rmsecv = float(np.sqrt(np.mean((r_oof - learner_oof) ** 2)))
        rli = 1.0 - rmsecv / sd
        if gate is False:
            return 1.0, rli
        if isinstance(gate, (int, float)) and not isinstance(gate, bool):
            return float(gate), rli
        g = float(np.dot(r_oof, learner_oof) / denom) if denom > 1e-12 else 0.0
        g = max(0.0, min(1.0, g))
        if rli <= rli_threshold:
            g = 0.0
        return g, rli

    @staticmethod
    def _rmse(a, b):
        try:
            return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
        except Exception:  # noqa: BLE001
            return None

    # ----- main ----------------------------------------------------------------------------------
    def execute(self, step_info, dataset, context, runtime_context, source: int = -1,
                mode: str = "train", loaded_binaries=None, prediction_store=None):
        op = self._resolve(getattr(step_info, "operator", None), getattr(step_info, "keyword", None))
        y_full = np.asarray(dataset.y({"y": "numeric"}), float).ravel()

        # 1) Fit BASE via the router (full fold loop) -> read its leakage-safe OOF/test from the store.
        base_name = self._fit_submodel(op.base, dataset, context, runtime_context, source, mode, prediction_store)
        v_idx, base_oof, y_tr = self._vec(prediction_store, base_name, "val")   # OOF (gate/lambda)
        t_idx, base_test, y_te = self._vec(prediction_store, base_name, "test")  # CV-ensemble test (consistent w/ learner)

        # 2) Residual target (numeric space): full-length, residual on the OOF train rows.
        r_oof = y_tr - base_oof
        residual_full = y_full.copy()
        if len(v_idx):
            residual_full[v_idx] = r_oof
        res_proc = f"residual__{base_name}_{id(op) & 0xffff:x}"
        from sklearn.preprocessing import FunctionTransformer
        identity = FunctionTransformer(validate=False).fit(residual_full.reshape(-1, 1))  # numeric-space => identity inverse
        dataset.add_processed_targets(res_proc, residual_full, ancestor_processing="numeric", transformer=identity)
        residual_context = context.with_y(res_proc)

        # 3) Fit LEARNER on residuals via the router (ANY framework: sklearn/nicon/tabpfn/xgb...).
        learner_name = self._fit_submodel(op.learner, dataset, residual_context, runtime_context, source, mode,
                                           prediction_store, op.train_params, op.finetune_space)
        _, learner_oof, _ = self._vec(prediction_store, learner_name, "val")
        _, learner_test, _ = self._vec(prediction_store, learner_name, "test")

        # 4) Gate + compose base + lambda*g*learner (align defensively).
        loof = learner_oof if len(learner_oof) == len(base_oof) else np.zeros_like(base_oof)
        ltest = learner_test if len(learner_test) == len(base_test) else np.zeros_like(base_test)
        g, rli = self._resolve_gate(op.gate, r_oof, loof, op.rli_threshold)
        comb_val = base_oof + op.lam * g * loof
        comb_test = base_test + op.lam * g * ltest

        # 5) Store the combined prediction so scoring/top(n) ranks this combination. Record base- and
        #    hybrid-test RMSE (both CV-ensemble, hence consistent) so analysis computes a clean gain.
        meta = {"lambda": op.lam, "gate": float(g), "rli": float(rli), "base": base_name, "learner": learner_name,
                "base_test_rmse": self._rmse(y_te, base_test), "hybrid_test_rmse": self._rmse(y_te, comb_test)}
        # Name from the fitted sub-model names so _cartesian_/_or_ variants stay distinct in the store.
        combined_name = op._name or f"Residual({base_name}+{learner_name})"
        self._store(prediction_store, dataset, combined_name, v_idx, comb_val, y_tr, t_idx, comb_test, y_te, meta)
        return context, {"residual": meta}

    def _store(self, store, dataset, model_name, v_idx, val, y_val, t_idx, test, y_test, meta):
        common = {"dataset_name": getattr(dataset, "name", "dataset"), "model_name": model_name,
                  "model_classname": "ResidualModel", "metadata": meta, "best_params": meta,
                  "task_type": "regression", "metric": "rmse"}
        try:
            store.add_prediction(partition="val", fold_id="final", sample_indices=list(map(int, v_idx)),
                                 y_true=np.asarray(y_val), y_pred=np.asarray(val),
                                 val_score=self._rmse(y_val, val), **common)
            store.add_prediction(partition="test", fold_id="final", sample_indices=list(map(int, t_idx)),
                                 y_true=np.asarray(y_test), y_pred=np.asarray(test),
                                 test_score=self._rmse(y_test, test), val_score=self._rmse(y_test, test), **common)
        except Exception as exc:  # noqa: BLE001 - surfaced during integration testing
            warnings.warn(f"ResidualModelController could not store combined predictions: {exc}", stacklevel=2)
