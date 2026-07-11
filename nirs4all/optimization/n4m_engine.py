# SPDX-License-Identifier: MIT
"""Native (``libn4m`` / ``n4m``) hyperparameter finetuning engine.

A drop-in peer of :class:`~nirs4all.optimization.optuna.OptunaManager` that runs
the **native ask/tell optimizer** shipped in ``nirs4all-methods`` instead of
Optuna. Selected per model step via ``finetune_params={"engine": "n4m", ...}``.

Why: the native engine (samplers ``random / sobol / lhs / ternary / ga / pso /
cmaes / tpe / gp_ei`` and pruners ``median / asha / hyperband / racing``) is a
portable C-ABI core, so the *same* optimizer drives finetuning in Python, R,
MATLAB/Octave and WASM — and at a fixed seed the trial sequence is identical
across bindings. Optuna stays the default; nothing here removes it.

The search-space DSL is **identical** to the Optuna path (``finetune_params
["model_params"]`` / ``["train_params"]`` with tuple ranges, choice lists, dict
specs and ``__``-nested groups), compiled here into a native ``SearchSpace``. The
eval loop reuses the controller hooks (``_get_model_instance`` /``_prepare_data``
/``_train_model`` /``_evaluate_model`` /``process_hyperparameters``) exactly as the
Optuna manager does, so the refit path is unchanged: it returns the same
:class:`FinetuneResult` objects.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from nirs4all.core.logging import get_logger

# Reuse the exact result types the controller refit path expects.
from nirs4all.optimization.optuna import FinetuneResult, TrialSummary

logger = get_logger(__name__)

try:
    from n4m.model_selection.optimizer import (
        Direction,
        Optimizer,
        Pruner,
        Sampler,
        SearchSpace,
        TrialStatus,
    )

    N4M_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    N4M_AVAILABLE = False


# ---- DSL type tables (mirror OptunaManager) --------------------------------
_INT_TYPES = ("int", int, "builtins.int")
_INT_LOG_TYPES = ("int_log", "log_int")
_FLOAT_TYPES = ("float", float, "builtins.float")
_FLOAT_LOG_TYPES = ("float_log", "log_float")
_ALL_RANGE_TYPES = _INT_TYPES + _INT_LOG_TYPES + _FLOAT_TYPES + _FLOAT_LOG_TYPES

# nirs4all finetune_params sampler/pruner names -> native enum.
_SAMPLER_MAP = {
    "auto": "tpe", "grid": "random", "binary": "ternary", "sample": "tpe",
    "random": "random", "sobol": "sobol", "lhs": "lhs", "ternary": "ternary",
    "ga": "ga", "pso": "pso", "cmaes": "cmaes", "tpe": "tpe", "gp_ei": "gp_ei",
}
_PRUNER_MAP = {
    "none": "none", "median": "median", "successive_halving": "asha",
    "asha": "asha", "hyperband": "hyperband", "racing": "racing",
}


class _Slot:
    """One sampled search-space axis (native name + how to read it back)."""

    __slots__ = ("native", "origin_name", "kind", "choices", "length", "elem_int", "is_train")

    def __init__(self, native, origin_name, kind, *, choices=None, length=0,
                 elem_int=False, is_train=False):
        self.native = native            # param name in the native SearchSpace
        self.origin_name = origin_name  # flat model name, or train param name
        self.kind = kind                # 'int' | 'float' | 'categorical' | 'sorted_tuple'
        self.choices = choices          # original Python choices (categorical)
        self.length = length            # sorted_tuple length
        self.elem_int = elem_int        # sorted_tuple element is int
        self.is_train = is_train


class N4MFinetuneManager:
    """Native-optimizer finetuning manager (mirrors ``OptunaManager.finetune``)."""

    def __init__(self) -> None:
        self.is_available = N4M_AVAILABLE
        if not self.is_available:
            logger.warning("n4m native optimizer not available - finetuning will be skipped")

    # -- public entry --------------------------------------------------------
    def finetune(
        self,
        dataset: Any,
        model_config: dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        folds: list | None,
        finetune_params: dict[str, Any],
        context: Any,
        controller: Any,
    ) -> FinetuneResult | list[FinetuneResult]:
        if not self.is_available:
            logger.warning("n4m native optimizer unavailable, skipping finetuning")
            return FinetuneResult(best_params={}, best_value=float("inf"), n_trials=0)

        params = self._normalize(finetune_params)
        params = self._resolve_metric_direction(params, dataset)

        approach = params.get("approach", "grouped")
        n_trials = int(params.get("n_trials", 50))
        eval_mode = params.get("eval_mode", "best")
        verbose = int(params.get("verbose", 0))

        if verbose > 1:
            logger.info(
                f"n4m finetuning: sampler={params.get('sampler', 'auto')} "
                f"pruner={params.get('pruner', 'none')} trials={n_trials} "
                f"approach={approach} folds={len(folds) if folds else 0}"
            )

        if folds and approach == "individual":
            results = []
            for fi, (tr, va) in enumerate(folds):
                results.append(self._optimize(
                    dataset, model_config, X_train, y_train, [(tr, va)],
                    params, n_trials, eval_mode, context, controller, verbose,
                    study_name=f"fold_{fi}"))
            return results

        if folds:  # grouped
            return self._optimize(dataset, model_config, X_train, y_train, folds,
                                  params, n_trials, eval_mode, context, controller, verbose)

        # no folds -> holdout split (matches the Optuna path)
        from sklearn.model_selection import train_test_split
        Xtr, Xva, ytr, yva = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return self._optimize(dataset, model_config, Xtr, ytr, [(None, None)], params,
                              n_trials, eval_mode, context, controller, verbose,
                              holdout=(Xva, yva))

    # -- validation / metric -------------------------------------------------
    def _normalize(self, finetune_params: dict[str, Any]) -> dict[str, Any]:
        p = dict(finetune_params)
        if "sample" in p and "sampler" not in p:
            p["sampler"] = p.pop("sample")
        p.pop("engine", None)
        sampler = p.get("sampler", "auto")
        if sampler not in _SAMPLER_MAP:
            raise ValueError(
                f"Unknown n4m sampler '{sampler}'. Valid: {sorted(_SAMPLER_MAP)}")
        pruner = p.get("pruner", "none")
        if pruner not in _PRUNER_MAP:
            raise ValueError(
                f"Unknown n4m pruner '{pruner}'. Valid: {sorted(_PRUNER_MAP)}")
        if p.get("eval_mode") == "avg":
            p["eval_mode"] = "mean"
        return p

    def _resolve_metric_direction(self, params: dict[str, Any], dataset: Any) -> dict[str, Any]:
        from nirs4all.core.metrics import is_higher_better
        p = dict(params)
        metric = p.get("metric")
        if metric is not None:
            if "direction" not in params:
                p["direction"] = "maximize" if is_higher_better(metric) else "minimize"
        else:
            task = getattr(dataset, "task_type", "regression")
            p.setdefault("direction", "maximize" if "classification" in task else "minimize")
        return p

    # -- search-space compilation -------------------------------------------
    def _compile_space(self, params: dict[str, Any]) -> tuple[Any, list[_Slot], dict, dict]:
        space = SearchSpace()
        slots: list[_Slot] = []
        static_model: dict[str, Any] = {}
        static_train: dict[str, Any] = {}

        flat_model = self._flatten(params.get("model_params", {}) or {})
        for name, spec in flat_model.items():
            slot = self._add_axis(space, name, name, spec, is_train=False)
            if slot is not None:
                slots.append(slot)
            else:
                static_model[name] = spec

        for name, spec in (params.get("train_params", {}) or {}).items():
            if not self._is_sampable(spec):
                static_train[name] = spec
                continue
            slot = self._add_axis(space, f"train.{name}", name, spec, is_train=True)
            if slot is not None:
                slots.append(slot)
            else:
                static_train[name] = spec
        return space, slots, static_model, static_train

    def _add_axis(self, space, native, origin, spec, *, is_train) -> _Slot | None:
        """Add one axis to the native space; return its slot, or None if static."""
        # list
        if isinstance(spec, list):
            if (len(spec) == 3 and spec[0] in _ALL_RANGE_TYPES
                    and isinstance(spec[1], (int, float)) and isinstance(spec[2], (int, float))):
                return self._add_range(space, native, origin, spec[0], spec[1], spec[2], is_train=is_train)
            if len(spec) == 2 and isinstance(spec[0], str) and spec[0] in ("bool", "categorical") \
                    and isinstance(spec[1], list):
                return self._add_categorical(space, native, origin, spec[1], is_train=is_train)
            return self._add_categorical(space, native, origin, spec, is_train=is_train)
        # tuple
        if isinstance(spec, tuple) and len(spec) == 3:
            return self._add_range(space, native, origin, spec[0], spec[1], spec[2], is_train=is_train)
        if isinstance(spec, tuple) and len(spec) == 2:
            th, val = spec
            if isinstance(th, str) and th in ("bool", "categorical") and isinstance(val, list):
                return self._add_categorical(space, native, origin, val, is_train=is_train)
            if isinstance(th, int) and isinstance(val, int):
                space.add_int(native, int(th), int(val))
                return _Slot(native, origin, "int", is_train=is_train)
            space.add_float(native, float(th), float(val))
            return _Slot(native, origin, "float", is_train=is_train)
        # dict
        if isinstance(spec, dict):
            return self._add_dict(space, native, origin, spec, is_train=is_train)
        # scalar -> static
        return None

    def _add_range(self, space, native, origin, ptype, lo, hi, *, step=None, log=False, is_train):
        use_log = log or ptype in _INT_LOG_TYPES or ptype in _FLOAT_LOG_TYPES
        if ptype in _INT_TYPES or ptype in _INT_LOG_TYPES:
            space.add_int(native, int(lo), int(hi), int(step) if step else 1, log=use_log)
            return _Slot(native, origin, "int", is_train=is_train)
        if ptype in _FLOAT_TYPES or ptype in _FLOAT_LOG_TYPES:
            space.add_float(native, float(lo), float(hi), float(step) if step else 0.0, log=use_log)
            return _Slot(native, origin, "float", is_train=is_train)
        raise ValueError(f"Unknown parameter type '{ptype}' for '{origin}'")

    def _add_categorical(self, space, native, origin, choices, *, is_train):
        choices = list(choices)
        if not choices:
            raise ValueError(f"Categorical parameter '{origin}' requires a non-empty choice list")
        space.add_categorical(native, choices)
        return _Slot(native, origin, "categorical", choices=choices, is_train=is_train)

    def _add_dict(self, space, native, origin, cfg, *, is_train):
        ptype = cfg.get("type", "categorical")
        if ptype == "categorical":
            choices = cfg.get("choices", cfg.get("values", []))
            return self._add_categorical(space, native, origin, choices, is_train=is_train)
        if ptype in ("int", "int_log", "float", "float_log"):
            lo = cfg.get("min", cfg.get("low"))
            hi = cfg.get("max", cfg.get("high"))
            if lo is None or hi is None:
                raise ValueError(f"Parameter '{origin}' requires min/max (or low/high)")
            step = cfg.get("step")
            log = cfg.get("log", ptype in ("int_log", "float_log"))
            return self._add_range(space, native, origin, ptype, lo, hi, step=step, log=log, is_train=is_train)
        if ptype == "sorted_tuple":
            length = cfg.get("length")
            if not isinstance(length, int):
                raise ValueError(
                    f"n4m engine supports only fixed-length sorted_tuple (got {length!r} "
                    f"for '{origin}'); use the Optuna engine for dynamic length")
            lo = float(cfg.get("min", 0.0))
            hi = float(cfg.get("max", 1.0))
            elem_int = cfg.get("element_type", "float") == "int"
            space.add_sorted_tuple(native, length, lo, hi, element_is_int=elem_int)
            return _Slot(native, origin, "sorted_tuple", length=length, elem_int=elem_int, is_train=is_train)
        raise ValueError(f"Unknown parameter type '{ptype}' for '{origin}'")

    # -- trial resolution ----------------------------------------------------
    def _resolve(self, trial, slots, static_model, static_train) -> tuple[dict, dict]:
        flat_model: dict[str, Any] = {}
        sampled_train: dict[str, Any] = dict(static_train)
        for s in slots:
            if not trial.is_active(s.native):
                continue
            if s.kind == "int":
                val: Any = trial.get_int(s.native)
            elif s.kind == "float":
                val = trial.get_float(s.native)
            elif s.kind == "categorical":
                idx, _ = trial.get_category(s.native)
                val = s.choices[idx]
            else:  # sorted_tuple
                vals = [trial.get_float(f"{s.native}#{i}") for i in range(s.length)]
                val = tuple(int(v) for v in vals) if s.elem_int else tuple(vals)
            (sampled_train if s.is_train else flat_model)[s.origin_name] = val
        model_params = self._unflatten(flat_model)
        model_params.update(static_model)
        return model_params, sampled_train

    # -- optimization loop ---------------------------------------------------
    def _optimize(self, dataset, model_config, X, y, folds, params, n_trials, eval_mode,
                  context, controller, verbose, *, holdout=None, study_name=None) -> FinetuneResult:
        space, slots, static_model, static_train = self._compile_space(params)
        direction = params.get("direction", "minimize")
        opt_metric = params.get("metric")
        pruner_name = _PRUNER_MAP[params.get("pruner", "none")]
        use_pruner = pruner_name != "none" and holdout is None and len(folds) > 1

        opt = Optimizer(
            space,
            sampler=Sampler[_SAMPLER_MAP[params.get("sampler", "auto")].upper()],
            pruner=Pruner[pruner_name.upper()],
            direction=Direction.MAXIMIZE if direction == "maximize" else Direction.MINIMIZE,
            n_startup_trials=int(params.get("n_startup_trials", 10)),
            seed=int(params.get("seed", 0)),
            max_resource=len(folds) if pruner_name == "hyperband" else 0,
            reduction_factor=int(params.get("reduction_factor", 0)),
        )

        # Warm-start: honour force_params as an enqueued first trial (like Optuna).
        force = params.get("force_params")
        if force:
            try:
                opt.enqueue({s.origin_name: force[s.origin_name] for s in slots
                             if not s.is_train and s.origin_name in force})
            except Exception as e:  # enqueue is best-effort
                if verbose > 1:
                    logger.debug(f"n4m enqueue(force_params) skipped: {e}")

        trials: list[TrialSummary] = []
        n_pruned = n_failed = 0
        for _ in range(n_trials):
            trial = opt.ask()
            model_params, sampled_train = self._resolve(trial, slots, static_model, static_train)
            if hasattr(controller, "process_hyperparameters"):
                model_params = controller.process_hyperparameters(model_params)
            t0 = time.perf_counter()
            score, state = self._eval_trial(
                opt, trial, dataset, model_config, X, y, folds, holdout,
                model_params, sampled_train, eval_mode, opt_metric, direction,
                context, controller, use_pruner, verbose)
            dur = time.perf_counter() - t0
            if state == "PRUNED":
                n_pruned += 1
            elif state == "FAIL":
                n_failed += 1
            trials.append(TrialSummary(number=trial.id, params=dict(model_params),
                                       value=(None if score == float("inf") else score),
                                       duration_seconds=dur, state=state))

        best = opt.best()
        if best is None:
            return FinetuneResult(best_params={}, best_value=float("inf"), n_trials=n_trials,
                                  n_pruned=n_pruned, n_failed=n_failed, trials=trials,
                                  study_name=study_name, metric=opt_metric, direction=direction)
        best_trial, best_value = best
        best_params, _ = self._resolve(best_trial, slots, static_model, static_train)
        if hasattr(controller, "process_hyperparameters"):
            best_params = controller.process_hyperparameters(best_params)
        if verbose > 1:
            logger.success(f"n4m best score: {best_value:.4f}  params: {best_params}")
        return FinetuneResult(best_params=best_params, best_value=best_value, n_trials=n_trials,
                              n_pruned=n_pruned, n_failed=n_failed, trials=trials,
                              study_name=study_name, metric=opt_metric, direction=direction)

    def _eval_trial(self, opt, trial, dataset, model_config, X, y, folds, holdout,
                    model_params, sampled_train, eval_mode, opt_metric, direction,
                    context, controller, use_pruner, verbose) -> tuple[float, str]:
        scores: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            try:
                if holdout is not None:
                    X_tr, y_tr = X, y
                    X_va, y_va = holdout
                else:
                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_va, y_va = X[va_idx], y[va_idx]
                model = controller._get_model_instance(dataset, model_config, force_params=model_params)
                X_tr_p, y_tr_p = controller._prepare_data(X_tr, y_tr, context)
                X_va_p, y_va_p = controller._prepare_data(X_va, y_va, context)
                train_kw = dict(sampled_train)
                train_kw.update(model_params)
                train_kw.setdefault("task_type", dataset.task_type)
                trained = controller._train_model(model, X_tr_p, y_tr_p, X_va_p, y_va_p, **train_kw)
                score = controller._evaluate_model(trained, X_va_p, y_va_p,
                                                   metric=opt_metric, direction=direction)
                scores.append(float(score))
            except Exception as e:  # a fold failure is a bad trial, not a crash
                if verbose >= 2:
                    logger.debug(f"   n4m fold failed: {e}")
                scores.append(float("inf"))
            if use_pruner and np.isfinite(scores[-1]):
                if opt.tell_intermediate(trial.id, fold_idx, self._aggregate(scores, eval_mode)):
                    opt.tell_result(trial.id, TrialStatus.PRUNED)
                    return self._aggregate(scores, eval_mode), "PRUNED"

        agg = self._aggregate(scores, eval_mode)
        if not np.isfinite(agg):
            opt.tell_result(trial.id, TrialStatus.FAILED, error="all folds failed")
            return float("inf"), "FAIL"
        opt.tell(trial.id, agg)
        return agg, "COMPLETE"

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _aggregate(scores: list[float], eval_mode: str) -> float:
        if eval_mode == "mean":
            return float(np.mean(scores))
        valid = [s for s in scores if np.isfinite(s)]  # best / robust_best
        return min(valid) if valid else float("inf")

    @staticmethod
    def _is_sampable(cfg: Any) -> bool:
        if isinstance(cfg, (list, tuple)):
            return True
        if isinstance(cfg, dict):
            return "type" in cfg or "min" in cfg or "low" in cfg or "choices" in cfg or "values" in cfg
        return False

    def _is_param_spec(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return ("type" in value or "min" in value or "low" in value
                or "choices" in value or "values" in value)

    def _flatten(self, params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in params.items():
            full = f"{prefix}__{key}" if prefix else key
            if isinstance(value, dict) and not self._is_param_spec(value):
                flat.update(self._flatten(value, full))
            else:
                flat[full] = value
        return flat

    @staticmethod
    def _unflatten(flat_params: dict[str, Any]) -> dict[str, Any]:
        nested: dict[str, Any] = {}
        for key, value in flat_params.items():
            parts = key.split("__")
            cur = nested
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
        return nested
