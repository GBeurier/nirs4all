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
from typing import Any, cast

import numpy as np

from nirs4all.core.logging import get_logger

# Reuse the exact result types the controller refit path expects.
from nirs4all.optimization.optuna import FinetuneResult, TrialSummary

logger = get_logger(__name__)

try:
    from n4m.model_selection.optimizer import (
        ConstraintKind,
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
    "auto": "tpe",
    "grid": "grid",
    "binary": "ternary",
    "sample": "tpe",
    "random": "random",
    "sobol": "sobol",
    "lhs": "lhs",
    "ternary": "ternary",
    "ga": "ga",
    "pso": "pso",
    "cmaes": "cmaes",
    "tpe": "tpe",
    "gp_ei": "gp_ei",
}
_PRUNER_MAP = {
    "none": "none",
    "median": "median",
    "successive_halving": "asha",
    "asha": "asha",
    "hyperband": "hyperband",
    "racing": "racing",
}


class _Slot:
    """One sampled search-space axis (native name + how to read it back)."""

    __slots__ = (
        "native",
        "origin_name",
        "kind",
        "choices",
        "native_choices",
        "length",
        "elem_int",
        "is_train",
    )

    def __init__(self, native, origin_name, kind, *, choices=None, length=0, native_choices=None, elem_int=False, is_train=False):
        self.native = native  # param name in the native SearchSpace
        self.origin_name = origin_name  # flat model name, or train param name
        self.kind = kind  # 'int' | 'float' | 'categorical' | 'sorted_tuple'
        self.choices = choices  # original Python choices (categorical)
        self.native_choices = native_choices  # labels passed to native SearchSpace
        self.length = length  # sorted_tuple length
        self.elem_int = elem_int  # sorted_tuple element is int
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
            logger.info(f"n4m finetuning: sampler={params.get('sampler', 'auto')} pruner={params.get('pruner', 'none')} trials={n_trials} approach={approach} folds={len(folds) if folds else 0}")

        if folds and approach == "individual":
            results = []
            for fi, (tr, va) in enumerate(folds):
                results.append(self._optimize(dataset, model_config, X_train, y_train, [(tr, va)], params, n_trials, eval_mode, context, controller, verbose, study_name=f"fold_{fi}"))
            return results

        if folds and approach == "grouped":
            return self._optimize(dataset, model_config, X_train, y_train, folds, params, n_trials, eval_mode, context, controller, verbose)

        # approach == "single" (even with folds), or no folds -> holdout split
        # (matches the Optuna path).
        from sklearn.model_selection import train_test_split

        Xtr, Xva, ytr, yva = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return self._optimize(dataset, model_config, Xtr, ytr, [(None, None)], params, n_trials, eval_mode, context, controller, verbose, holdout=(Xva, yva))

    # -- validation / metric -------------------------------------------------
    def _normalize(self, finetune_params: dict[str, Any]) -> dict[str, Any]:
        p = dict(finetune_params)
        if "sample" in p:
            sample_alias = p.pop("sample")
            if "sampler" not in p:
                p["sampler"] = sample_alias
        p.pop("engine", None)

        sampler = self._normalize_token(p.get("sampler"), default="auto")
        if sampler not in _SAMPLER_MAP:
            raise ValueError(f"Unknown n4m sampler '{sampler}'. Valid: {sorted(_SAMPLER_MAP)}")
        p["sampler"] = sampler

        pruner = self._normalize_token(p.get("pruner"), default="none")
        if pruner not in _PRUNER_MAP:
            raise ValueError(f"Unknown n4m pruner '{pruner}'. Valid: {sorted(_PRUNER_MAP)}")
        p["pruner"] = pruner

        approach = self._normalize_token(p.get("approach"), default="grouped")
        if approach not in ("single", "grouped", "individual"):
            raise ValueError(f"Unknown approach '{approach}'. Valid: single, grouped, individual")
        p["approach"] = approach

        em = self._normalize_token(p.get("eval_mode"), default="best")
        if em == "avg":
            em = "mean"
        if em not in ("best", "mean", "robust_best"):
            raise ValueError(f"Unknown eval_mode '{em}'. Valid: best, mean, robust_best")
        p["eval_mode"] = em

        direction = p.get("direction")
        if direction is not None:
            direction = self._normalize_token(direction)
            if direction not in ("minimize", "maximize"):
                raise ValueError(f"Unknown direction '{direction}'. Valid: minimize, maximize")
            p["direction"] = direction
        return p

    @staticmethod
    def _normalize_token(value: Any, *, default: str | None = None) -> str:
        if value is None:
            if default is None:
                return ""
            value = default
        return str(value).strip().lower()

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
        static_model: dict[str, Any] = {}  # FLAT keys (merged + unflattened at resolve time)
        static_train: dict[str, Any] = {}
        conditions: list[tuple] = []  # (child_native, parent, labels, is_in)

        flat_model = self._flatten(params.get("model_params", {}) or {})
        for name, spec in flat_model.items():
            cond, spec = self._extract_when(name, spec)
            slot = self._add_axis(space, name, name, spec, is_train=False)
            if slot is not None:
                slots.append(slot)
                if cond is not None:
                    conditions.append(cond)
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

        # Conditional axes (the ``when`` clause): a param is active only when another
        # param's chosen label matches — compiled into native conditional-activation
        # constraints. This is how operator/sub-model attributes enter the search
        # space (object__attribute conditioned on a sibling choice).
        for child, parent, labels, is_in in conditions:
            kind = ConstraintKind.CONDITION_IN if is_in else ConstraintKind.CONDITION_NOT_IN
            for lab in labels:
                space.add_constraint(kind, [child, parent], ["", str(lab)])
        return space, slots, static_model, static_train

    @staticmethod
    def _extract_when(name, spec):
        """Pull an optional ``when``/``when_not`` clause off a dict param spec.

        ``{"type": "float", ..., "when": {"kernel": "rbf"}}`` -> the param is active
        only when the categorical ``kernel`` == "rbf". Accepts a single value or a
        list of labels. Returns ``(condition | None, spec_without_when)``.
        """
        if not isinstance(spec, dict) or ("when" not in spec and "when_not" not in spec):
            return None, spec
        spec = dict(spec)
        if "when" in spec:
            raw = spec.pop("when")
            spec.pop("when_not", None)
            is_in = True
        else:
            raw = spec.pop("when_not")
            is_in = False
        if not isinstance(raw, dict) or len(raw) != 1:
            raise ValueError(f"'when'/'when_not' for '{name}' must be a single {{parent: value_or_list}} mapping")
        parent, val = next(iter(raw.items()))
        labels = list(val) if isinstance(val, (list, tuple)) else [val]
        if not labels:
            raise ValueError(f"'when'/'when_not' for '{name}' has an empty label set")
        return (name, parent, labels, is_in), spec

    def _add_axis(self, space, native, origin, spec, *, is_train) -> _Slot | None:
        """Add one axis to the native space; return its slot, or None if static."""
        # list
        if isinstance(spec, list):
            if len(spec) == 3 and spec[0] in _ALL_RANGE_TYPES and isinstance(spec[1], (int, float)) and isinstance(spec[2], (int, float)):
                use_log = spec[0] in _INT_LOG_TYPES or spec[0] in _FLOAT_LOG_TYPES
                return cast(_Slot | None, self._add_range(space, native, origin, spec[0], spec[1], spec[2], use_log=use_log, is_train=is_train))
            if len(spec) == 2 and isinstance(spec[0], str) and spec[0] in ("bool", "categorical") and isinstance(spec[1], list):
                return cast(_Slot | None, self._add_categorical(space, native, origin, spec[1], is_train=is_train))
            return cast(_Slot | None, self._add_categorical(space, native, origin, spec, is_train=is_train))
        # tuple
        if isinstance(spec, tuple) and len(spec) == 3:
            use_log = spec[0] in _INT_LOG_TYPES or spec[0] in _FLOAT_LOG_TYPES
            return cast(_Slot | None, self._add_range(space, native, origin, spec[0], spec[1], spec[2], use_log=use_log, is_train=is_train))
        if isinstance(spec, tuple) and len(spec) == 2:
            th, val = spec
            if isinstance(th, str) and th in ("bool", "categorical") and isinstance(val, list):
                return cast(_Slot | None, self._add_categorical(space, native, origin, val, is_train=is_train))
            if isinstance(th, int) and isinstance(val, int):
                space.add_int(native, int(th), int(val))
                return _Slot(native, origin, "int", is_train=is_train)
            space.add_float(native, float(th), float(val))
            return _Slot(native, origin, "float", is_train=is_train)
        # dict
        if isinstance(spec, dict):
            return cast(_Slot | None, self._add_dict(space, native, origin, spec, is_train=is_train))
        # scalar -> static
        return None

    def _add_range(self, space, native, origin, ptype, lo, hi, *, use_log, step=None, is_train):
        if step is not None and step <= 0:
            raise ValueError(f"step for '{origin}' must be positive (got {step})")
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
        return _Slot(native, origin, "categorical", choices=choices, native_choices=choices, is_train=is_train)

    def _add_dict(self, space, native, origin, cfg, *, is_train):
        ptype = cfg.get("type", "categorical")
        if ptype == "categorical":
            # Named options: {"options": {"pls": PLSRegression(), "ridge": Ridge()}}.
            # The native categorical ranges over the stable NAMES (so `when` clauses
            # key on a name, not a fragile str(instance)); resolution returns the
            # mapped object. This is how operators/estimators enter the search space.
            if "options" in cfg:
                opts = cfg["options"]
                if not isinstance(opts, dict) or not opts:
                    raise ValueError(f"'options' for '{origin}' must be a non-empty mapping")
                names = list(opts.keys())
                for n in names:  # names ARE the native labels (`when` matches them)
                    if not isinstance(n, str) or n == "" or "\0" in n:
                        raise ValueError(f"'options' names for '{origin}' must be non-empty NUL-free strings (got {n!r}); wrap the value's identity in a name")
                space.add_categorical(native, names)
                return _Slot(
                    native,
                    origin,
                    "categorical",
                    choices=list(opts.values()),
                    native_choices=names,
                    is_train=is_train,
                )
            choices = cfg.get("choices", cfg.get("values", []))
            return self._add_categorical(space, native, origin, choices, is_train=is_train)
        if ptype in ("int", "int_log", "float", "float_log"):
            lo = cfg.get("min", cfg.get("low"))
            hi = cfg.get("max", cfg.get("high"))
            if lo is None or hi is None:
                raise ValueError(f"Parameter '{origin}' requires min/max (or low/high)")
            step = cfg.get("step")
            # Explicit `log` wins; otherwise the type suffix decides.
            use_log = bool(cfg["log"]) if "log" in cfg else ptype in ("int_log", "float_log")
            return self._add_range(space, native, origin, ptype, lo, hi, use_log=use_log, step=step, is_train=is_train)
        if ptype == "sorted_tuple":
            # sorted_tuple has non-trivial dynamic-length / step / log semantics that
            # the native space does not yet cover losslessly; fail loudly rather than
            # silently change the space. Use the Optuna engine for sorted_tuple.
            raise NotImplementedError(f'the n4m engine does not support sorted_tuple (\'{origin}\'); use the Optuna engine (drop "engine": "n4m") for this parameter')
        raise ValueError(f"Unknown parameter type '{ptype}' for '{origin}'")

    # -- trial resolution ----------------------------------------------------
    def _resolve(self, trial, slots, static_model, static_train, flat_heads) -> tuple[dict, dict]:
        # Static model params carry FLAT keys; merge them with the sampled flat
        # params BEFORE unflattening so a nested static value (e.g. cfg__mode) lands
        # in the right nested group instead of a bogus top-level "cfg__mode" key.
        flat_model: dict[str, Any] = dict(static_model)
        sampled_train: dict[str, Any] = dict(static_train)
        for s in slots:
            if not trial.is_active(s.native):
                continue
            if s.kind == "int":
                val: Any = trial.get_int(s.native)
            elif s.kind == "float":
                val = trial.get_float(s.native)
            else:  # categorical
                idx, _ = trial.get_category(s.native)
                val = s.choices[idx]
            (sampled_train if s.is_train else flat_model)[s.origin_name] = val
        # A key `a__b` whose head `a` is a DECLARED standalone param is scikit-learn
        # set_params addressing (choose operator `a`, then set its sub-param `a__b`)
        # — keep it flat. `flat_heads` is computed from the declared slots/static keys
        # (NOT the resolved active dict), so an inactive `est` still keeps `est__alpha`
        # flat. A head that only ever appears with "__" is a genuine nested group
        # (e.g. cfg__mode) and IS unflattened.
        model_params = self._unflatten(flat_model, keep_flat_heads=flat_heads)
        return model_params, sampled_train

    # -- optimization loop ---------------------------------------------------
    def _optimize(self, dataset, model_config, X, y, folds, params, n_trials, eval_mode, context, controller, verbose, *, holdout=None, study_name=None) -> FinetuneResult:
        space, slots, static_model, static_train = self._compile_space(params)
        # Standalone declared param heads (set_params addressing targets) — computed
        # from the DECLARATION, not per-trial activity, so it is stable across trials.
        flat_heads = {s.origin_name for s in slots if not s.is_train and "__" not in s.origin_name} | {k for k in static_model if "__" not in k}
        direction = params.get("direction", "minimize")
        opt_metric = params.get("metric")
        pruner_name = _PRUNER_MAP[params.get("pruner", "none")]
        use_pruner = pruner_name != "none" and holdout is None and len(folds) > 1

        opt = Optimizer(
            space,
            sampler=self._native_sampler(params.get("sampler", "auto")),
            pruner=Pruner[pruner_name.upper()],
            direction=Direction.MAXIMIZE if direction == "maximize" else Direction.MINIMIZE,
            n_startup_trials=int(params.get("n_startup_trials", 10)),
            seed=int(params.get("seed") or 0),
            max_resource=len(folds) if pruner_name == "hyperband" else 0,
            reduction_factor=int(params.get("reduction_factor", 0)),
        )

        # Warm-start: honour force_params as an enqueued first trial (like Optuna).
        force = params.get("force_params")
        if force:
            self._enqueue_force_params(opt, slots, force)

        trials: list[TrialSummary] = []
        n_pruned = n_failed = 0
        for _ in range(n_trials):
            trial = opt.ask()
            t0 = time.perf_counter()
            # Resolving the trial + processing the hyperparameters happens BEFORE the
            # fold loop and must not escape untold, or the optimizer state and best()
            # would be left inconsistent and the whole run would abort.
            try:
                model_params, sampled_train = self._resolve(trial, slots, static_model, static_train, flat_heads)
                if hasattr(controller, "process_hyperparameters"):
                    model_params = controller.process_hyperparameters(model_params)
            except Exception as e:
                opt.tell_result(trial.id, TrialStatus.FAILED, error=str(e)[:200])
                n_failed += 1
                trials.append(TrialSummary(number=trial.id, params={}, value=None, duration_seconds=time.perf_counter() - t0, state="FAIL"))
                if verbose >= 2:
                    logger.debug(f"   n4m trial resolution failed: {e}")
                continue
            score, state = self._eval_trial(opt, trial, dataset, model_config, X, y, folds, holdout, model_params, sampled_train, eval_mode, opt_metric, direction, context, controller, use_pruner, verbose)
            dur = time.perf_counter() - t0
            if state == "PRUNED":
                n_pruned += 1
            elif state == "FAIL":
                n_failed += 1
            trials.append(TrialSummary(number=trial.id, params=dict(model_params), value=(None if not np.isfinite(score) else score), duration_seconds=dur, state=state))

        best = opt.best()
        if best is None:
            return FinetuneResult(best_params={}, best_value=float("inf"), n_trials=n_trials, n_pruned=n_pruned, n_failed=n_failed, trials=trials, study_name=study_name, metric=opt_metric, direction=direction)
        best_trial, best_value = best
        best_params, _ = self._resolve(best_trial, slots, static_model, static_train, flat_heads)
        if hasattr(controller, "process_hyperparameters"):
            best_params = controller.process_hyperparameters(best_params)
        if verbose > 1:
            logger.success(f"n4m best score: {best_value:.4f}  params: {best_params}")
        return FinetuneResult(best_params=best_params, best_value=best_value, n_trials=n_trials, n_pruned=n_pruned, n_failed=n_failed, trials=trials, study_name=study_name, metric=opt_metric, direction=direction)

    def _native_sampler(self, sampler_name: str) -> Any:
        native_name = _SAMPLER_MAP[sampler_name]
        enum_name = native_name.upper()
        try:
            return Sampler[enum_name]
        except Exception as exc:
            raise NotImplementedError(f"n4m sampler '{sampler_name}' requires native Sampler.{enum_name} support; upgrade n4m/libn4m or choose another sampler") from exc

    def _enqueue_force_params(self, opt: Any, slots: list[_Slot], force_params: dict[str, Any]) -> None:
        if not hasattr(opt, "enqueue"):
            raise NotImplementedError("n4m force_params requires native optimizer.enqueue(...) support; upgrade n4m/libn4m or use the Optuna engine")

        payload = self._encode_force_params(slots, force_params)
        try:
            opt.enqueue(payload)
        except Exception as exc:
            raise RuntimeError("n4m force_params could not be enqueued through optimizer.enqueue(...); the warm-start contract is fail-closed") from exc

    def _encode_force_params(self, slots: list[_Slot], force_params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(force_params, dict):
            raise TypeError("n4m force_params must be a mapping of model parameter names to values")

        model_slots = {slot.origin_name: slot for slot in slots if not slot.is_train}
        unknown = sorted(set(force_params) - set(model_slots))
        if unknown:
            raise ValueError(f"n4m force_params keys must be a subset of sampled finetune_params.model_params; unknown keys: {unknown}")

        encoded: dict[str, Any] = {}
        for name, value in force_params.items():
            slot = model_slots[name]
            if slot.kind == "categorical":
                encoded[name] = self._encode_categorical_force_value(slot, value)
            else:
                encoded[name] = value
        return encoded

    @staticmethod
    def _encode_categorical_force_value(slot: _Slot, value: Any) -> Any:
        choices = list(slot.choices or [])
        native_choices = list(slot.native_choices or choices)
        for index, choice in enumerate(choices):
            if value is choice:
                return native_choices[index]
            try:
                if value == choice:
                    return native_choices[index]
            except Exception:
                continue
        raise ValueError(f"n4m force_params for categorical parameter '{slot.origin_name}' must use one of the public decoded choices")

    def _eval_trial(self, opt, trial, dataset, model_config, X, y, folds, holdout, model_params, sampled_train, eval_mode, opt_metric, direction, context, controller, use_pruner, verbose) -> tuple[float, str]:
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
                score = controller._evaluate_model(trained, X_va_p, y_va_p, metric=opt_metric, direction=direction)
                scores.append(float(score))
            except Exception as e:  # a fold failure is a bad trial, not a crash
                if verbose >= 2:
                    logger.debug(f"   n4m fold failed: {e}")
                # a failed fold takes the worst value for the direction
                scores.append(float("inf") if direction != "maximize" else float("-inf"))
            # Prune only on a FINITE aggregate — a non-finite intermediate (e.g. a
            # failed fold under mean-mode) would trip the native finite-score guard.
            if use_pruner:
                agg = self._aggregate(scores, eval_mode, direction)
                if np.isfinite(agg) and opt.tell_intermediate(trial.id, fold_idx, agg):
                    opt.tell_result(trial.id, TrialStatus.PRUNED)
                    return agg, "PRUNED"

        agg = self._aggregate(scores, eval_mode, direction)
        if not np.isfinite(agg):
            opt.tell_result(trial.id, TrialStatus.FAILED, error="all folds failed")
            return agg, "FAIL"
        opt.tell(trial.id, agg)
        return agg, "COMPLETE"

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _aggregate(scores: list[float], eval_mode: str, direction: str = "minimize") -> float:
        if eval_mode == "mean":
            return float(np.mean(scores))
        valid = [s for s in scores if np.isfinite(s)]  # best / robust_best
        if not valid:
            return float("-inf") if direction == "maximize" else float("inf")
        return max(valid) if direction == "maximize" else min(valid)

    @staticmethod
    def _is_sampable(cfg: Any) -> bool:
        if isinstance(cfg, (list, tuple)):
            return True
        if isinstance(cfg, dict):
            return any(k in cfg for k in ("type", "min", "low", "choices", "values", "options"))
        return False

    def _is_param_spec(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return any(k in value for k in ("type", "min", "low", "choices", "values", "options"))

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
    def _unflatten(flat_params: dict[str, Any], keep_flat_heads: set | None = None) -> dict[str, Any]:
        keep_flat_heads = keep_flat_heads or set()
        nested: dict[str, Any] = {}
        for key, value in flat_params.items():
            if "__" in key and key.split("__", 1)[0] in keep_flat_heads:
                nested[key] = value  # set_params addressing (operator sub-param): keep flat
                continue
            parts = key.split("__")
            cur = nested
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
        return nested
