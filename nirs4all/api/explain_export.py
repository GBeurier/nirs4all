"""Export fitted host estimators to the Methods model-specific explain inputs.

The SDK owns its Python models; Methods owns attribution numerics. This module
only converts fitted scikit-learn, XGBoost and LightGBM tree ensembles to the
flat path-dependent TreeSHAP layout of ``n4m.explain.TreeEnsemble`` and recovers
the affine coefficients of linear estimators. Callers must verify that the
exported model reproduces the explained predictor before reporting attributions.

XGBoost and LightGBM are optional: a model can only be an instance of their
classes when the package is already imported, so no import is attempted here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

# Objectives whose prediction is the raw additive margin (identity link).
_XGBOOST_IDENTITY_OBJECTIVES = frozenset({"reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror", "reg:quantileerror"})
_LIGHTGBM_IDENTITY_OBJECTIVES = frozenset({"regression", "regression_l1", "huber", "fair", "quantile", "mape"})


@dataclass(frozen=True)
class FlatTreeEnsemble:
    """Flat tree ensemble in the ``n4m.explain.TreeEnsemble`` layout.

    Attributes:
        tree_offsets: ``n_trees + 1`` node offsets; tree ``t`` owns nodes
            ``[tree_offsets[t], tree_offsets[t + 1])``.
        feature: Split feature per node, ``-1`` for leaves.
        threshold: Split threshold per node (ignored for leaves).
        left: Local left child index per node, ``-1`` for leaves.
        right: Local right child index per node, ``-1`` for leaves.
        cover: Training weight reaching each node.
        value: Leaf values, ``n_nodes * n_outputs`` row-major.
        n_outputs: Output columns of every leaf.
        decision_type: ``0`` for ``x <= threshold`` going left, ``1`` for ``x < threshold``.
        default_left: Optional per-node missing-value direction.
        base_offset: Optional per-output constant added to the ensemble output.
        tree_scale: Factor applied to the summed leaf values.
        input_dtype: Precision in which the estimator compares its inputs;
            rows must be rounded to it before explanation.
    """

    tree_offsets: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    cover: np.ndarray
    value: np.ndarray
    n_outputs: int
    decision_type: int = 0
    default_left: np.ndarray | None = None
    base_offset: np.ndarray | None = None
    tree_scale: float = 1.0
    input_dtype: type = np.float64

    def ensemble_fields(self) -> dict[str, Any]:
        """Return the keyword arguments of ``n4m.explain.TreeEnsemble``."""
        return {
            "tree_offsets": self.tree_offsets, "feature": self.feature, "threshold": self.threshold, "left": self.left, "right": self.right,
            "cover": self.cover, "value": self.value, "n_outputs": self.n_outputs, "decision_type": self.decision_type,
            "default_left": self.default_left, "base_offset": self.base_offset, "tree_scale": self.tree_scale,
        }


def _concatenate(trees: list[dict[str, np.ndarray]], n_outputs: int, **fields: Any) -> FlatTreeEnsemble:
    sizes = [len(tree["feature"]) for tree in trees]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    default_left = np.concatenate([tree["default_left"] for tree in trees]).astype(np.uint8) if "default_left" in trees[0] else None
    return FlatTreeEnsemble(
        tree_offsets=offsets,
        feature=np.concatenate([tree["feature"] for tree in trees]).astype(np.int64),
        threshold=np.concatenate([tree["threshold"] for tree in trees]).astype(np.float64),
        left=np.concatenate([tree["left"] for tree in trees]).astype(np.int64),
        right=np.concatenate([tree["right"] for tree in trees]).astype(np.int64),
        cover=np.concatenate([tree["cover"] for tree in trees]).astype(np.float64),
        value=np.concatenate([tree["value"] for tree in trees]).astype(np.float64),
        n_outputs=n_outputs, default_left=default_left, **fields,
    )


def _sklearn_tree(estimator: Any, *, probabilities: bool) -> dict[str, np.ndarray]:
    tree = estimator.tree_
    leaf = tree.children_left == -1
    value = np.asarray(tree.value, dtype=np.float64)
    if probabilities:
        # predict_proba normalizes each leaf's class weights (counts in older
        # scikit-learn releases, fractions in newer ones).
        value = value[:, 0, :]
        value = value / value.sum(axis=1, keepdims=True)
    else:
        value = value[:, :, 0]
    return {
        "feature": np.where(leaf, -1, tree.feature),
        "threshold": np.where(leaf, 0.0, tree.threshold),
        "left": tree.children_left, "right": tree.children_right,
        "cover": np.asarray(tree.weighted_n_node_samples, dtype=np.float64),
        "value": value.reshape(-1),
    }


def _sklearn_tree_types() -> tuple[tuple[type[Any], ...], tuple[type[Any], ...]]:
    """Return the supported scikit-learn (classifier, regressor) tree types."""
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    return (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier), (
        DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)


def _sklearn_ensemble(model: Any, *, probabilities: bool) -> FlatTreeEnsemble | None:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    classifiers, regressors = _sklearn_tree_types()
    if probabilities:
        if isinstance(model, classifiers):
            if model.n_outputs_ != 1:
                raise ValueError("TreeSHAP export supports single-output classifiers only")
            members = [model] if isinstance(model, DecisionTreeClassifier) else list(model.estimators_)
            return _concatenate([_sklearn_tree(tree, probabilities=True) for tree in members], int(model.n_classes_), tree_scale=1.0 / len(members),
                                input_dtype=np.float32)
        return None
    if isinstance(model, regressors) and not isinstance(model, GradientBoostingRegressor):
        members = [model] if isinstance(model, DecisionTreeRegressor) else list(model.estimators_)
        return _concatenate([_sklearn_tree(tree, probabilities=False) for tree in members], int(model.n_outputs_), tree_scale=1.0 / len(members),
                            input_dtype=np.float32)
    if isinstance(model, GradientBoostingRegressor):
        if isinstance(model.init_, DummyRegressor):
            offset = float(np.asarray(model.init_.constant_).reshape(-1)[0])
        elif model.init_ == "zero":
            offset = 0.0
        else:
            raise ValueError("GradientBoostingRegressor with a custom init estimator has a non-constant offset; TreeSHAP export refuses it")
        trees = [_sklearn_tree(stage[0], probabilities=False) for stage in model.estimators_]
        return _concatenate(trees, 1, base_offset=np.array([offset]), tree_scale=float(model.learning_rate), input_dtype=np.float32)
    return None


def _xgboost_ensemble(model: Any) -> FlatTreeEnsemble | None:
    xgboost = sys.modules.get("xgboost")
    if xgboost is None or not isinstance(model, (xgboost.XGBModel, xgboost.Booster)):
        return None
    booster = model if isinstance(model, xgboost.Booster) else model.get_booster()
    if not isinstance(model, xgboost.Booster):
        # The scikit-learn wrapper predicts with rounds [0, best_iteration] after
        # early stopping (every parallel tree of those rounds); export that slice.
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            booster = booster[: int(best_iteration) + 1]
    learner = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))["learner"]
    objective = learner["objective"]["name"]
    if objective not in _XGBOOST_IDENTITY_OBJECTIVES:
        raise ValueError(f"XGBoost objective {objective!r} applies a link function; TreeSHAP export supports identity-link regression objectives only")
    booster_json = learner["gradient_booster"]
    if booster_json["name"] != "gbtree":
        raise ValueError(f"XGBoost booster {booster_json['name']!r} is not supported; TreeSHAP export requires 'gbtree'")
    parameters = learner["learner_model_param"]
    if int(parameters.get("num_class", 0)) > 1 or int(parameters.get("num_target", 1)) > 1:
        raise ValueError("TreeSHAP export supports single-output XGBoost regressors only")
    trees = []
    for tree in booster_json["model"]["trees"]:
        if any(int(kind) != 0 for kind in tree.get("split_type", [])):
            raise ValueError("XGBoost categorical splits are not supported by TreeSHAP export")
        left = np.asarray(tree["left_children"], dtype=np.int64)
        leaf = left == -1
        # XGBoost stores float32 values; the JSON decimals must be restored to
        # float32 before promotion or a threshold such as 0.7 changes branch.
        conditions = np.asarray(tree["split_conditions"], dtype=np.float32).astype(np.float64)
        trees.append({
            "feature": np.where(leaf, -1, np.asarray(tree["split_indices"], dtype=np.int64)),
            # XGBoost stores each leaf's (learning-rate scaled) value in split_conditions.
            "threshold": np.where(leaf, 0.0, conditions), "value": np.where(leaf, conditions, 0.0),
            "left": left, "right": np.asarray(tree["right_children"], dtype=np.int64),
            "cover": np.asarray(tree["sum_hessian"], dtype=np.float32).astype(np.float64),
            "default_left": np.asarray(tree["default_left"], dtype=np.uint8),
        })
    base_score = float(np.float32(str(parameters["base_score"]).strip("[]")))
    return _concatenate(trees, 1, decision_type=1, base_offset=np.array([base_score]), input_dtype=np.float32)


def _lightgbm_tree(root: dict[str, Any]) -> dict[str, np.ndarray]:
    # LightGBM keeps float64 thresholds and leaf values, compares float64 inputs
    # with ``<=`` and writes them with round-trip (17 digit) precision in dump_model.
    columns: dict[str, list[float]] = {name: [] for name in ("feature", "threshold", "left", "right", "cover", "value", "default_left")}
    stack: list[tuple[dict[str, Any], int, bool]] = [(root, -1, False)]
    while stack:
        node, parent, is_left = stack.pop()
        index = len(columns["feature"])
        if parent >= 0:
            columns["left" if is_left else "right"][parent] = index
        columns["left"].append(-1)
        columns["right"].append(-1)
        if "split_feature" in node:
            if node["decision_type"] != "<=":
                raise ValueError("LightGBM categorical splits are not supported by TreeSHAP export")
            if node.get("missing_type") == "Zero":
                raise ValueError("LightGBM zero-as-missing splits are not supported by TreeSHAP export")
            columns["feature"].append(int(node["split_feature"]))
            columns["threshold"].append(float(node["threshold"]))
            columns["cover"].append(float(node["internal_count"]))
            columns["value"].append(0.0)
            columns["default_left"].append(1 if node.get("default_left") else 0)
            stack.append((node["right_child"], index, False))
            stack.append((node["left_child"], index, True))
        else:
            if "leaf_coeff" in node:
                raise ValueError("LightGBM linear trees are not supported by TreeSHAP export")
            columns["feature"].append(-1)
            columns["threshold"].append(0.0)
            # A constant single-leaf tree records no count; its cover only needs to be positive.
            columns["cover"].append(float(node.get("leaf_count", 1)))
            columns["value"].append(float(node["leaf_value"]))
            columns["default_left"].append(0)
    return {name: np.asarray(values) for name, values in columns.items()}


def _lightgbm_ensemble(model: Any) -> FlatTreeEnsemble | None:
    lightgbm = sys.modules.get("lightgbm")
    if lightgbm is None or not isinstance(model, (lightgbm.LGBMModel, lightgbm.Booster)):
        return None
    booster = model if isinstance(model, lightgbm.Booster) else model.booster_
    # Like Booster.predict, dump_model() defaults to the best iteration when early
    # stopping recorded one and to every iteration otherwise.
    dump = booster.dump_model()
    objective = str(dump.get("objective", "")).split()[0] if dump.get("objective") else ""
    if objective not in _LIGHTGBM_IDENTITY_OBJECTIVES:
        raise ValueError(f"LightGBM objective {objective!r} applies a link function; TreeSHAP export supports identity-link regression objectives only")
    if int(dump.get("num_class", 1)) != 1 or int(dump.get("num_tree_per_iteration", 1)) != 1:
        raise ValueError("TreeSHAP export supports single-output LightGBM regressors only")
    trees = [_lightgbm_tree(info["tree_structure"]) for info in dump["tree_info"]]
    scale = 1.0 / len(trees) if dump.get("average_output") else 1.0
    return _concatenate(trees, 1, tree_scale=scale)


def export_tree_ensemble(model: Any, *, probabilities: bool) -> FlatTreeEnsemble:
    """Export a fitted tree model to the flat path-dependent TreeSHAP layout.

    Supported regressors: scikit-learn ``DecisionTreeRegressor``,
    ``ExtraTreeRegressor``, ``RandomForestRegressor``, ``ExtraTreesRegressor``,
    ``GradientBoostingRegressor`` (constant init), XGBoost ``gbtree`` and
    LightGBM models with identity-link regression objectives. Supported
    probability outputs: scikit-learn single-output decision-tree, random-forest
    and extra-trees classifiers (``predict_proba`` columns).

    Args:
        model: Fitted terminal estimator.
        probabilities: Whether the explained output is ``predict_proba``.

    Returns:
        The flat ensemble whose output equals the model prediction.

    Raises:
        ValueError: If the model is not a supported tree ensemble.
    """
    ensemble = _sklearn_ensemble(model, probabilities=probabilities)
    if ensemble is None and not probabilities:
        ensemble = _xgboost_ensemble(model) or _lightgbm_ensemble(model)
    if ensemble is not None:
        return ensemble
    output = "class-probability" if probabilities else "regression"
    raise ValueError(f"{type(model).__name__} is not a supported {output} tree ensemble for TreeSHAP export")


def is_tree_ensemble(model: Any, *, probabilities: bool) -> bool:
    """Return whether :func:`export_tree_ensemble` recognizes the model type."""
    classifiers, regressors = _sklearn_tree_types()
    if probabilities:
        return isinstance(model, classifiers)
    if isinstance(model, regressors):
        return True
    xgboost, lightgbm = sys.modules.get("xgboost"), sys.modules.get("lightgbm")
    return (xgboost is not None and isinstance(model, (xgboost.XGBModel, xgboost.Booster))) or (
        lightgbm is not None and isinstance(model, (lightgbm.LGBMModel, lightgbm.Booster)))


def affine_coefficients(model: Any, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover verified affine coefficients of a fitted linear estimator.

    The coefficient orientation is resolved from ``coef_`` against the
    estimator's own predictions, which covers the transposed
    ``PLSRegression.coef_`` layouts of different scikit-learn releases. The
    intercept is recovered from the predictions rather than ``intercept_``
    because estimators such as ``PLSRegression`` center inputs before applying
    ``coef_``.

    Args:
        model: Fitted estimator exposing ``coef_`` and ``predict``.
        Z: Reference rows (at least two distinct rows) in the estimator's input space.

    Returns:
        ``(coefficients, intercept)`` with shapes ``(n_features, n_outputs)``
        and ``(n_outputs,)`` such that ``model.predict(Z) == Z @ coefficients + intercept``.

    Raises:
        ValueError: If the estimator exposes no coefficients reproducing its predictions.
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise ValueError(f"{type(model).__name__} exposes no coef_; linear SHAP requires an affine estimator")
    predicted = np.asarray(model.predict(Z), dtype=np.float64).reshape(len(Z), -1)
    coef = np.asarray(coef, dtype=np.float64)
    candidates = [coef.reshape(-1, 1)] if coef.ndim == 1 else [coef.T, coef] if coef.ndim == 2 else []
    for coefficients in candidates:
        if coefficients.shape != (Z.shape[1], predicted.shape[1]):
            continue
        residual = predicted - Z @ coefficients
        intercept = residual.mean(axis=0)
        scale = max(1.0, float(np.max(np.abs(Z) @ np.abs(coefficients))))
        if float(np.max(np.abs(residual - intercept))) <= 1e-8 * scale:
            return coefficients, intercept
    raise ValueError(f"{type(model).__name__}.coef_ does not reproduce its predictions as an affine function; linear SHAP refuses it")
