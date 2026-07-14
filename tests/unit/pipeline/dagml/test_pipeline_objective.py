"""Unit locks for the native DAG-ML shared pipeline objective seam."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from nirs4all.pipeline.dagml.pipeline_objective import (
    PipelineObjective,
    apply_trial_parameter_patch,
    make_conformal_prediction_score_extractor,
    make_prediction_score_extractor,
)
from nirs4all.pipeline.dagml.pipeline_objective_compiler import (
    CompiledPipelineObjective,
    compile_pipeline_objective,
    run_single_estimator_tuning_to_run_result,
)
from nirs4all.pipeline.dagml.tuning_contracts import DagMLTuningSpec, TuningResult, parse_tuning_spec


class _ObjectiveEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0, nested: Any = None) -> None:
        self.alpha = alpha
        self.nested = nested

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> _ObjectiveEstimator:
        self.fit_X_ = X
        self.fit_y_ = y
        self.fit_sample_ids_ = sample_ids
        self.fit_groups_ = groups
        self.fit_metadata_ = metadata
        self.training_result_ = {"score": float(self.alpha) + 0.5}
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), float(self.alpha))


class _Nested:
    def __init__(self) -> None:
        self.depth = 1


class _StringifiesAs:
    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


class _NoPredictEstimator(BaseEstimator):
    def fit(self, X: Any, y: Any, **_kwargs: Any) -> _NoPredictEstimator:
        return self


class _ScaleTransformer(BaseEstimator):
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def fit(self, X: Any, y: Any | None = None) -> _ScaleTransformer:
        self.fit_seen_y_ = y
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float) * float(self.factor)


class _LinearEstimator(BaseEstimator):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> _LinearEstimator:
        self.fit_X_ = np.asarray(X, dtype=float)
        self.fit_y_ = np.asarray(y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0] + float(self.alpha)


def test_pipeline_objective_evaluates_one_candidate_on_cloned_estimator() -> None:
    estimator = _ObjectiveEstimator(alpha=1.0)
    objective = PipelineObjective(
        estimator,
        parse_tuning_spec({"engine": "n4m", "space": {"alpha": [0.2, 0.7]}}),
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    result = objective.evaluate(
        {"alpha": 0.7},
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        trial_index=3,
        sample_ids=["s1", "s2"],
        groups=["g", "g"],
        metadata={"s1": {"batch": "a"}, "s2": {"batch": "a"}},
    )

    assert estimator.alpha == 1.0
    assert result.trial_index == 3
    assert result.params == {"alpha": 0.7}
    assert result.score == pytest.approx(1.2)
    assert result.metric == "rmse"
    assert result.direction == "minimize"
    assert len(result.tuning_fingerprint) == 64
    assert result.diagnostics["engine"] == "n4m"


def test_pipeline_objective_can_reuse_estimator_when_explicitly_requested() -> None:
    estimator = _ObjectiveEstimator(alpha=1.0)
    objective = PipelineObjective(
        estimator,
        {"engine": "optuna", "space": {"alpha": [0.2, 0.7]}, "direction": "maximize", "metric": "accuracy"},
        score_extractor=lambda fitted: fitted.training_result_["score"],
        clone_estimator=False,
    )

    result = objective.evaluate({"alpha": 0.2}, [[1.0]], [1.0])

    assert estimator.alpha == 0.2
    assert result.score == pytest.approx(0.7)
    assert result.metric == "accuracy"
    assert result.direction == "maximize"


def test_apply_trial_parameter_patch_supports_dotted_attributes() -> None:
    estimator = _ObjectiveEstimator(nested=_Nested())

    apply_trial_parameter_patch(estimator, "nested.depth", 4)

    assert estimator.nested.depth == 4


def test_pipeline_objective_rejects_unknown_parameter_paths_before_fit() -> None:
    estimator = _ObjectiveEstimator(alpha=1.0)
    objective = PipelineObjective(
        estimator,
        {"engine": "n4m", "space": {"missing": [1, 2]}},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    with pytest.raises(ValueError, match="not supported"):
        objective.evaluate({"missing": 2}, [[1.0]], [1.0])


def test_pipeline_objective_rejects_non_finite_scores() -> None:
    objective = PipelineObjective(
        _ObjectiveEstimator(alpha=1.0),
        {"engine": "n4m", "space": {"alpha": [1.0]}},
        score_extractor=lambda _fitted: float("nan"),
    )

    with pytest.raises(ValueError, match="score must be finite"):
        objective.evaluate({"alpha": 1.0}, [[1.0]], [1.0])


def test_prediction_score_extractor_scores_fitted_estimator_predictions() -> None:
    objective = PipelineObjective(
        _ObjectiveEstimator(alpha=1.0),
        {"engine": "optuna", "space": {"alpha": [0.9, 0.2]}, "sampler": "grid", "n_trials": 2},
        score_extractor=make_prediction_score_extractor(
            "rmse",
            np.asarray([[10.0], [20.0]]),
            np.asarray([0.0, 0.0]),
        ),
    )

    result = objective.evaluate({"alpha": 0.2}, np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0]))

    assert result.score == pytest.approx(0.2)
    assert result.metric == "rmse"


def test_conformal_prediction_score_extractor_scores_temporary_dev_calibrator() -> None:
    objective = PipelineObjective(
        _ObjectiveEstimator(alpha=1.0),
        {
            "direction": "minimize",
            "engine": "optuna",
            "metric": "conformal_mean_width",
            "n_trials": 2,
            "sampler": "grid",
            "space": {"alpha": [0.9, 0.2]},
        },
        score_extractor=make_conformal_prediction_score_extractor(
            "conformal_mean_width",
            np.asarray([[10.0], [20.0]]),
            np.asarray([0.0, 0.0]),
            {
                "X": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
                "y_true": np.asarray([0.0, 0.0, 0.0, 0.0]),
            },
            coverage=0.8,
        ),
    )

    wider = objective.evaluate({"alpha": 0.9}, np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0]))
    narrower = objective.evaluate({"alpha": 0.2}, np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0]))

    assert wider.score == pytest.approx(1.8)
    assert narrower.score == pytest.approx(0.4)
    assert narrower.score < wider.score
    assert narrower.metric == "conformal_mean_width"
    assert narrower.diagnostics["score_family"] == "conformal"
    assert narrower.diagnostics["score_extractor"] == "conformal_temporary_calibration"
    assert narrower.diagnostics["final_calibration_scope"] == "unmodified_by_score_data"


def test_prediction_score_extractor_rejects_invalid_score_cohorts() -> None:
    with pytest.raises(ValueError, match="X_score contains 2 samples"):
        make_prediction_score_extractor("rmse", [[1.0], [2.0]], [1.0])

    objective = PipelineObjective(
        _NoPredictEstimator(),
        {"engine": "optuna", "space": {"alpha": [0.2]}},
        score_extractor=make_prediction_score_extractor("rmse", [[1.0]], [1.0]),
        clone_estimator=False,
    )

    with pytest.raises(TypeError, match="predict"):
        objective.evaluate({}, [[1.0]], [1.0])


def test_pipeline_objective_refits_exactly_one_terminal_winner() -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.2, 0.7]}})
    objective = PipelineObjective(
        _ObjectiveEstimator(alpha=1.0),
        tuning,
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.7},
        best_value=1.2,
        trials=(),
        optimizer="optuna",
    )

    fitted = objective.refit_best(
        result,
        [[1.0], [2.0]],
        [1.0, 2.0],
        sample_ids=["s1", "s2"],
        groups=["g", "g"],
        metadata={"s1": {"batch": "a"}, "s2": {"batch": "a"}},
    )

    assert fitted.alpha == 0.7
    assert fitted.fit_sample_ids_ == ["s1", "s2"]
    assert fitted.fit_groups_ == ["g", "g"]
    assert fitted.fit_metadata_ == {"s1": {"batch": "a"}, "s2": {"batch": "a"}}


def test_compile_pipeline_objective_accepts_single_model_step() -> None:
    compiled = compile_pipeline_objective(
        [{"model": _ObjectiveEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {"engine": "optuna", "space": {"alpha": [0.2, 0.7]}, "n_trials": 2, "sampler": "grid"},
        score_extractor=lambda fitted: fitted.training_result_["score"],
        sample_ids=["s1", "s2"],
        groups=["g", "g"],
        metadata={"s1": {"batch": "a"}, "s2": {"batch": "a"}},
    )

    assert isinstance(compiled, CompiledPipelineObjective)
    result = compiled.evaluate({"alpha": 0.2}, trial_index=4)

    assert result.trial_index == 4
    assert result.params == {"alpha": 0.2}
    assert result.score == pytest.approx(0.7)
    assert result.diagnostics["engine"] == "optuna"
    assert "search_space_fingerprint" in result.diagnostics
    assert compiled.search_space.paths == ("alpha",)


def test_compile_pipeline_objective_accepts_direct_estimator() -> None:
    compiled = compile_pipeline_objective(
        _ObjectiveEstimator(alpha=1.0),
        [[1.0]],
        [1.0],
        {"engine": "n4m", "space": {"alpha": [0.3]}},
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )

    assert compiled.evaluate({"alpha": 0.3}).score == pytest.approx(0.8)


def test_compile_pipeline_objective_accepts_sklearn_pipeline_dotted_params() -> None:
    compiled = compile_pipeline_objective(
        Pipeline([("scale", _ScaleTransformer(factor=2.0)), ("ridge", _LinearEstimator(alpha=0.0))]),
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.factor": [2.0, 10.0], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.named_steps["ridge"].fit_X_[0, 0] + fitted.named_steps["ridge"].alpha),
    )

    result = compiled.evaluate({"scale.factor": 10.0, "ridge.alpha": 0.2})

    assert result.params == {"scale.factor": 10.0, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(10.2)
    assert [patch.to_dict() for patch in compiled.parameter_patches({"ridge__alpha": 0.2, "scale.factor": 10.0})] == [
        {"path": "ridge.alpha", "segments": ["ridge", "alpha"], "value": 0.2},
        {"path": "scale.factor", "segments": ["scale", "factor"], "value": 10.0},
    ]


def test_compile_pipeline_objective_accepts_sklearn_like_step_tuples() -> None:
    compiled = compile_pipeline_objective(
        [("scale", _ScaleTransformer(factor=2.0)), ("ridge", _LinearEstimator(alpha=0.0))],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.factor": [2.0, 10.0], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"scale.factor": 10.0, "ridge.alpha": 0.2})

    assert result.params == {"scale.factor": 10.0, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(10.2)


def test_compile_pipeline_objective_accepts_linear_passthrough_steps() -> None:
    compiled = compile_pipeline_objective(
        [
            ("identity", "passthrough"),
            {"name": "also_identity", "transform": None},
            ("structured_identity", {"kind": "passthrough"}),
            ("ridge", _LinearEstimator(alpha=0.0)),
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"ridge.alpha": 0.2})

    assert result.params == {"ridge.alpha": 0.2}
    assert result.score == pytest.approx(1.2)


def test_compile_pipeline_objective_rejects_stringified_structured_passthrough_step() -> None:
    with pytest.raises(TypeError, match=r"preprocessing steps must expose transform\(\)"):
        compile_pipeline_objective(
            [
                {"name": "identity", "transform": {"kind": _StringifiesAs("passthrough")}},
                ("ridge", _LinearEstimator(alpha=0.0)),
            ],
            np.asarray([[1.0], [2.0]]),
            np.asarray([1.0, 2.0]),
            {
                "engine": "optuna",
                "space": {"ridge.alpha": [0.2]},
                "sampler": "grid",
                "n_trials": 1,
            },
            score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
        )


def test_compile_pipeline_objective_can_replace_preprocessing_step_with_passthrough() -> None:
    compiled = compile_pipeline_objective(
        [("scale", _ScaleTransformer(factor=10.0)), ("ridge", _LinearEstimator(alpha=0.0))],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale": [{"kind": "passthrough"}], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"scale": {"kind": "passthrough"}, "ridge.alpha": 0.2})

    assert result.params == {"scale": {"kind": "passthrough"}, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(1.2)


def test_compile_pipeline_objective_accepts_direct_named_model_tuple() -> None:
    compiled = compile_pipeline_objective(
        ("ridge", _ObjectiveEstimator(alpha=1.0)),
        [[1.0]],
        [1.0],
        {"engine": "n4m", "space": {"ridge.alpha": [0.3]}},
        score_extractor=lambda fitted: fitted.ridge.training_result_["score"],
    )

    assert compiled.evaluate({"ridge.alpha": 0.3}).score == pytest.approx(0.8)


def test_compile_pipeline_objective_accepts_linear_transformer_chain() -> None:
    compiled = compile_pipeline_objective(
        [
            {"name": "scale", "transform": _ScaleTransformer(factor=2.0)},
            {"model": _LinearEstimator(alpha=0.0)},
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.factor": [2.0, 10.0], "model.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.model.fit_X_[0, 0] + fitted.model.alpha),
    )

    result = compiled.evaluate({"scale.factor": 10.0, "model.alpha": 0.2})

    assert result.params == {"scale.factor": 10.0, "model.alpha": 0.2}
    assert result.score == pytest.approx(10.2)


def test_compile_pipeline_objective_accepts_public_steps_mapping() -> None:
    compiled = compile_pipeline_objective(
        {
            "name": "mapped-linear",
            "steps": [
                {"name": "scale", "transform": _ScaleTransformer(factor=2.0)},
                {"name": "ridge", "model": _LinearEstimator(alpha=0.0)},
            ],
        },
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.factor": [2.0, 10.0], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"scale.factor": 10.0, "ridge.alpha": 0.2})

    assert result.params == {"scale.factor": 10.0, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(10.2)


def test_compile_pipeline_objective_accepts_public_pipeline_mapping_alias() -> None:
    compiled = compile_pipeline_objective(
        {
            "name": "pipeline-wrapper-linear",
            "pipeline": [
                {"name": "scale", "transform": _ScaleTransformer(factor=2.0)},
                {"name": "ridge", "model": _LinearEstimator(alpha=0.0)},
            ],
        },
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.factor": [2.0, 10.0], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"scale.factor": 10.0, "ridge.alpha": 0.2})

    assert result.params == {"scale.factor": 10.0, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(10.2)


def test_compile_pipeline_objective_accepts_sklearn_class_mapping_steps() -> None:
    compiled = compile_pipeline_objective(
        [
            {"name": "scale", "class": "sklearn.preprocessing.StandardScaler", "params": {"with_mean": False}},
            {"name": "ridge", "model": _LinearEstimator(alpha=0.0)},
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.with_mean": [False], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"scale.with_mean": False, "ridge.alpha": 0.2})

    assert result.params == {"scale.with_mean": False, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(2.2)


def test_compile_pipeline_objective_accepts_explicit_sklearn_string_steps() -> None:
    compiled = compile_pipeline_objective(
        [
            ("scale", "sklearn.preprocessing.StandardScaler"),
            ("ridge", "sklearn.linear_model.Ridge"),
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.alpha),
    )

    result = compiled.evaluate({"ridge.alpha": 0.2})

    assert result.params == {"ridge.alpha": 0.2}
    assert result.score == pytest.approx(0.2)


def test_compile_pipeline_objective_accepts_direct_sklearn_string_model() -> None:
    compiled = compile_pipeline_objective(
        "sklearn.dummy.DummyRegressor",
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"strategy": ["mean"]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.constant_[0][0]),
    )

    result = compiled.evaluate({"strategy": "mean"})

    assert result.params == {"strategy": "mean"}
    assert result.score == pytest.approx(3.0)


def test_compile_pipeline_objective_accepts_sklearn_string_mapping_values() -> None:
    compiled = compile_pipeline_objective(
        [
            {"name": "scale", "transform": "sklearn.preprocessing.StandardScaler"},
            {"name": "dummy", "model": "sklearn.dummy.DummyRegressor"},
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"dummy.strategy": ["mean"]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.dummy.constant_[0][0]),
    )

    result = compiled.evaluate({"dummy.strategy": "mean"})

    assert result.params == {"dummy.strategy": "mean"}
    assert result.score == pytest.approx(3.0)


def test_compile_pipeline_objective_accepts_sklearn_string_model_mapping_with_params() -> None:
    compiled = compile_pipeline_objective(
        [
            {
                "name": "ridge",
                "model": "sklearn.linear_model.Ridge",
                "params": {"alpha": 0.3, "fit_intercept": False},
            },
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.alpha + fitted.ridge.coef_[0]),
    )

    result = compiled.evaluate({"ridge.alpha": 0.0})

    assert result.params == {"ridge.alpha": 0.0}
    assert result.score == pytest.approx(2.0)


def test_compile_pipeline_objective_accepts_direct_sklearn_string_model_mapping_with_params() -> None:
    compiled = compile_pipeline_objective(
        {
            "name": "ridge",
            "model": "sklearn.linear_model.Ridge",
            "params": {"alpha": 0.3, "fit_intercept": False},
        },
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.alpha + fitted.ridge.coef_[0]),
    )

    result = compiled.evaluate({"ridge.alpha": 0.0})

    assert result.params == {"ridge.alpha": 0.0}
    assert result.score == pytest.approx(2.0)


def test_compile_pipeline_objective_accepts_sklearn_string_transform_mapping_with_params() -> None:
    compiled = compile_pipeline_objective(
        [
            {
                "name": "scale",
                "transform": "sklearn.preprocessing.StandardScaler",
                "params": {"with_mean": False},
            },
            {"name": "ridge", "model": _LinearEstimator(alpha=0.0)},
        ],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"scale.with_mean": [False], "ridge.alpha": [0.2]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.fit_X_[0, 0] + fitted.ridge.alpha),
    )

    result = compiled.evaluate({"scale.with_mean": False, "ridge.alpha": 0.2})

    assert result.params == {"scale.with_mean": False, "ridge.alpha": 0.2}
    assert result.score == pytest.approx(2.2)


def test_compile_pipeline_objective_accepts_declarative_sklearn_class_model_step() -> None:
    compiled = compile_pipeline_objective(
        {
            "steps": [
                {"name": "scale", "class": "sklearn.preprocessing.StandardScaler", "params": {"with_mean": False}},
                {"name": "ridge", "class": "sklearn.linear_model.Ridge", "params": {"fit_intercept": False}},
            ]
        },
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"scale.with_mean": [False], "ridge.alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.ridge.alpha + fitted.ridge.coef_[0]),
    )

    result = compiled.evaluate({"scale.with_mean": False, "ridge.alpha": 0.0})

    assert result.params == {"scale.with_mean": False, "ridge.alpha": 0.0}
    assert result.score == pytest.approx(1.0)


def test_compile_pipeline_objective_accepts_direct_declarative_sklearn_class_model() -> None:
    compiled = compile_pipeline_objective(
        {"class": "sklearn.linear_model.Ridge", "params": {"fit_intercept": False}},
        np.asarray([[1.0], [2.0]]),
        np.asarray([2.0, 4.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.0]},
            "sampler": "grid",
            "n_trials": 1,
        },
        score_extractor=lambda fitted: float(fitted.alpha + fitted.coef_[0]),
    )

    result = compiled.evaluate({"alpha": 0.0})

    assert result.params == {"alpha": 0.0}
    assert result.score == pytest.approx(2.0)


def test_compile_pipeline_objective_accepts_named_single_model_step() -> None:
    compiled = compile_pipeline_objective(
        {"steps": [{"name": "ridge", "model": _ObjectiveEstimator(alpha=1.0)}]},
        [[1.0]],
        [1.0],
        {"engine": "n4m", "space": {"ridge.alpha": [0.3]}},
        score_extractor=lambda fitted: fitted.ridge.training_result_["score"],
    )

    assert compiled.evaluate({"ridge.alpha": 0.3}).score == pytest.approx(0.8)


def test_compile_pipeline_objective_rejects_broader_pipeline_shapes() -> None:
    with pytest.raises(ValueError, match="either 'steps' or 'pipeline', not both"):
        compile_pipeline_objective(
            {
                "steps": [{"model": _ObjectiveEstimator()}],
                "pipeline": [{"model": _ObjectiveEstimator()}],
            },
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="pipeline mappings support only"):
        compile_pipeline_objective(
            {
                "steps": [{"model": _ObjectiveEstimator()}],
                "branch": [],
            },
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="preprocessing step supports only"):
        compile_pipeline_objective(
            [{"branch": []}, {"model": _ObjectiveEstimator()}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match=r"only explicit sklearn\.\* import paths"):
        compile_pipeline_objective(
            [{"name": "scale", "class": "nirs4all.some.Transform"}, {"model": _ObjectiveEstimator()}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(TypeError, match="params must be a mapping"):
        compile_pipeline_objective(
            [{"name": "scale", "class": "sklearn.preprocessing.StandardScaler", "params": ["not", "a", "mapping"]}, {"model": _ObjectiveEstimator()}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(ValueError, match="params keys must be canonical non-empty strings"):
        compile_pipeline_objective(
            [{"name": "scale", "transform": "sklearn.preprocessing.StandardScaler", "params": {" with_mean ": False}}, {"model": _ObjectiveEstimator()}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(ValueError, match=r"params\[bad\] must contain TCV1-compatible JSON-native values"):
        compile_pipeline_objective(
            [{"name": "ridge", "model": "sklearn.linear_model.Ridge", "params": {"bad": object()}}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="estimator_or_sklearn_path"):
        compile_pipeline_objective(
            [{"model": _ObjectiveEstimator(), "finetune_params": {"model_params": {"alpha": [0.1]}}}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="params are supported only with explicit sklearn"):
        compile_pipeline_objective(
            [{"model": _ObjectiveEstimator(), "params": {"alpha": 0.1}}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="preprocessing step params are supported only with explicit sklearn"):
        compile_pipeline_objective(
            [{"name": "scale", "transform": _ScaleTransformer(), "params": {"factor": 2.0}}, {"model": _ObjectiveEstimator()}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(ValueError, match="step names must be unique"):
        compile_pipeline_objective(
            [
                {"name": "scale", "transform": _ScaleTransformer()},
                {"name": "scale", "transform": _ScaleTransformer()},
                {"model": _ObjectiveEstimator()},
            ],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="string model aliases"):
        compile_pipeline_objective(
            [{"model": "PLSRegression"}],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    with pytest.raises(NotImplementedError, match="string model aliases"):
        compile_pipeline_objective(
            [("model", "passthrough")],
            [[1.0]],
            [1.0],
            {"engine": "n4m", "space": {"alpha": [0.3]}},
            score_extractor=lambda fitted: fitted.training_result_["score"],
        )

    compiled = compile_pipeline_objective(
        [("scale", _ScaleTransformer()), ("ridge", _LinearEstimator())],
        [[1.0]],
        [1.0],
        DagMLTuningSpec(engine="n4m", space={"ridge": ["replacement-sentinel"]}),
        score_extractor=lambda fitted: fitted.training_result_["score"],
    )
    with pytest.raises(ValueError, match="TCV1-compatible JSON-native value"):
        compiled.evaluate({"ridge": _LinearEstimator(alpha=0.3)})


def test_run_single_estimator_tuning_to_run_result_compiles_tunes_refits_and_projects_winner(tmp_path) -> None:
    result = run_single_estimator_tuning_to_run_result(
        [{"model": _ObjectiveEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        score_extractor=make_prediction_score_extractor(
            "rmse",
            np.asarray([[10.0], [20.0]]),
            np.asarray([0.0, 0.0]),
        ),
        sample_ids=["train-a", "train-b"],
        workspace_path=tmp_path / "workspace",
        workspace_tuning_id="single-estimator-tune",
        winner_x=np.asarray([[10.0], [20.0]]),
        winner_y_true=np.asarray([0.0, 0.0]),
        winner_score=0.2,
        winner_metric="rmse",
        winner_sample_ids=["test-a", "test-b"],
        winner_dataset_name="external",
        winner_model_name="CompiledWinner",
        per_dataset={"external": {"role": "winner_projection"}},
    )

    assert result.tuning_id == "single-estimator-tune"
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.2)
    assert result.per_dataset == {"external": {"role": "winner_projection"}}
    assert result.num_predictions == 1
    assert result.best["model_name"] == "CompiledWinner"
    assert result.best_score == pytest.approx(0.2)
    [entry] = result.filter(fold_id="final")
    np.testing.assert_allclose(entry["y_pred"], [0.2, 0.2])
    assert entry["metadata"]["physical_sample_id"] == ["test-a", "test-b"]


def test_run_single_estimator_tuning_to_run_result_keeps_tuning_only_projection_when_no_winner() -> None:
    result = run_single_estimator_tuning_to_run_result(
        _ObjectiveEstimator(alpha=1.0),
        [[1.0], [2.0]],
        [1.0, 2.0],
        {"engine": "optuna", "space": {"alpha": [0.9, 0.2]}, "sampler": "grid", "n_trials": 2},
        score_extractor=lambda fitted: fitted.training_result_["score"],
        refit=False,
    )

    assert result.num_predictions == 0
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.validate(raise_on_failure=False)["valid"] is True
