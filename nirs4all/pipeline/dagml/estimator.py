"""sklearn-cloneable DAG-ML training estimator seam for nirs4all.

``NIRSPipeline`` is a prediction-only wrapper around already-trained nirs4all
bundles.  ``DagMLPipelineEstimator`` is the separate W2-PY training estimator:
it owns no compilation logic itself, but it can call native DAG-ML training and
replay once the nirs4all→DAG-ML contract compiler is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .fit_identity import DagMLFitIdentityFrame, normalize_fit_identity
from .native_client import DagMLNativeClient, DagMLNativeCoverageError


@dataclass(frozen=True)
class DagMLTrainingExecution:
    """Already-compiled native training call inputs.

    These fields map one-to-one to ``dag_ml.execute_training(...)``.  The class
    intentionally carries contracts and callbacks as opaque values; validation
    remains owned by DAG-ML.
    """

    request: Any
    data_envelopes: Any
    relations: Any
    training_influence: Any
    op_callback: Any
    outcome_id: str
    run_id: str
    bundle_id: str
    warnings: Any = ()
    diagnostics: Any = None


@dataclass(frozen=True)
class DagMLReplayExecution:
    """Already-compiled native loaded-package replay call inputs."""

    request: Any
    data_envelopes: Any
    artifact_handles: Any
    op_callback: Any
    outcome_id: str
    run_id: str
    warnings: Any = ()
    diagnostics: Any = None


class DagMLTrainingCompiler(Protocol):
    """Protocol for future nirs4all→DAG-ML fit contract compilers."""

    def compile_fit(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: DagMLFitIdentityFrame,
    ) -> DagMLTrainingExecution: ...


class DagMLReplayCompiler(Protocol):
    """Protocol for future nirs4all→DAG-ML predict/proba replay compilers."""

    def compile_replay(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        *,
        mode: str,
    ) -> DagMLReplayExecution: ...


class DagMLPipelineEstimator(BaseEstimator):
    """sklearn-compatible estimator backed by native DAG-ML training contracts.

    The estimator is intentionally constructor-only and cloneable. It does not
    compile nirs4all syntax yet; callers must provide a ``training_compiler``
    and, for prediction, a ``prediction_compiler`` plus explicit decoders.
    Missing pieces raise :class:`DagMLNativeCoverageError` instead of falling
    back or fabricating predictions.
    """

    def __init__(
        self,
        pipeline: Any = None,
        *,
        task_type: str = "auto",
        selection_output_id: str | None = None,
        package_id: str | None = None,
        dagml_module: str = "dag_ml",
        native_client: Any = None,
        training_compiler: Any = None,
        prediction_compiler: Any = None,
        prediction_decoder: Any = None,
        probability_decoder: Any = None,
        require_explicit_sample_ids: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.task_type = task_type
        self.selection_output_id = selection_output_id
        self.package_id = package_id
        self.dagml_module = dagml_module
        self.native_client = native_client
        self.training_compiler = training_compiler
        self.prediction_compiler = prediction_compiler
        self.prediction_decoder = prediction_decoder
        self.probability_decoder = probability_decoder
        self.require_explicit_sample_ids = require_explicit_sample_ids

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> DagMLPipelineEstimator:
        """Fit the pipeline through native DAG-ML training contracts."""

        identity_frame = normalize_fit_identity(
            X,
            y,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=self.require_explicit_sample_ids,
        )
        execution = self._compile_fit(
            X,
            y,
            sample_ids=identity_frame.sample_ids,
            groups=identity_frame.groups,
            metadata=identity_frame.metadata_by_sample_id(),
            identity_frame=identity_frame,
        )
        client = self._client()
        training_result = client.execute_training(
            execution.request,
            execution.data_envelopes,
            execution.relations,
            execution.training_influence,
            execution.op_callback,
            outcome_id=execution.outcome_id,
            run_id=execution.run_id,
            bundle_id=execution.bundle_id,
            warnings=execution.warnings,
            diagnostics=execution.diagnostics,
        )

        self.training_result_ = training_result
        self.training_outcome_ = getattr(training_result, "outcome", None)
        self.outputs_ = list(getattr(training_result, "outputs", []) or [])
        self.output_binding_ = self._select_output_binding(self.outputs_)
        self.predictor_package_ = self._export_predictor_package(
            training_result,
            execution,
        )
        self.fit_identity_frame_ = identity_frame
        self.n_features_in_ = self._infer_n_features(X)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict via native loaded-package replay.

        No Python pseudo-prediction path exists here.  Until the replay compiler
        and decoder are supplied, this method fails with a typed coverage error.
        """

        check_is_fitted(self, attributes=["training_result_", "output_binding_"])
        replay_outcome = self._execute_replay(X, mode="predict")
        if self.prediction_decoder is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator.predict() requires a native prediction decoder; P1 does not synthesize Python predictions from replay JSON")
        return np.asarray(self.prediction_decoder(replay_outcome))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via native loaded-package replay.

        This method never creates one-hot pseudo-probabilities.  A native replay
        decoder for probability outputs must be supplied explicitly.
        """

        check_is_fitted(self, attributes=["training_result_", "output_binding_"])
        replay_outcome = self._execute_replay(X, mode="predict_proba")
        if self.probability_decoder is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator.predict_proba() requires an explicit native probability decoder; pseudo-probabilities are forbidden")
        return np.asarray(self.probability_decoder(replay_outcome))

    def _compile_fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: DagMLFitIdentityFrame,
    ) -> DagMLTrainingExecution:
        compiler = self.training_compiler
        if compiler is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator.fit() requires the nirs4all→DAG-ML training contract compiler from W2-PY P3")
        if hasattr(compiler, "compile_fit"):
            execution = compiler.compile_fit(
                self,
                X,
                y,
                sample_ids=sample_ids,
                groups=groups,
                metadata=metadata,
                identity_frame=identity_frame,
            )
        elif callable(compiler):
            execution = compiler(
                self,
                X,
                y,
                sample_ids=sample_ids,
                groups=groups,
                metadata=metadata,
                identity_frame=identity_frame,
            )
        else:
            raise TypeError("training_compiler must be callable or expose compile_fit()")
        if not isinstance(execution, DagMLTrainingExecution):
            raise TypeError("training_compiler must return DagMLTrainingExecution")
        return execution

    def _execute_replay(self, X: Any, *, mode: str) -> Any:
        if self.prediction_compiler is None:
            raise DagMLNativeCoverageError(f"DagMLPipelineEstimator.{mode}() requires the nirs4all→DAG-ML loaded-package replay compiler")
        if self.predictor_package_ is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator has no portable predictor package to replay")

        replay = self._compile_replay(X, mode=mode)
        return self._client().replay_loaded_predictor_package(
            self.predictor_package_,
            replay.request,
            replay.data_envelopes,
            replay.artifact_handles,
            replay.op_callback,
            outcome_id=replay.outcome_id,
            run_id=replay.run_id,
            warnings=replay.warnings,
            diagnostics=replay.diagnostics,
        )

    def _compile_replay(self, X: Any, *, mode: str) -> DagMLReplayExecution:
        compiler = self.prediction_compiler
        if hasattr(compiler, "compile_replay"):
            replay = compiler.compile_replay(self, X, mode=mode)
        elif callable(compiler):
            replay = compiler(self, X, mode=mode)
        else:
            raise TypeError("prediction_compiler must be callable or expose compile_replay()")
        if not isinstance(replay, DagMLReplayExecution):
            raise TypeError("prediction_compiler must return DagMLReplayExecution")
        return replay

    def _client(self) -> Any:
        return self.native_client if self.native_client is not None else DagMLNativeClient(self.dagml_module)

    def _select_output_binding(self, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        if self.selection_output_id is not None:
            for output in outputs:
                if output.get("output_id") == self.selection_output_id:
                    return output
            raise DagMLNativeCoverageError(f"native training output '{self.selection_output_id}' was not produced")
        if len(outputs) == 1:
            return outputs[0]
        raise DagMLNativeCoverageError("native training produced ambiguous outputs; set selection_output_id explicitly")

    def _export_predictor_package(self, training_result: Any, execution: DagMLTrainingExecution) -> Any:
        export_package = getattr(training_result, "export_portable_predictor_package", None)
        if not callable(export_package):
            return None
        package_id = self.package_id or f"{execution.outcome_id}-predictor"
        return export_package(package_id)

    @staticmethod
    def _infer_n_features(X: Any) -> int | None:
        shape = getattr(X, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
        array = np.asarray(X)
        return int(array.shape[1]) if array.ndim >= 2 else None


__all__ = [
    "DagMLPipelineEstimator",
    "DagMLFitIdentityFrame",
    "DagMLReplayCompiler",
    "DagMLReplayExecution",
    "DagMLTrainingCompiler",
    "DagMLTrainingExecution",
]
