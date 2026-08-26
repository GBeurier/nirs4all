"""sklearn-cloneable DAG-ML training estimator seam for nirs4all.

``NIRSPipeline`` is a prediction-only wrapper around already-trained nirs4all
bundles.  ``DagMLPipelineEstimator`` is the separate W2-PY training estimator:
it owns no compilation logic itself, but it can call native DAG-ML training and
replay once the nirs4all→DAG-ML contract compiler is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .fit_identity import (
    DagMLFitIdentityFrame,
    DagMLPredictIdentityFrame,
    normalize_fit_identity,
    normalize_predict_identity,
)
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
    # The native Methods lane owns its numeric provider in dag-ml.  Keeping
    # this optional prevents the generic host-callback execution route from
    # gaining a second, implicit meaning.
    methods_inputs: Any = None
    methods_library_path: str | None = None


@dataclass(frozen=True)
class DagMLReplayExecution:
    """Already-compiled native loaded-package replay call inputs."""

    request: Any
    data_envelopes: Any
    artifact_handles: Any
    op_callback: Any
    outcome_id: str
    run_id: str
    artifact_callback: Any = None
    cleanup: Any = None
    warnings: Any = ()
    diagnostics: Any = None
    methods_inputs: Any = None
    methods_library_path: str | None = None


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
        identity_frame: DagMLPredictIdentityFrame,
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
        prediction_identity_decoder: Any = None,
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
        self.prediction_identity_decoder = prediction_identity_decoder
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
        if execution.methods_inputs is not None or execution.methods_library_path is not None:
            if execution.methods_inputs is None or execution.methods_library_path is None:
                raise DagMLNativeCoverageError(
                    "native Methods training requires both methods inputs and an explicit libn4m path"
                )
            training_result = client.execute_methods_training(
                execution.request,
                execution.data_envelopes,
                execution.relations,
                execution.training_influence,
                execution.methods_inputs,
                methods_library_path=execution.methods_library_path,
                outcome_id=execution.outcome_id,
                run_id=execution.run_id,
                bundle_id=execution.bundle_id,
                warnings=execution.warnings,
                diagnostics=execution.diagnostics,
            )
        else:
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
        # Retain the exact signed contracts that produced this attached native
        # result. A later portable full-refit operation needs the parent
        # request only as provenance evidence; it must never reconstruct it
        # from a Python pipeline or rerun CV/SELECT.
        self.native_training_execution_ = execution
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
        replay_outcome, identity_frame = self._execute_replay_with_identity_frame(
            X,
            mode="predict",
        )
        if self.prediction_identity_decoder is not None:
            return cast(
                np.ndarray,
                np.asarray(self.prediction_identity_decoder(replay_outcome, identity_frame)),
            )
        if self.prediction_decoder is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator.predict() requires a native prediction decoder; P1 does not synthesize Python predictions from replay JSON")
        return cast(np.ndarray, np.asarray(self.prediction_decoder(replay_outcome)))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via native loaded-package replay.

        This method never creates one-hot pseudo-probabilities.  A native replay
        decoder for probability outputs must be supplied explicitly.
        """

        check_is_fitted(self, attributes=["training_result_", "output_binding_"])
        replay_outcome = self._execute_replay_with_identity(X, mode="predict_proba")
        if self.probability_decoder is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator.predict_proba() requires an explicit native probability decoder; pseudo-probabilities are forbidden")
        return cast(np.ndarray, np.asarray(self.probability_decoder(replay_outcome)))

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

    def predict_with_identity(
        self,
        X: Any,
        *,
        sample_ids: Any,
        groups: Any = None,
        metadata: Any = None,
    ) -> np.ndarray:
        """Predict through native replay with an explicit X-only identity frame.

        Persisted or relation-sensitive packages should use this method rather
        than sklearn's positional ``predict(X)`` surface: it gives the replay
        compiler the exact row identities it must bind into the target-free
        PREDICT envelope.  The method never accepts or creates targets.
        """

        check_is_fitted(self, attributes=["training_result_", "output_binding_"])
        replay_outcome, identity_frame = self._execute_replay_with_identity_frame(
            X,
            mode="predict",
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=True,
        )
        if self.prediction_identity_decoder is not None:
            return cast(
                np.ndarray,
                np.asarray(self.prediction_identity_decoder(replay_outcome, identity_frame)),
            )
        if self.prediction_decoder is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator.predict_with_identity() requires a native prediction decoder; P1 does not synthesize Python predictions from replay JSON")
        return cast(np.ndarray, np.asarray(self.prediction_decoder(replay_outcome)))

    def replay_with_identity(
        self,
        X: Any,
        *,
        sample_ids: Any,
        groups: Any = None,
        metadata: Any = None,
    ) -> Any:
        """Return the native PREDICT replay for an explicitly identified cohort.

        This is the evidence-preserving counterpart of
        :meth:`predict_with_identity`.  It intentionally returns the DAG-ML
        replay contract rather than decoding it into a NumPy array, so callers
        such as the conformal attachment lane can bind the exact replay to
        identity-keyed truth in DAG-ML.  It never supplies targets, computes
        provenance, or invokes a legacy runner.
        """

        check_is_fitted(self, attributes=["training_result_", "output_binding_"])
        identity_frame = normalize_predict_identity(
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=True,
        )
        replay = self._compile_replay(X, mode="predict", identity_frame=identity_frame)
        return self.execute_compiled_replay(replay)

    def execute_compiled_replay(self, replay: DagMLReplayExecution) -> Any:
        """Execute one already-compiled native replay exactly once.

        Compilation owns request, relation and identity attestation.  This
        method only transports that exact contract to DAG-ML and guarantees
        the compiler's cleanup hook executes on both success and failure.
        It is public so higher-level native operations can retain the same
        replay evidence for a later DAG-ML attachment rather than compiling a
        second, potentially distinct request.
        """

        check_is_fitted(self, attributes=["training_result_", "output_binding_"])
        if not isinstance(replay, DagMLReplayExecution):
            raise TypeError("execute_compiled_replay requires DagMLReplayExecution")
        if self.predictor_package_ is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator has no portable predictor package to replay")
        try:
            if replay.methods_inputs is not None or replay.methods_library_path is not None:
                if replay.methods_inputs is None or replay.methods_library_path is None:
                    raise DagMLNativeCoverageError(
                        "native Methods replay requires inputs and an explicit libn4m path"
                    )
                return self._client().replay_loaded_methods_predictor_package(
                    self.predictor_package_,
                    replay.request,
                    replay.data_envelopes,
                    replay.methods_inputs,
                    methods_library_path=replay.methods_library_path,
                    outcome_id=replay.outcome_id,
                    run_id=replay.run_id,
                    warnings=replay.warnings,
                    diagnostics=replay.diagnostics,
                )
            return self._client().replay_loaded_predictor_package(
                self.predictor_package_,
                replay.request,
                replay.data_envelopes,
                replay.artifact_handles,
                replay.op_callback,
                outcome_id=replay.outcome_id,
                run_id=replay.run_id,
                artifact_callback=replay.artifact_callback,
                warnings=replay.warnings,
                diagnostics=replay.diagnostics,
            )
        finally:
            if replay.cleanup is not None:
                replay.cleanup()

    def export_native_archive(
        self,
        archive_path: str | Path,
        *,
        archive_id: str,
    ) -> dict[str, str]:
        """Write this fitted native predictor as a portable Archive V2.

        The archive consists solely of the exact native ``TrainingOutcome``
        and Package V2 emitted by DAG-ML.  Assembly remains owned by DAG-ML
        and ZIP persistence by Core; this method neither serializes a Python
        estimator nor retrains through :class:`PipelineRunner`.

        Args:
            archive_path: New ``.n4a`` Archive V2 destination.
            archive_id: Explicit stable archive identity for the closed
                DAG-ML/Core archive contract.

        Returns:
            The Core-issued archive id and SHA-256 reference.

        Raises:
            DagMLNativeCoverageError: If fit did not retain an exportable
                portable predictor package.
        """

        check_is_fitted(self, attributes=["training_outcome_", "predictor_package_"])
        if not isinstance(archive_id, str) or not archive_id.strip():
            raise ValueError("archive_id must be a non-empty string")
        if self.predictor_package_ is None:
            raise DagMLNativeCoverageError("native training did not retain a portable predictor package for Archive V2 export")
        from .native_archive_replay import write_methods_archive_v2

        return cast(
            dict[str, str],
            write_methods_archive_v2(
                archive_path,
                archive_id=archive_id,
                outcome=self.training_outcome_,
                package=self.predictor_package_,
            ),
        )

    def _execute_replay_with_identity(
        self,
        X: Any,
        *,
        mode: str,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        require_explicit_sample_ids: bool | None = None,
    ) -> Any:
        """Run replay and return its native outcome (legacy private helper)."""

        outcome, _identity_frame = self._execute_replay_with_identity_frame(
            X,
            mode=mode,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=require_explicit_sample_ids,
        )
        return outcome

    def _execute_replay_with_identity_frame(
        self,
        X: Any,
        *,
        mode: str,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        require_explicit_sample_ids: bool | None = None,
    ) -> tuple[Any, DagMLPredictIdentityFrame]:
        """Run replay and retain the exact current identity frame for decoding.

        A decoder that needs to verify output ordering (for example the raw
        Methods path) receives this frame rather than reconstructing identities
        from replay JSON.  The older outcome-only helper remains above for
        existing decoders.
        """
        if self.prediction_compiler is None:
            raise DagMLNativeCoverageError(f"DagMLPipelineEstimator.{mode}() requires the nirs4all→DAG-ML loaded-package replay compiler")
        if self.predictor_package_ is None:
            raise DagMLNativeCoverageError("DagMLPipelineEstimator has no portable predictor package to replay")

        identity_frame = normalize_predict_identity(
            X,
            sample_ids=sample_ids,
            groups=groups,
            metadata=metadata,
            require_explicit_sample_ids=(self.require_explicit_sample_ids if require_explicit_sample_ids is None else require_explicit_sample_ids),
        )
        replay = self._compile_replay(X, mode=mode, identity_frame=identity_frame)
        outcome = self.execute_compiled_replay(replay)
        return outcome, identity_frame

    def _compile_replay(
        self,
        X: Any,
        *,
        mode: str,
        identity_frame: DagMLPredictIdentityFrame,
    ) -> DagMLReplayExecution:
        compiler = self.prediction_compiler
        if hasattr(compiler, "compile_replay"):
            replay = compiler.compile_replay(
                self,
                X,
                mode=mode,
                identity_frame=identity_frame,
            )
        elif callable(compiler):
            replay = compiler(self, X, mode=mode, identity_frame=identity_frame)
        else:
            raise TypeError("prediction_compiler must be callable or expose compile_replay()")
        if not isinstance(replay, DagMLReplayExecution):
            raise TypeError("prediction_compiler must return DagMLReplayExecution")
        return replay

    def _client(self) -> Any:
        return self.native_client if self.native_client is not None else DagMLNativeClient(self.dagml_module)

    def native_runtime_client(self) -> DagMLNativeClient:
        """Return the exact client selected for this attached native result."""

        return cast(DagMLNativeClient, self._client())

    def _select_output_binding(self, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        if self.selection_output_id is not None:
            for output in outputs:
                if self._output_binding_id(output) == self.selection_output_id:
                    return output
            raise DagMLNativeCoverageError(f"native training output '{self.selection_output_id}' was not produced")
        if len(outputs) == 1:
            return outputs[0]
        raise DagMLNativeCoverageError("native training produced ambiguous outputs; set selection_output_id explicitly")

    @staticmethod
    def _output_binding_id(output: dict[str, Any]) -> str | None:
        """Read the output id from both legacy and Package V2 result shapes.

        The native 0.3.4 binding returns an output wrapper with its stable id
        at ``binding.binding_id``.  Test doubles and older bindings used the
        flattened ``output_id`` form.  They express the same selected output;
        accepting both avoids silently choosing by position.
        """

        output_id = output.get("output_id")
        if isinstance(output_id, str):
            return output_id
        binding = output.get("binding")
        if isinstance(binding, dict):
            binding_id = binding.get("binding_id")
            if isinstance(binding_id, str):
                return binding_id
        return None

    def _export_predictor_package(self, training_result: Any, execution: DagMLTrainingExecution) -> Any:
        export_package = getattr(training_result, "export_portable_predictor_package", None)
        if not callable(export_package):
            return None
        package_id = self.package_id or f"{execution.outcome_id}-predictor"
        if execution.methods_inputs is not None:
            return export_package(
                package_id,
                fitted_artifact_mode="portable_required",
                artifact_load_mode="native_portable",
            )
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
    "DagMLPredictIdentityFrame",
    "DagMLReplayCompiler",
    "DagMLReplayExecution",
    "DagMLTrainingCompiler",
    "DagMLTrainingExecution",
]
