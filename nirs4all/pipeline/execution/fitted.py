"""Captured Python-host fitted dependency graphs; replay never fits or opens a workspace."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from nirs4all.pipeline.config.context import ExecutionPhase
from nirs4all.pipeline.execution.preprocessing import PreprocessingReplay


@dataclass(frozen=True, eq=False)
class FittedNode:
    """One immutable dependency in the actual fitted raw-input execution graph."""

    operation: str
    inputs: tuple[FittedNode, ...] = ()
    state: Any = None
    artifact_id: str = ""
    target_transformer: Any = None
    target_processing: str = "numeric"
    input_dtype: Any = None

    def __deepcopy__(self, memo: dict[int, Any]) -> FittedNode:
        """Contexts may share already captured, read-only fitted dependencies."""
        return self

    def evaluate(self, X: np.ndarray, cache: dict[int, np.ndarray]) -> np.ndarray:
        """Evaluate this dependency using only its captured owner states."""
        key = id(self)
        if key in cache:
            return cache[key]
        if self.operation == "input":
            result = np.asarray(X)
            if result.ndim != 2 or result.shape[1] != self.state:
                raise ValueError(f"Fitted predictor expects a two-dimensional raw input with {self.state} features")
            if self.input_dtype is not None:
                result = result.astype(self.input_dtype, copy=False)
        elif self.operation == "transform":
            result = self.state.transform(self.inputs[0].evaluate(X, cache))
        elif self.operation == "model":
            result = np.asarray(self.state.predict(self.inputs[0].evaluate(X, cache)))
            if self.target_transformer is not None and self.target_processing != "numeric":
                shape = result.shape
                values = result.reshape(-1, 1) if result.ndim == 1 else result
                result = self.target_transformer.transform(values, self.target_processing, "numeric", {}).reshape(shape)
        elif self.operation == "probability_features":
            producer = self.inputs[0]
            values = producer.state.predict_proba(producer.inputs[0].evaluate(X, cache))
            result = self.state.extract_features({"y_proba": values}, len(X))
        elif self.operation == "source_mean":
            from nirs4all.controllers.models.stacking.reconstructor import aggregate_test_fold_predictions
            from nirs4all.operators.models.meta import TestAggregation

            # Reconstructor source names may denote several same-named producers.
            # Keep its ordered, float64 accumulation/mean contract, including all
            # exact fitted identities instead of choosing an arbitrary model.
            values = np.asarray([node.evaluate(X, cache) for node in self.inputs], dtype=float)
            result = aggregate_test_fold_predictions(values, np.full(len(values), np.nan), TestAggregation.MEAN, "rmse")
        elif self.operation == "source_folds":
            from nirs4all.controllers.models.stacking.reconstructor import aggregate_test_fold_predictions
            from nirs4all.operators.models.meta import TestAggregation

            values = np.asarray([node.evaluate(X, cache) for node in self.inputs], dtype=float)
            result = aggregate_test_fold_predictions(
                values, np.asarray(self.state["scores"]), TestAggregation(self.state["aggregation"]), self.state["metric"],
            )
        elif self.operation == "concat":
            blocks = [node.evaluate(X, cache) for node in self.inputs]
            result = np.concatenate([block.reshape(len(X), -1) for block in blocks], axis=1)
            if self.state is not None:
                result = result.astype(self.state, copy=False)
        elif self.operation == "aggregate":
            from nirs4all.controllers.shared import PredictionAggregator
            from nirs4all.operators.data.merge import AggregationStrategy

            result = PredictionAggregator.aggregate(
                {name: node.evaluate(X, cache) for name, node in zip(self.state["names"], self.inputs, strict=True)},
                strategy=AggregationStrategy(self.state["strategy"]), model_scores=self.state["scores"],
                proba=self.state["proba"], metric=self.state["metric"],
            )
        else:
            raise ValueError(f"Unsupported fitted dependency operation: {self.operation!r}")
        cache[key] = np.asarray(result)
        return cache[key]


@dataclass
class FittedPredictor:
    """Self-contained Python-host predictor over an actually captured fitted graph."""

    root: FittedNode
    training_steps: list[Any] | None = None
    profile: str = field(default="nirs4all.python.fitted_predictor.v1", init=False)

    def fit(self, X: Any, y: Any = None) -> None:
        """Refuse refitting a captured predictor; replay its original training recipe."""
        raise NotImplementedError("A captured fitted predictor is predict-only; retrain its original training recipe")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict raw rows without fitting, selecting models, or reading a workspace."""
        return self.root.evaluate(np.asarray(X), {})

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Keep the fitted final classifier's probability columns and vocabulary."""
        return cast(np.ndarray, np.asarray(self.root.state.predict_proba(self.root.inputs[0].evaluate(np.asarray(X), {}))))

    @property
    def classes_(self) -> np.ndarray:
        """Expose the exact fitted final classifier column vocabulary."""
        return cast(np.ndarray, np.asarray(self.root.state.classes_))

    @property
    def class_labels_(self) -> np.ndarray:
        """Decode that numeric vocabulary with the captured original target codec."""
        classes = self.classes_
        if self.root.target_transformer is None:
            return classes
        return cast(np.ndarray, self.root.target_transformer.transform(classes.reshape(-1, 1), "numeric", "raw", {}).reshape(-1))


def is_sequential_meta_pipeline(steps: list[Any]) -> bool:
    """Identify the declared sequential MetaModel profile without admitting branches."""
    from nirs4all.operators.models.meta import MetaModel

    if any(isinstance(step, dict) and step.keys() & {"branch", "merge", "merge_sources", "merge_predictions"} for step in steps):
        return False
    for step in steps:
        candidate = step.get("model", step.get("meta_model")) if isinstance(step, dict) else step
        if isinstance(candidate, MetaModel):
            return True
        class_name = candidate.get("class") if isinstance(candidate, dict) else candidate
        if isinstance(class_name, str) and class_name in ("nirs4all.operators.models.meta.MetaModel", "nirs4all.operators.models.MetaModel"):
            return True
    return False


def prepare_sequential_fitted_capture(steps: list[Any], runtime: Any) -> bool:
    """Start a pipeline-local inference capture, independently of CV fitting."""
    from nirs4all.pipeline.config.component_serialization import serialize_component

    if runtime.capture_fitted_predictor or not is_sequential_meta_pipeline(steps):
        return False
    if runtime.fitted_training_steps is None:
        runtime.fitted_training_steps = serialize_component(steps)
    runtime.capture_fitted_predictor = True
    return True


def prepare_fitted_input(dataset: Any, context: Any, runtime: Any) -> FittedNode | None:
    """Start the explicitly captured CV or refit inference dependency graph."""
    if runtime is None or not runtime.capture_fitted_predictor or runtime.phase not in (ExecutionPhase.CV, ExecutionPhase.REFIT):
        return None
    node = context.custom.get("_fitted_input")
    if node is None:
        values = np.asarray(dataset.x(context.selector, layout="2d"))
        node = FittedNode("input", state=values.shape[1], input_dtype=values.dtype)
        context.custom["_fitted_input"] = node
    return cast(FittedNode, node)


def _source_node(predictions: Any, runtime: Any, name: str, branch: int | None, max_step: int,
                 exact_step: int | None = None, *, use_proba: bool = False, test_aggregation: Any = None) -> FittedNode:
    """Resolve the exact fitted producer identity selected during this execution."""
    rows = [
        row for row in predictions.iter_entries()
        if row.get("model_name") == name and row.get("branch_id") == branch and row.get("step_idx", 0) < max_step
        and (exact_step is None or row.get("step_idx") == exact_step)
    ]
    if runtime.phase == ExecutionPhase.CV and test_aggregation is not None:
        from nirs4all.operators.models.meta import TestAggregation

        # The reconstructor's scalar MEAN path consumes the first precomputed
        # average for this source name. Preserve that exact producing step.
        if test_aggregation == TestAggregation.MEAN and not use_proba:
            average = next((row for row in rows if row.get("partition") == "test" and row.get("fold_id") == "avg"), None)
            producer_step = average["step_idx"] if average is not None else rows[0]["step_idx"] if rows else None
            rows = [row for row in rows if row.get("step_idx") == producer_step]
    ids = list(dict.fromkeys(
        row["model_artifact_id"] for row in rows
        if row.get("partition") == "train"
        and row.get("model_artifact_id") in runtime.fitted_predictors
    ))
    if not ids:
        raise ValueError(f"Cannot capture fitted source {name!r} in branch {branch}: no exact fitted producer")
    nodes = tuple(runtime.fitted_predictors[artifact_id] for artifact_id in ids)
    if use_proba:
        from nirs4all.controllers.models.stacking.classification import ClassificationFeatureExtractor, ClassificationInfo, StackingTaskType

        probability_nodes = []
        for node in nodes:
            classes = np.asarray(node.state.classes_)
            info = ClassificationInfo(
                StackingTaskType.BINARY_CLASSIFICATION if len(classes) == 2 else StackingTaskType.MULTICLASS_CLASSIFICATION,
                n_classes=len(classes), class_labels=classes.tolist(), has_probabilities=True,
            )
            probability_nodes.append(FittedNode("probability_features", (node,), ClassificationFeatureExtractor(info, use_proba=True)))
        nodes = tuple(probability_nodes)
    if runtime.phase == ExecutionPhase.CV and test_aggregation is not None:
        source_rows = {row["model_artifact_id"]: row for row in rows if row.get("partition") == "train"
                       and row.get("model_artifact_id") in ids}
        metric = next((row["metric"] for row in source_rows.values() if row.get("metric")), "rmse")
        return FittedNode("source_folds", nodes, {
            "aggregation": test_aggregation.value, "metric": metric,
            "scores": tuple(source_rows[artifact_id].get("val_score") if source_rows[artifact_id].get("val_score") is not None else np.nan
                            for artifact_id in ids),
        })
    return cast(FittedNode, nodes[0]) if len(nodes) == 1 else FittedNode("source_mean", nodes)



def _register_predictor(record: Any, node: FittedNode, runtime: Any) -> None:
    """Persist the complete state separately from the original numerical model."""
    from nirs4all.pipeline.storage.artifacts.operator_chain import OperatorChain, OperatorNode
    from nirs4all.pipeline.storage.artifacts.types import ArtifactType

    registry = runtime.artifact_registry
    chain = OperatorChain.from_path(record.chain_path).append(OperatorNode(
        step_index=record.step_index, operator_class="FittedPredictor", branch_path=record.branch_path,
        substep_index=record.substep_index,
    ))
    path = chain.to_path()
    artifact_id = registry.generate_id(path, record.fold_id, record.pipeline_id)
    registry.register(
        obj=FittedPredictor(node, runtime.fitted_training_steps), artifact_id=artifact_id, artifact_type=ArtifactType.MODEL,
        params={"raw_input_closure_for": record.artifact_id}, format_hint="joblib",
        chain_path=path, custom_name=f"{record.custom_name} raw-input predictor",
    )


def capture_fitted_step(
    step: Any, dataset: Any, before: Any, after: Any, runtime: Any,
    predictions: Any, output: Any, input_node: FittedNode, replay: PreprocessingReplay | None,
    *, is_model: bool, is_meta: bool,
) -> None:
    """Capture the actual fitted controller output, without changing its computation."""
    from nirs4all.pipeline.execution.executor import _StepArtifactValue
    from nirs4all.pipeline.execution.result import StepOutput
    from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord, ArtifactType

    raw_records = output.artifacts if isinstance(output, StepOutput) else output
    records: list[ArtifactRecord] = []
    for value in raw_records or []:
        record = value if isinstance(value, ArtifactRecord) else runtime.artifact_registry.resolve(value["artifact_id"]) if isinstance(value, dict) else None
        if isinstance(value, dict) and record is None:
            raise ValueError(f"Cannot capture declared fitted artifact {value['artifact_id']!r}: registry entry is missing")
        if record is not None:
            records.append(record)
    node = input_node
    if replay is not None:
        values = []
        for record in records:
            value = runtime.artifact_registry.load_artifact(record)
            name = record.custom_name or record.artifact_id
            if isinstance(value, _StepArtifactValue):
                name, value = value.name, value.value
            pair = (name, value)
            values.append(pair)
            replay.source_artifacts.setdefault((runtime.step_number, record.source_index or 0), []).append(pair)
        replay.artifacts[runtime.step_number] = values
        node = FittedNode("transform", (input_node,), replay)
    if isinstance(step, dict) and "branch" in step:
        after.custom["_fitted_pre_branch"] = input_node
    if isinstance(step, dict) and "merge" in step:
        if not isinstance(output, StepOutput) or output.metadata.get("output_as") != "features":
            raise NotImplementedError("Final fitted stacking export requires its declared numeric feature merge")
        info = output.metadata
        branches = {branch["branch_id"]: branch for branch in before.custom.get("branch_contexts", [])}
        blocks = []
        if info.get("include_original"):
            blocks.append(before.custom["_fitted_pre_branch"])
        for branch in info.get("feature_branches_used", []):
            blocks.append(branches[branch]["context"].custom["_fitted_input"])
        for selection in info.get("prediction_selection", []):
            sources = tuple(_source_node(predictions, runtime, name, selection["branch"], runtime.step_number, use_proba=selection.get("use_proba", False))
                            for name in selection["selected_models"])
            blocks.append(FittedNode("aggregate", sources, {
                "names": tuple(selection["selected_models"]), "strategy": selection["aggregation_strategy"],
                "scores": selection.get("model_scores"), "metric": selection.get("metric"), "proba": selection.get("proba", False),
            }))
        if not blocks:
            raise ValueError("Final fitted stacking capture has no declared merge inputs")
        node = FittedNode("concat", tuple(blocks), np.asarray(dataset.x(after.selector, layout="2d")).dtype)
    for record in records:
        if not is_model or record.artifact_type not in (ArtifactType.MODEL, ArtifactType.META_MODEL):
            continue
        model_input = input_node
        if is_meta:
            from nirs4all.controllers.models.meta_model import build_unique_source_names

            candidates = after.custom.get("_fitted_source_models")
            if not candidates:
                raise ValueError("Final MetaModel capture requires its actual ordered source candidates")
            names, mapping = build_unique_source_names(candidates)
            sources = tuple(_source_node(predictions, runtime, *mapping.get(name, (name, None)), runtime.step_number,
                                         use_proba=after.custom["meta_operator"].use_proba and dataset.is_classification,
                                         test_aggregation=after.custom["meta_operator"].stacking_config.test_aggregation)
                            for name in names)
            model_input = FittedNode("concat", sources, after.custom["_reconstruction_result"].X_train_meta.dtype)
        model = runtime.artifact_registry.load_artifact(record)
        processing = after.state.y_processing
        inverse = deepcopy(dataset._targets._transformer) if processing != "numeric" or dataset.is_classification else None
        model_node = FittedNode("model", (model_input,), model, record.artifact_id, inverse, processing)
        runtime.fitted_predictors[record.artifact_id] = model_node
        _register_predictor(record, model_node, runtime)
    after.custom["_fitted_input"] = node
    runtime.fitted_feature_output = node
