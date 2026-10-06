"""Fold-local fitting and replay of learned legacy feature preprocessing."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.types import Layout
from nirs4all.pipeline.config.context import ExecutionContext, MapArtifactProvider, RuntimeContext
from nirs4all.pipeline.steps.step_runner import StepRunner


@dataclass
class PreprocessingReplay:
    """Replay fitted controllers using feature schemas, without training rows."""

    shapes: list[tuple[int, int]]
    processings: list[list[str]]
    headers: list[list[str]]
    units: list[str]
    context: ExecutionContext
    input_step: int = 0
    steps: list[tuple[int, Any]] = field(default_factory=list)
    artifacts: dict[int, list[tuple[str, Any]]] = field(default_factory=dict)
    source_artifacts: dict[tuple[int, int], list[tuple[str, Any]]] = field(default_factory=dict)

    @classmethod
    def capture(cls, dataset: SpectroDataset, context: ExecutionContext) -> "PreprocessingReplay":
        arrays = dataset.x(context.selector, "3d", concat_source=False, include_excluded=True)
        if not isinstance(arrays, list):
            arrays = [arrays]
        ctx = context.copy()
        ctx.custom = {}
        ctx.selector.partition = None
        ctx.selector._extra.pop("sample", None)
        return cls([a.shape[1:] for a in arrays],
                   [list(dataset.features_processings(i)) for i in range(len(arrays))],
                   [dataset.headers(i) or [] for i in range(len(arrays))],
                   [dataset.header_unit(i) for i in range(len(arrays))], ctx)

    @property
    def start_step(self) -> int:
        return self.input_step or self.steps[0][0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return X
        values = np.asarray(X)
        sources = []
        offset = 0
        for n_processings, n_features in self.shapes:
            width = n_processings * n_features
            if values.ndim == 3 and self.context.selector.layout == "3d_transpose":
                source = values[:, :, offset:offset + n_processings].transpose(0, 2, 1)
                offset += n_processings
            elif values.ndim == 3:
                source = values[:, :, offset:offset + n_features]
                offset += n_features
            else:
                flat_source = values.reshape(len(X), -1)[:, offset:offset + width]
                source = (flat_source.reshape(len(X), n_features, n_processings).transpose(0, 2, 1)
                          if self.context.selector.layout == "2d_interleaved"
                          else flat_source.reshape(len(X), n_processings, n_features))
                offset += width
            if source.shape[1:] != (n_processings, n_features):
                raise ValueError("Fold preprocessing input does not match its saved source/processing schema")
            sources.append(source)
        dataset = SpectroDataset("preprocessing_replay")
        dataset.add_samples(sources[0][:, 0, :] if len(sources) == 1 else [a[:, 0, :] for a in sources], {"partition": "test"},
                            headers=(self.headers[0] or None) if len(sources) == 1 else (self.headers if any(self.headers) else None),
                            header_unit=self.units[0] if len(sources) == 1 else self.units)
        dataset.add_targets(np.zeros(len(X)))
        for index, (array, names) in enumerate(zip(sources, self.processings, strict=True)):
            if array.shape[1] > 1:
                dataset.update_features(source_processings=[""] * (array.shape[1] - 1),
                                        features=[array[:, i, :] for i in range(1, array.shape[1])],
                                        processings=names[1:], source=index)
            dataset._features.sources[index]._processing_mgr.reset_processings(names)
        context = self.context.copy()
        context.state.mode = "predict"
        runner = StepRunner(mode="predict", verbose=0, show_spinner=False)
        runtime = RuntimeContext(step_runner=runner, save_artifacts=False, save_charts=False,
                                 artifact_provider=MapArtifactProvider(self.artifacts, source_artifact_map=self.source_artifacts))
        for index, step in self.steps:
            runtime.step_number = index
            runtime.reset_processing_counter()
            context = context.with_step_number(index)
            context = runner.execute(step, dataset, context, runtime).updated_context
        return np.asarray(dataset.x(context.selector, cast(Layout, context.selector.layout or "2d")))


@dataclass
class FoldPreprocessing:
    """Keep raw features and stage specifications until a fold is fitted."""

    dataset: SpectroDataset
    replay: PreprocessingReplay
    active: bool = False
    unsupported_reason: str | None = None

    @classmethod
    def capture(cls, dataset: SpectroDataset, context: ExecutionContext) -> "FoldPreprocessing":
        return cls(deepcopy(dataset), PreprocessingReplay.capture(dataset, context))

    def raw_features(self, ids: Any, layout: str = "2d") -> np.ndarray:
        if self.unsupported_reason:
            raise ValueError(self.unsupported_reason)
        # Exact child IDs must address their own rows, rather than select all
        # siblings through the base-keyed sample selector.
        return np.asarray(self.dataset._feature_accessor.x_rows(list(ids), cast(Layout, layout)))

    def prepare_fold(self, dataset: SpectroDataset, model_context: ExecutionContext, train_ids: Any) -> PreprocessingReplay:
        from nirs4all.pipeline.execution.executor import PipelineExecutor, _StepArtifactValue
        from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry

        local = deepcopy(self.dataset)
        # Targets may have acquired a y-processing after the feature snapshot.
        local._targets = deepcopy(dataset._targets)
        local._target_accessor = deepcopy(dataset._target_accessor)
        local._target_accessor._block = local._targets
        local._target_accessor._indexer = local._indexer
        keep = {int(i) for i in train_ids}
        local._indexer.mark_excluded([i for i in range(local.num_samples) if i not in keep],
                                    reason="cv_validation", cascade_to_augmented=False)
        replay = deepcopy(self.replay)
        replay.context.selector.layout = model_context.selector.layout
        context = replay.context.copy()
        context.state.mode = "train"
        context.state.y_processing = model_context.state.y_processing
        runner = StepRunner(mode="train", verbose=0, show_spinner=False)
        with TemporaryDirectory(prefix="nirs4all-cv-preprocessing-") as directory:
            registry = ArtifactRegistry(Path(directory), local.name, pipeline_id="cv_preprocessing")
            runtime: Any = RuntimeContext(step_runner=runner, artifact_registry=registry,
                                     pipeline_id="cv_preprocessing", pipeline_uid="cv_preprocessing", save_charts=False)
            runtime._cv_fitting = True
            executor = PipelineExecutor(runner, save_charts=False)
            for index, step in replay.steps:
                runtime.step_number = index
                runtime.reset_processing_counter()
                context = context.with_step_number(index)
                if isinstance(step, dict) and "fit_on_all" in step:
                    step = dict(step, fit_on_all=False)
                result = runner.execute(step, local, context, runtime)
                context = result.updated_context
                executor._process_step_artifacts(result.artifacts, runtime_context=runtime, context=context)
                records = registry.get_artifacts_for_step("cv_preprocessing", index)
                values = []
                for record in records:
                    obj = registry.load_artifact(record)
                    value = (obj.name, obj.value) if isinstance(obj, _StepArtifactValue) else (record.custom_name or record.artifact_id, obj)
                    values.append(value)
                    key = (index, record.source_index or 0)
                    replay.source_artifacts.setdefault(key, []).append(value)
                replay.artifacts[index] = values
        return replay


@dataclass
class FoldPreprocessedModel:
    """Persist a model together with its fold's fitted preprocessing."""

    model: Any
    preprocessing: PreprocessingReplay

    def predict(self, X: np.ndarray, **kwargs: Any) -> Any:
        return self.model.predict(self.preprocessing.transform(X), **kwargs)

    def predict_proba(self, X: np.ndarray, **kwargs: Any) -> Any:
        return self.model.predict_proba(self.preprocessing.transform(X), **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)


def feature_step(step: Any) -> tuple[bool, bool]:
    """Identify feature stages and target-dependent stages, including containers."""
    from nirs4all.controllers.transforms.transformer import TransformerMixinController
    from nirs4all.pipeline.steps.parser import StepParser

    if isinstance(step, list):
        flags = [feature_step(item) for item in step]
        return any(f for f, _ in flags), any(s for _, s in flags)
    if isinstance(step, dict):
        branch = step.get("branch")
        if isinstance(branch, dict) and branch.get("by_source"):
            stages = branch.get("steps", [])
            if isinstance(stages, dict):
                stages = list(stages.values())
            return False, feature_step(stages)[1]
        if "merge_sources" in step or (isinstance(step.get("merge"), dict) and "sources" in step["merge"]):
            return True, False
        for key in ("concat_transform", "feature_augmentation"):
            if key in step:
                spec = step[key]
                if isinstance(spec, dict):
                    spec = spec.get("operations", [])
                return True, feature_step(spec)[1]
        if "auto_transfer_preproc" in step:
            return True, False
    parsed = StepParser().parse(step)
    from nirs4all.pipeline.steps.router import ControllerRouter

    op = parsed.operator
    controller = ControllerRouter().route(parsed, step)
    is_model = controller.__class__.__module__.startswith("nirs4all.controllers.models")
    is_feature = not is_model and parsed.keyword not in ("model", "meta_model", "y_processing") and hasattr(op, "transform") and hasattr(op, "fit")
    return is_feature, bool(is_feature and (TransformerMixinController._uses_y(op) or controller.__class__.__name__ == "FeatureSelectionController"))


def observe_feature_step(step: Any, dataset: SpectroDataset, context: ExecutionContext, runtime: Any) -> bool:
    """Capture stages even inside branches; return whether replay owns this stage."""
    if runtime is None or getattr(runtime, "_cv_fitting", False):
        return False
    is_feature, supervised = feature_step(step)
    plan = context.custom.get("cv_preprocessing")
    if context.state.mode not in ("predict", "explain") and isinstance(step, dict):
        branch = step.get("branch")
        if isinstance(branch, dict) and branch.get("by_source") and (supervised or (plan is not None and plan.active)):
            raise ValueError(
                "Fold-local supervised preprocessing with by_source branch routing is unsupported; "
                "use merge_sources before supervised selection or separate per-source pipelines."
            )
        if "sample_augmentation" in step and plan is not None:
            plan.unsupported_reason = (
                "Fold-local preprocessing cannot replay sample augmentation introduced after learned preprocessing; "
                "place sample_augmentation before the learned feature stages."
            )
            if plan.active:
                raise ValueError(plan.unsupported_reason)
        if "merge" in step and not is_feature:
            spec = step["merge"]
            merges_features = not isinstance(spec, dict) or bool(spec.get("features")) or "predictions" not in spec
            branch_plans = [branch["context"].custom.get("cv_preprocessing")
                            for branch in context.custom.get("branch_contexts", []) if "context" in branch]
            if merges_features and any(p is not None and p.active for p in [plan, *branch_plans]):
                raise ValueError(
                    "Fold-local supervised preprocessing followed by a branch feature merge is unsupported; "
                    "merge the raw branch features before supervised feature selection, or train models inside each branch."
                )
    if not is_feature:
        return False
    if context.state.mode in ("predict", "explain"):
        bounds = context.custom.get("cv_replay_bounds")
        return bool(bounds and bounds[0] <= runtime.step_number < bounds[1])
    if getattr(runtime, "_cv_container", False):
        return False
    if plan is None:
        plan = FoldPreprocessing.capture(dataset, context)
        plan.replay.input_step = runtime.step_number
        context.custom["cv_preprocessing"] = plan
    plan.replay.steps.append((len(plan.replay.steps) + 1, step))
    plan.active = plan.active or supervised
    if plan.active and plan.unsupported_reason:
        raise ValueError(plan.unsupported_reason)
    return False
