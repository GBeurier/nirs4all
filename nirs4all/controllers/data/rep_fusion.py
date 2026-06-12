"""Controller for the ``rep_fusion`` relational materialisation step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.data.raw_multisource import RawMultiSourceDataset
from nirs4all.operators.data.rep_fusion import RepFusionConfig
from nirs4all.pipeline.execution.result import StepOutput

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class RepFusionController(OperatorController):
    """Materialise raw multi-source staging via a replayable RepresentationPlan."""

    priority = 3

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Return True for the ``rep_fusion`` pipeline keyword."""
        return keyword == "rep_fusion"

    @classmethod
    def use_multi_source(cls) -> bool:
        """The step consumes the whole relation-aware staging object."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """The same representation plan is replayable in prediction mode."""
        return True

    def execute(
        self,
        step_info: ParsedStep,
        dataset: Any,
        context: ExecutionContext,
        runtime_context: RuntimeContext,
        source: int = -1,
        mode: str = "train",
        loaded_binaries: list[tuple[str, Any]] | None = None,
        prediction_store: Any | None = None,
    ) -> tuple[ExecutionContext, StepOutput]:
        """Execute ``rep_fusion`` on a :class:`RawMultiSourceDataset`."""
        raw_config = step_info.original_step.get("rep_fusion") if isinstance(step_info.original_step, dict) else None
        config = RepFusionConfig.from_step_value(raw_config if raw_config is not None else "per_source_aggregate")
        if not isinstance(dataset, RawMultiSourceDataset):
            raise ValueError(
                "rep_fusion requires a RawMultiSourceDataset staging object. "
                "Legacy SpectroDataset -> relational staging conversion is intentionally not implicit."
            )
        if context.custom.get("branch_contexts"):
            raise ValueError(
                "rep_fusion must run before branch execution. Materialising relation-aware staging inside branch contexts "
                "is not supported because dataset overrides are pipeline-wide."
            )
        materialized = config.materialize(dataset)
        materialization_manifest = materialized.to_manifest()
        materialized_dataset = materialized.to_spectro_dataset(
            name=f"{getattr(dataset, 'name', 'raw_multisource')}::{materialized.representation}"
        )
        result_context = context.copy()
        result_context.custom["dataset_override"] = materialized_dataset
        result_context.custom["relation_materialization_manifest"] = {
            "representation": materialization_manifest["representation"],
            "representation_plan": materialization_manifest["representation_plan"],
            "fingerprint": materialization_manifest["fingerprint"],
            "shape": materialization_manifest["shape"],
            "has_feature_mask": materialization_manifest["has_feature_mask"],
        }
        output = StepOutput()
        output.metadata.update(
            {
                "transformation": "rep_fusion",
                "representation": materialized.representation,
                "shape": list(materialized.X.shape),
                "representation_plan": config.plan.to_dict(),
                "materialization_manifest": materialization_manifest,
                "dataset_override": True,
                "mode": mode,
            }
        )
        return result_context, output


__all__ = ["RepFusionController"]
