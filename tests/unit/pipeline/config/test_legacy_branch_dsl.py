"""Regression locks for legacy separation-branch DSL parsing."""

from nirs4all.controllers.data.branch import BranchController
from nirs4all.operators.filters.base import SampleFilter
from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.operators.transforms import MSC, SNV
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs


def test_by_tag_bool_step_keys_survive_pipeline_config_preprocessing() -> None:
    """Per-value by_tag steps may use Python bool keys from the public examples."""
    config = PipelineConfigs(
        [
            {
                "branch": {
                    "by_tag": "y_z_outlier",
                    "steps": {
                        True: [MSC()],
                        False: [SNV()],
                    },
                }
            }
        ],
        name="legacy_by_tag_bool_keys",
    )

    branch_steps = config.steps[0][0]["branch"]["steps"]

    assert set(branch_steps) == {True, False}
    assert BranchController._resolve_branch_steps(branch_steps, "True") == branch_steps[True]
    assert BranchController._resolve_branch_steps(branch_steps, "False") == branch_steps[False]


def test_by_tag_json_bool_strings_resolve_to_branch_steps() -> None:
    """Replay/config JSON stores bool keys as strings; branch names still resolve."""
    branch_steps = {
        "true": [MSC()],
        "false": [SNV()],
    }

    assert BranchController._resolve_branch_steps(branch_steps, "True") == branch_steps["true"]
    assert BranchController._resolve_branch_steps(branch_steps, "False") == branch_steps["false"]


def test_by_filter_serialized_filter_rehydrates_as_sample_filter() -> None:
    """by_filter receives serialized operator payloads before controller execution."""
    serialized = serialize_component(YOutlierFilter(method="iqr", threshold=2.5))

    restored = deserialize_component(serialized)

    assert isinstance(restored, SampleFilter)
    assert isinstance(restored, YOutlierFilter)
    assert restored.method == "iqr"
    assert restored.threshold == 2.5
