"""Long structural identities retain full lineage through bounded record keys."""

from copy import deepcopy

import pytest

from nirs4all.pipeline.dagml.node_runner import _build_result


def _task():
    return {
        "run_id": "run:structural-torch",
        "node_plan": {
            "node_id": "gen:generator_torch-topolog_96977ab9:c3:n2.a3_source_metadata_fd35d496",
            "kind": "model", "controller_id": "controller:nirs4all.model",
            "controller_version": "1.4.1", "params_fingerprint": "a" * 64,
        },
        "phase": "FIT_CV", "variant_id": "host_hpo:trial:0000000003",
        "fold_id": "fold0.inner.fold0", "seed": 17,
    }


def test_structural_lineage_key_is_bounded_deterministic_and_preserves_identity():
    task = _task()
    original = deepcopy(task)
    lineage = _build_result(task, [], [], {})["lineage"]
    assert len(lineage["record_id"].encode("utf-8")) <= 128
    assert lineage["record_id"].startswith("lineage:")
    assert lineage == _build_result(task, [], [], {})["lineage"]
    for field in ("run_id", "phase", "variant_id", "fold_id", "seed"):
        assert lineage[field] == task[field]
    assert lineage["node_id"] == task["node_plan"]["node_id"]
    assert lineage["params_fingerprint"] == task["node_plan"]["params_fingerprint"]
    assert task == original


@pytest.mark.parametrize("field", ["run_id", "node_id", "phase", "variant_id", "fold_id"])
def test_long_lineage_keys_do_not_alias_distinct_task_identities(field):
    task = _task()
    first = _build_result(task, [], [], {})["lineage"]["record_id"]
    target = task["node_plan"] if field == "node_id" else task
    target[field] += ":other"
    second = _build_result(task, [], [], {})["lineage"]["record_id"]
    assert first != second
    assert len(second.encode("utf-8")) <= 128


@pytest.mark.parametrize("length", [127, 128, 129])
def test_lineage_byte_limit_preserves_existing_short_keys(length):
    task = _task()
    task.update(phase="FIT_CV", variant_id="v", fold_id="f")
    suffix = ":FIT_CV:v:f"
    task["node_plan"]["node_id"] = "n" * (length - len("lineage:") - len(suffix))
    expected = "lineage:" + task["node_plan"]["node_id"] + suffix
    record_id = _build_result(task, [], [], {})["lineage"]["record_id"]
    assert len(record_id.encode("utf-8")) <= 128
    if length <= 128:
        assert record_id == expected
    else:
        assert record_id != expected
