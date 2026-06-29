"""Unit tests for ``_index_sample_blocks`` aggregated-block surfacing + id-matched y_true pairing.

These pin the pure indexing logic (no dag-ml campaign): the SAMPLE-level ``aggregated_predictions``
surfacing (Gap 2), its collision rule (a real ``predictions`` block WINS the shared key), and — the
load-bearing one — that an aggregated block is paired to its ``regression_targets`` y_true BY SAMPLE ID,
never by row count, so two EQUAL-LENGTH but differently-id'd sample-level target blocks never cross-pair.
"""

from __future__ import annotations

from nirs4all.pipeline.dagml.result import _index_sample_blocks


def _unit_ids(ids: list[str]) -> list[dict[str, str]]:
    return [{"level": "sample", "id": sample_id} for sample_id in ids]


def test_aggregated_block_pairs_y_true_by_id_not_by_row_count() -> None:
    """An aggregated block pairs with the SAME-ID target, not the first same-length one.

    Two sample-level ``regression_targets`` of EQUAL length (2) but different ids (s1-group vs s2-group)
    sit on the node; the aggregated block is for the s2 group. The pairing must select the s2 target (its
    real y_true), NOT the first same-length s1 target — the count-match bug Codex reproduced.
    """
    s1_ids, s2_ids = ["s10", "s11"], ["s20", "s21"]
    s1_y_true, s2_y_true = [[1.0], [2.0]], [[30.0], [40.0]]
    result = {
        "node_id": "model:0",
        "predictions": [],
        "regression_targets": [
            {"level": "sample", "unit_ids": _unit_ids(s1_ids), "values": s1_y_true, "target_names": ["y"]},
            {"level": "sample", "unit_ids": _unit_ids(s2_ids), "values": s2_y_true, "target_names": ["y"]},
        ],
        "aggregated_predictions": [
            {"producer_node": "model:0", "partition": "test", "fold_id": None, "level": "sample", "unit_ids": _unit_ids(s2_ids), "values": [[33.0], [44.0]], "target_names": ["y"]},
        ],
    }

    index = _index_sample_blocks([result])
    block, target = index[("model:0", "test", None)]

    assert block["sample_ids"] == s2_ids  # normalized unit_ids -> flat sample_ids
    assert target is not None
    assert target["values"] == s2_y_true  # the s2 y_true, NOT the same-length s1 [[1.0],[2.0]]


def test_aggregated_block_y_true_reordered_to_block_id_order() -> None:
    """When the target's id order differs from the block's, y_true is realigned to the block order."""
    block_ids = ["s20", "s21", "s22"]
    target_ids = ["s22", "s20", "s21"]  # different order
    target_values = [[2.2], [2.0], [2.1]]  # aligned to target_ids
    result = {
        "node_id": "model:0",
        "predictions": [],
        "regression_targets": [
            {"level": "sample", "unit_ids": _unit_ids(target_ids), "values": target_values, "target_names": ["y"]},
        ],
        "aggregated_predictions": [
            {"producer_node": "model:0", "partition": "test", "fold_id": None, "level": "sample", "unit_ids": _unit_ids(block_ids), "values": [[9.0], [9.1], [9.2]], "target_names": ["y"]},
        ],
    }

    _block, target = _index_sample_blocks([result])[("model:0", "test", None)]
    assert target is not None
    # Realigned to block order s20,s21,s22 -> 2.0, 2.1, 2.2.
    assert target["values"] == [[2.0], [2.1], [2.2]]


def test_aggregated_block_no_id_match_leaves_y_true_none() -> None:
    """No id-matching sample target (only a same-LENGTH but different-id one) -> y_true is None.

    There is NO count-match fallback: attaching the wrong-id target would be silently wrong, so the row
    is left score-only instead.
    """
    result = {
        "node_id": "model:0",
        "predictions": [],
        "regression_targets": [
            {"level": "sample", "unit_ids": _unit_ids(["sX", "sY"]), "values": [[1.0], [2.0]], "target_names": ["y"]},
        ],
        "aggregated_predictions": [
            {"producer_node": "model:0", "partition": "test", "fold_id": None, "level": "sample", "unit_ids": _unit_ids(["s20", "s21"]), "values": [[9.0], [9.1]], "target_names": ["y"]},
        ],
    }

    block, target = _index_sample_blocks([result])[("model:0", "test", None)]
    assert block["sample_ids"] == ["s20", "s21"]
    assert target is None


def test_predictions_block_wins_shared_key_over_aggregated() -> None:
    """A direct ``predictions`` block claims the key first; the aggregated block must NOT clobber it."""
    direct_pred = {"producer_node": "model:0", "partition": "test", "fold_id": None, "sample_ids": ["s1", "s2"], "values": [[5.0], [6.0]], "target_names": ["y"]}
    result = {
        "node_id": "model:0",
        "predictions": [direct_pred],
        "regression_targets": [
            {"level": "sample", "unit_ids": _unit_ids(["s1", "s2"]), "values": [[5.5], [6.5]], "target_names": ["y"]},
        ],
        "aggregated_predictions": [
            {"producer_node": "model:0", "partition": "test", "fold_id": None, "level": "sample", "unit_ids": _unit_ids(["s1", "s2"]), "values": [[99.0], [99.0]], "target_names": ["y"]},
        ],
    }

    block, _target = _index_sample_blocks([result])[("model:0", "test", None)]
    assert block is direct_pred  # the direct predictions block wins, not the aggregated one


def test_non_sample_level_aggregated_block_is_skipped() -> None:
    """A target/group-level aggregated block is NOT surfaced (only sample-level rows are direct-block rows)."""
    result = {
        "node_id": "model:0",
        "predictions": [],
        "regression_targets": [],
        "aggregated_predictions": [
            {"producer_node": "model:0", "partition": "test", "fold_id": None, "level": "target", "unit_ids": [{"level": "target", "id": "t0"}], "values": [[1.0]], "target_names": ["y"]},
        ],
    }
    assert _index_sample_blocks([result]) == {}
