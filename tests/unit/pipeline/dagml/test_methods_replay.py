"""Contract tests for the narrow portable Methods replay host adapter."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.pipeline.dagml.methods_replay import (
    MethodsN4mmReplayCallbacks,
    MethodsPortableReplayError,
)
from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchiveReplayError,
    _target_names_by_node,
)


class _Resolver:
    def resolve_features(self, sample_ids, *, include_augmented):
        assert sample_ids == ["sample:one", "sample:two"]
        assert include_augmented is False
        return {"values": np.asarray([[1.0, 2.0], [3.0, 4.0]])}


class _Context:
    closed = 0

    def close(self):
        type(self).closed += 1


class _Model:
    imported: list[bytes] = []
    closed = 0

    @classmethod
    def from_bytes(cls, context, payload):
        assert isinstance(context, _Context)
        cls.imported.append(payload)
        return cls()

    def predict(self, context, values):
        assert isinstance(context, _Context)
        return np.asarray([[10.0], [20.0]])

    def close(self):
        type(self).closed += 1


def _task(handle: int | None = None) -> dict:
    input_handles = {} if handle is None else {
        "artifact:artifact:model:methods": {
            "handle": handle,
            "kind": "model",
            "owner_controller": "controller:methods.pls",
        }
    }
    return {
        "run_id": "run:portable",
        "phase": "PREDICT",
        "variant_id": None,
        "fold_id": None,
        "branch_path": [],
        "seed": None,
        "node_plan": {
            "node_id": "model:methods",
            "kind": "model",
            "controller_id": "controller:methods.pls",
            "controller_version": "1.0.0",
            "params_fingerprint": "a" * 64,
        },
        "data_views": {
            "x": {
                "partition": "predict",
                "sample_ids": ["sample:one", "sample:two"],
            }
        },
        "input_handles": input_handles,
    }


def test_native_methods_callbacks_import_predict_and_release_exact_n4mm_bytes():
    _Context.closed = _Model.closed = 0
    _Model.imported = []
    callbacks = MethodsN4mmReplayCallbacks(
        _Resolver(),
        target_names_by_node={"model:methods": ["protein"]},
        context_type=_Context,
        model_type=_Model,
    )
    handle = callbacks.artifact_callback(
        {
            "operation": "hydrate",
            "request": {
                "controller_id": "controller:methods.pls",
                "artifact": {"kind": "n4m_model"},
            },
            "payload": [1, 2, 3, 255],
        }
    )
    assert handle == {
        "handle": 1,
        "kind": "model",
        "owner_controller": "controller:methods.pls",
    }
    result = callbacks.op_callback(_task(handle["handle"]))
    assert result["predictions"] == [
        {
            "prediction_id": "pred:model:methods:PREDICT:portable",
            "producer_node": "model:methods",
            "partition": "final",
            "fold_id": None,
            "sample_ids": ["sample:one", "sample:two"],
            "values": [[10.0], [20.0]],
            "target_names": ["protein"],
        }
    ]
    assert _Model.imported == [b"\x01\x02\x03\xff"]
    assert callbacks.artifact_callback({"operation": "release", "handle": handle}) is None
    assert callbacks.active_handle_count == 0
    assert (_Model.closed, _Context.closed) == (1, 1)


def test_native_methods_callback_refuses_missing_or_wrong_n4mm_handle():
    callbacks = MethodsN4mmReplayCallbacks(
        _Resolver(),
        target_names_by_node={"model:methods": ["protein"]},
        context_type=_Context,
        model_type=_Model,
    )
    with pytest.raises(MethodsPortableReplayError, match="exactly one hydrated"):
        callbacks.op_callback(_task())
    with pytest.raises(MethodsPortableReplayError, match="only hydrates n4m_model"):
        callbacks.artifact_callback(
            {
                "operation": "hydrate",
                "request": {
                    "controller_id": "controller:methods.pls",
                    "artifact": {"kind": "joblib"},
                },
                "payload": [1],
            }
        )


def test_archive_package_target_schema_requires_one_exact_schema_per_node():
    assert _target_names_by_node(
        {
            "output_bindings": [
                {"node_id": "model:methods", "target_names": ["protein"]},
                {"node_id": "model:methods", "target_names": ["protein"]},
            ]
        }
    ) == {"model:methods": ["protein"]}
    with pytest.raises(NativeArchiveReplayError, match="incompatible target schemas"):
        _target_names_by_node(
            {
                "output_bindings": [
                    {"node_id": "model:methods", "target_names": ["protein"]},
                    {"node_id": "model:methods", "target_names": ["moisture"]},
                ]
            }
        )
