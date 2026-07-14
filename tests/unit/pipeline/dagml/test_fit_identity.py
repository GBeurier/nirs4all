"""Unit tests for native DAG-ML fit identity normalization."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.pipeline.dagml.fit_identity import normalize_fit_identity


def test_explicit_sample_ids_groups_and_column_metadata_are_normalized() -> None:
    frame = normalize_fit_identity(
        np.ones((3, 2)),
        np.arange(3),
        sample_ids=["sample.1", "sample.2", "sample.3"],
        groups=["batch-a", "batch-a", "batch-b"],
        metadata={"instrument": ["i1", "i1", "i2"], "temperature": [21.0, 22.5, 23]},
    )

    assert frame.n_samples == 3
    assert frame.explicit_sample_ids is True
    assert frame.sample_ids == ("sample.1", "sample.2", "sample.3")
    assert frame.groups == ("batch-a", "batch-a", "batch-b")
    assert frame.metadata_rows == (
        {"instrument": "i1", "temperature": 21.0},
        {"instrument": "i1", "temperature": 22.5},
        {"instrument": "i2", "temperature": 23},
    )
    assert frame.metadata_by_sample_int() == {
        "instrument": {0: "i1", 1: "i1", 2: "i2"},
        "temperature": {0: 21.0, 1: 22.5, 2: 23},
    }
    assert frame.group_by_sample_int() == {0: "batch-a", 1: "batch-a", 2: "batch-b"}
    assert frame.metadata_by_sample_id()["sample.2"] == {"instrument": "i1", "temperature": 22.5}
    assert len(frame.fingerprint) == 64


def test_row_metadata_shape_is_supported() -> None:
    frame = normalize_fit_identity(
        np.ones((2, 1)),
        np.arange(2),
        sample_ids=["s.1", "s.2"],
        metadata=[{"fold_hint": 0}, {"fold_hint": 1, "site": "lab"}],
    )

    assert frame.metadata_rows == ({"fold_hint": 0}, {"fold_hint": 1, "site": "lab"})


def test_missing_sample_ids_uses_deterministic_compatibility_ids() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.5, 1.5])

    first = normalize_fit_identity(X, y)
    second = normalize_fit_identity(X.copy(), y.copy())

    assert first.explicit_sample_ids is False
    assert first.sample_ids == second.sample_ids
    assert first.sample_ids[0].startswith("n4a.")
    assert first.fingerprint == second.fingerprint


def test_explicit_sample_ids_can_be_required() -> None:
    with pytest.raises(ValueError, match="requires explicit sample_ids"):
        normalize_fit_identity(
            np.ones((2, 2)),
            np.ones(2),
            require_explicit_sample_ids=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_ids": ["same", "same"]}, "unique"),
        ({"sample_ids": ["bad:id", "ok"]}, "invalid dag-ml-data id"),
        ({"sample_ids": ["s1"]}, "sample_ids length"),
        ({"groups": ["g1"]}, "groups length"),
        ({"metadata": {"site": ["a"]}}, "metadata column"),
        ({"metadata": [{"site": object()}, {"site": "b"}]}, "JSON scalar"),
    ],
)
def test_invalid_identity_inputs_fail_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        normalize_fit_identity(np.ones((2, 2)), np.ones(2), **kwargs)


def test_x_y_length_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        normalize_fit_identity(np.ones((3, 2)), np.ones(2))
