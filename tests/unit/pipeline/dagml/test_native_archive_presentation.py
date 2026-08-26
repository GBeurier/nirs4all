"""Transport-only validation for native conformal archive presentation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nirs4all.pipeline.dagml.native_archive_replay import (
    NativeArchiveReplayError,
    _validate_conformal_presentation_transport,
)


def _package() -> dict[str, object]:
    return {
        "package_fingerprint": "a" * 64,
        "conformal_calibration": {"calibration_fingerprint": "b" * 64},
    }


def _outcome() -> SimpleNamespace:
    return SimpleNamespace(
        to_dict=lambda: {
            "outcome_fingerprint": "c" * 64,
            "outputs": [
                {
                    "binding": {"binding_id": "binding:pls"},
                    "predictions": [
                        {
                            "sample_ids": ["p1", "p2"],
                            "values": [[2.0], [3.0]],
                        }
                    ],
                }
            ],
        }
    )


def _presentation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_fingerprint": "a" * 64,
        "replay_outcome_fingerprint": "c" * 64,
        "binding_id": "binding:pls",
        "target_name": "moisture",
        "sample_ids": ["p1", "p2"],
        "point_predictions": [2.0, 3.0],
        "intervals": [
            {"coverage": 0.9, "lower": [1.0, 2.0], "upper": [3.0, 4.0], "qhat": 1.0}
        ],
        "calibration_fingerprint": "b" * 64,
        "presentation_fingerprint": "d" * 64,
    }


def test_native_conformal_presentation_closes_exact_replay_points() -> None:
    _validate_conformal_presentation_transport(
        _presentation(),
        package_document=_package(),
        outcome=_outcome(),
        sample_ids=("p1", "p2"),
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(sample_ids=["p2", "p1"]), "identities"),
        (lambda p: p.update(point_predictions=[2.0, 3.1]), "points"),
        (lambda p: p.update(package_fingerprint="e" * 64), "provenance"),
    ],
)
def test_native_conformal_presentation_refuses_resigned_transport_mismatch(mutate, message: str) -> None:
    presentation = _presentation()
    mutate(presentation)
    with pytest.raises(NativeArchiveReplayError, match=message):
        _validate_conformal_presentation_transport(
            presentation,
            package_document=_package(),
            outcome=_outcome(),
            sample_ids=("p1", "p2"),
        )
