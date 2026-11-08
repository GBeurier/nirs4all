"""Dedicated real-Core gate for canonical ``.n4a`` retrain through the public SDK.

This gate drives the installed ``nirs4all_core`` wheel, the shared Core/CLI fixture
files and a real Methods library. It has no stub and no skip: missing inputs FAIL, so
a green run is the only evidence of runtime qualification. Run it explicitly by path::

    NIRS4ALL_N4A_FIXTURE_DIR=<dir> N4M_LIBRARY_PATH=<libn4m> \\
        pytest tests/integration/canonical_n4a_retrain_gate.py

Inputs:

``NIRS4ALL_N4A_FIXTURE_DIR``
    Directory written by Core's CLI test (``N4A_CLI_WRITE_FIXTURES=<dir>``) holding
    ``train.json``, ``train-v2.json``, ``future.json``, ``future-permuted.json`` and
    ``recipe.json``. ``future-permuted.json`` is ``future.json`` with its rows reordered.
``N4M_LIBRARY_PATH``
    Methods library file (otherwise the installed ``n4m`` wheel is used).
``N4M_EXPECTED_LIBRARY_SHA256``
    Optional independent SHA-256 of that file, checked against the SDK-resolved identity.

The installed ``nirs4all_core`` must expose ``retrain_n4a_native``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all
from nirs4all.api.native_n4a import CANONICAL_N4A_NATIVE_PROFILE, CanonicalN4aRunResult, inspect_n4a
from nirs4all.pipeline.dagml.core_archive_replay import _resolve_methods_library_identity

_FIXTURES = ("train.json", "train-v2.json", "future.json", "future-permuted.json", "recipe.json")
_SOURCE_ID = "spectra"  # the source id recorded by Core's shared fixture
_TOLERANCE = 1e-9  # the shared Core/CLI fixture tolerance


@pytest.fixture
def gate() -> dict[str, Any]:
    """Resolve the real inputs; any missing one fails the gate instead of skipping it."""

    problems: list[str] = []
    fixtures = Path(os.environ.get("NIRS4ALL_N4A_FIXTURE_DIR", ""))
    if not os.environ.get("NIRS4ALL_N4A_FIXTURE_DIR"):
        problems.append("NIRS4ALL_N4A_FIXTURE_DIR is not set")
    else:
        problems.extend(f"fixture file missing: {fixtures / name}" for name in _FIXTURES if not (fixtures / name).is_file())
    library = None
    try:
        core = importlib.import_module("nirs4all_core")
        problems.extend(
            f"installed nirs4all_core lacks {name}"
            for name in ("produce_n4a_native", "inspect_n4a_bytes", "predict_n4a_native", "retrain_n4a_native")
            if not callable(getattr(core, name, None))
        )
        library, digest = _resolve_methods_library_identity(os.environ.get("N4M_LIBRARY_PATH") or None)
        expected = os.environ.get("N4M_EXPECTED_LIBRARY_SHA256")
        if expected and expected.lower() != digest:
            problems.append(f"Methods library SHA-256 {digest} differs from N4M_EXPECTED_LIBRARY_SHA256")
    except Exception as error:  # noqa: BLE001 - every provisioning failure is a gate failure
        problems.append(f"{type(error).__name__}: {error}")
    if problems:
        pytest.fail("canonical .n4a retrain gate inputs are not provisioned:\n  " + "\n  ".join(problems))
    return {"fixtures": fixtures, "library": library}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_sample_ids(path: Path) -> list[str]:
    ids = json.loads(path.read_text(encoding="utf-8"))["dataset"]["sample_ids"]
    assert isinstance(ids, list) and ids and all(isinstance(item, str) for item in ids)
    return ids


def test_public_retrain_of_a_canonical_archive_end_to_end(gate: dict[str, Any], tmp_path: Path) -> None:
    fixtures: Path = gate["fixtures"]
    library: str = gate["library"]

    def predict(archive: Path, record: str) -> Any:
        return nirs4all.predict(
            model=archive, data={"record": fixtures / record, "partition": "predict"}, engine="native", methods_library_path=library
        )

    # Source: one public produce, exported as the immutable sealed bytes.
    produced = nirs4all.run(
        fixtures / "recipe.json",
        fixtures / "train.json",
        source_id=_SOURCE_ID,
        engine="native",
        native_profile=CANONICAL_N4A_NATIVE_PROFILE,
        methods_library_path=library,
        verbose=0,
    )
    assert isinstance(produced, CanonicalN4aRunResult)
    source = produced.export(tmp_path / "source.n4a")
    source_sha = _sha256(source)
    assert produced.native_archive_reference["archive_sha256"] == source_sha
    original_prediction = predict(source, "future.json")

    # Public retrain from the archive path on the new record: a fresh result and fresh sealed bytes.
    retrained = nirs4all.retrain(source, fixtures / "train-v2.json", engine="native", methods_library_path=library, verbose=0)
    assert isinstance(retrained, CanonicalN4aRunResult)
    fresh = retrained.export(tmp_path / "retrained.n4a")
    fresh_sha = _sha256(fresh)
    assert fresh_sha != source_sha
    assert retrained.native_archive_reference["archive_sha256"] == fresh_sha
    assert retrained.native_archive_reference["container_id"] != produced.native_archive_reference["container_id"]
    assert _sha256(source) == source_sha  # the original archive is unchanged
    assert produced.native_archive_reference["archive_sha256"] == source_sha  # and so is the original result

    # Owner OOF scores: the public score is Core's single SELECT decision, read back from the sealed bytes.
    view = inspect_n4a(fresh)
    assert view["receipt"]["container_sha256"] == fresh_sha
    outcome = view["training"]["outcome"]
    (decision,) = outcome["execution_bundle"]["selections"].values()
    assert decision["evaluation_scope"] == "oof"
    assert decision["selected_candidate_id"] == outcome["selected_variant_id"] == retrained.selection["selected_variant_id"]
    assert retrained.selection["selected_score"] == decision["selected_score"]
    assert retrained.best_score == decision["selected_score"]
    assert retrained.selection["metric_name"] == decision["metric_name"] == outcome["score_set"]["selection_metric"]

    # Retrain from the produced result is the same owner action on the same stored recipe.
    from_result = nirs4all.retrain(produced, {"record": fixtures / "train-v2.json"}, engine="native", methods_library_path=library, verbose=0)
    assert isinstance(from_result, CanonicalN4aRunResult)
    assert from_result.native_archive_reference["archive_sha256"] != source_sha
    assert _sha256(source) == source_sha

    # Predict on the retrained archive: rows come back in record order, whatever that order is.
    future_ids = _record_sample_ids(fixtures / "future.json")
    permuted_ids = _record_sample_ids(fixtures / "future-permuted.json")
    assert sorted(future_ids) == sorted(permuted_ids) and future_ids != permuted_ids
    natural = predict(fresh, "future.json")
    permuted = predict(fresh, "future-permuted.json")
    assert natural.sample_ids == future_ids and permuted.sample_ids == permuted_ids
    assert natural.metadata["training_performed"] is False and natural.metadata["archive_sha256"] == fresh_sha
    assert natural.y_pred.shape == (len(future_ids), 2) and permuted.y_pred.shape == (len(permuted_ids), 2)
    assert np.isfinite(natural.y_pred).all() and np.isfinite(permuted.y_pred).all()

    by_id = dict(zip(future_ids, natural.y_pred, strict=True))
    aligned = np.stack([by_id[sample_id] for sample_id in permuted_ids])
    assert np.max(np.abs(permuted.y_pred - aligned)) < _TOLERANCE  # each sample keeps its prediction wherever it sits
    assert np.max(np.abs(permuted.y_pred - natural.y_pred)) > 1e-6  # and the permutation is visible positionally

    # The retrain is real: the same future record is predicted differently by the original and retrained archives.
    assert np.max(np.abs(natural.y_pred - original_prediction.y_pred)) > 1e-6
    assert _sha256(source) == source_sha and _sha256(fresh) == fresh_sha  # predictions never rewrite an archive
