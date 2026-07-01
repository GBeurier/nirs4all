"""Native results WRITER + READER for the dag-ml backend (P3 Slice 2b-i, B-C HYBRID).

ADDITIVE, OFF-by-default on-disk results for ``run(engine="dag-ml")``: a per-run directory with
``manifest.json`` + the VERBATIM ``score_set.json`` + a ``predictions.parquet`` projection. These pin:

* (a) ROUND-TRIP — enabling native results writes the 3 files; ``score_set.json`` equals the raw native
  ScoreSet (the manifest hash matches); the reader round-trips y_pred/y_true/sample_indices == the
  RunResult; the manifest header + capability flags are correct.
* (b) OFF by default — a plain dag-ml run writes NOTHING, creates no ``./nirs4all_results``, and the
  RunResult is unchanged.
* (c) the LEGACY workspace is untouched — ``engine="legacy"`` ignores ``results_path`` (no native dir).
* (d) a SWEEP writes per-variant predictions.
* (e) MULTI-TARGET round-trips WITH shape (the parquet carries ``*_shape`` so a 2D y recovers its width).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.transforms.scalers import StandardNormalVariate as SNV
from nirs4all.pipeline.dagml.native_results import (
    MANIFEST_SCHEMA_VERSION,
    _score_set_hash,
    native_results_enabled,
    read_native_results,
)
from nirs4all.pipeline.dagml.run_backend import run_via_dagml

pytestmark = [pytest.mark.parity]

from ._datasets import dataset_path  # noqa: E402

_DAGML_CLI = Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"
_N_SPLITS = 3


# ---------------------------------------------------------------------------
# Gate (env / explicit path) — no campaign needed.
# ---------------------------------------------------------------------------


def test_native_results_gate_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF when NEITHER an explicit results_path NOR $N4A_NATIVE_RESULTS is set."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    assert native_results_enabled(None) is False


def test_native_results_gate_explicit_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit results_path enables the writer regardless of the env."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    assert native_results_enabled(str(tmp_path)) is True


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("/some/dir", True), ("0", False), ("false", False), ("", False)],
)
def test_native_results_gate_env(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    """$N4A_NATIVE_RESULTS enables env-only (truthy values on; 0/false/empty off)."""
    monkeypatch.setenv("N4A_NATIVE_RESULTS", value)
    assert native_results_enabled(None) is expected


# ---------------------------------------------------------------------------
# (a) ROUND-TRIP — single pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_round_trip_single(tmp_path: Path) -> None:
    """Enabling native results writes the 3 files; score_set.json is the VERBATIM ScoreSet (hash matches);
    the reader round-trips the direct-block arrays == the RunResult; the manifest header + flags are right."""
    results_root = tmp_path / "results"
    pipeline = [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )

    run_dirs = sorted(results_root.iterdir())
    assert len(run_dirs) == 1, "exactly one run directory written"
    run_dir = run_dirs[0]
    # The 3 core files + the artifacts/ model tree (P3 Slice 2c-i: the in-process path captures the fitted
    # REFIT estimator). The subprocess path would write no artifacts/ — but the default here is in-process.
    assert {p.name for p in run_dir.iterdir()} == {"manifest.json", "score_set.json", "predictions.parquet", "artifacts"}

    # score_set.json == the RAW native ScoreSet captured on the result (VERBATIM, no re-author).
    raw_score_set = result._dagml_score_set  # noqa: SLF001
    on_disk_score_set = json.loads((run_dir / "score_set.json").read_text())
    assert on_disk_score_set == raw_score_set
    assert "reports" in on_disk_score_set and on_disk_score_set["reports"], "the ScoreSet carries native reports"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["engine"] == "dag-ml"
    assert manifest["nirs4all_version"] == nirs4all.__version__
    assert manifest["dag_ml_version"] not in ("", None)
    assert manifest["metric"] == "rmse"
    assert manifest["task_type"] == "regression"
    assert manifest["num_predictions"] == result.num_predictions
    # P3 Slice 2c-i: the in-process run captures the fitted REFIT model, so has_model_artifacts is now true.
    assert manifest["capabilities"] == {"has_model_artifacts": True, "has_aggregate_predictions": False}
    # The manifest hash matches the on-disk ScoreSet content hash.
    assert manifest["score_set_hash"] == _score_set_hash(on_disk_score_set)

    # READER: round-trips into a Predictions-consumable form, validates the hash, recovers the arrays.
    read = read_native_results(run_dir)
    assert read["score_set"] == raw_score_set
    assert read["predictions"].num_predictions == result.num_predictions

    # The fold-0 val direct-block row round-trips y_pred / y_true / sample_indices == the RunResult.
    rt_val = read["predictions"].filter_predictions(partition="val", fold_id="0")
    og_val = result.predictions.filter_predictions(partition="val", fold_id="0")
    assert len(rt_val) == 1 and len(og_val) == 1
    assert np.allclose(np.asarray(rt_val[0]["y_pred"]).ravel(), np.asarray(og_val[0]["y_pred"]).ravel())
    assert np.allclose(np.asarray(rt_val[0]["y_true"]).ravel(), np.asarray(og_val[0]["y_true"]).ravel())
    assert sorted(rt_val[0]["sample_indices"]) == sorted(int(i) for i in og_val[0]["sample_indices"])

    # The final-test direct-block row round-trips too.
    rt_test = read["predictions"].filter_predictions(partition="test", fold_id="final")
    og_test = result.predictions.filter_predictions(partition="test", fold_id="final")
    assert len(rt_test) == 1 and len(og_test) == 1
    assert np.allclose(np.asarray(rt_test[0]["y_pred"]).ravel(), np.asarray(og_test[0]["y_pred"]).ravel())


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_reader_detects_tampering(tmp_path: Path) -> None:
    """The reader VALIDATES the ScoreSet against the manifest hash — an edited score_set.json raises."""
    results_root = tmp_path / "results"
    run_via_dagml(
        [SNV(), KFold(n_splits=_N_SPLITS), {"model": PLSRegression(n_components=5)}], dataset_path("regression"),
        workdir=tmp_path / "work", dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    run_dir = sorted(results_root.iterdir())[0]
    score_set = json.loads((run_dir / "score_set.json").read_text())
    score_set["reports"].append({"tampered": True})
    (run_dir / "score_set.json").write_text(json.dumps(score_set))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_native_results(run_dir)


# ---------------------------------------------------------------------------
# (b) OFF by default — writes NOTHING + result unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_off_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No results_path + unset env → NOTHING written, no ./nirs4all_results, result still valid."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    result = run_via_dagml(
        [SNV(), KFold(n_splits=_N_SPLITS), {"model": PLSRegression(n_components=5)}], dataset_path("regression"),
        workdir=tmp_path / "work", dagml_cli=str(_DAGML_CLI), venv_python=sys.executable,
    )
    after = set(os.listdir(tmp_path))
    assert after - before <= {"work"}, "only the explicit workdir may appear; no native results dir"
    assert not (tmp_path / "nirs4all_results").exists()
    assert result.num_predictions > 0


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_off_is_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A public run() with results_path=None is behaviorally identical (no native dir written)."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    monkeypatch.chdir(tmp_path)
    result = nirs4all.run(
        [SNV(), KFold(n_splits=_N_SPLITS), {"model": PLSRegression(n_components=5)}], dataset_path("regression"), engine="dag-ml",
    )
    assert isinstance(result, RunResult)
    assert result.num_predictions > 0
    assert not (tmp_path / "nirs4all_results").exists()


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_public_run_results_path_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The public run(results_path=...) threads past the kwarg allowlist (named param) and writes."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    results_root = tmp_path / "results"
    result = nirs4all.run(
        [SNV(), KFold(n_splits=_N_SPLITS), {"model": PLSRegression(n_components=5)}], dataset_path("regression"),
        engine="dag-ml", results_path=str(results_root),
    )
    assert isinstance(result, RunResult) and result.num_predictions > 0
    run_dirs = sorted(results_root.iterdir())
    assert len(run_dirs) == 1
    # The 3 core files + the artifacts/ model tree (the in-process path captures the fitted REFIT model).
    assert {p.name for p in run_dirs[0].iterdir()} == {"manifest.json", "score_set.json", "predictions.parquet", "artifacts"}


# ---------------------------------------------------------------------------
# (c) LEGACY workspace untouched.
# ---------------------------------------------------------------------------


def test_legacy_engine_ignores_results_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """engine="legacy" never writes native results, even with results_path set (additive, dag-ml-only)."""
    monkeypatch.delenv("N4A_NATIVE_RESULTS", raising=False)
    results_root = tmp_path / "results"
    result = nirs4all.run(
        [SNV(), KFold(n_splits=_N_SPLITS), {"model": PLSRegression(n_components=5)}], dataset_path("regression"),
        engine="legacy", results_path=str(results_root), save_artifacts=False, save_charts=False, verbose=0,
    )
    assert isinstance(result, RunResult)
    assert not results_root.exists(), "the legacy path must not write native results"


# ---------------------------------------------------------------------------
# (d) SWEEP — per-variant predictions.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_sweep_per_variant(tmp_path: Path) -> None:
    """An operator (_or_) SWEEP writes per-variant prediction rows (both variants' configs present)."""
    results_root = tmp_path / "results"
    pipeline = [{"_or_": [SNV(), MinMaxScaler()]}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    run_dir = sorted(results_root.iterdir())[0]
    read = read_native_results(run_dir)
    assert read["predictions"].num_predictions == result.num_predictions

    rows = read["predictions"].filter_predictions(load_arrays=False)
    distinct_configs = {r["config_name"] for r in rows}
    # Two variants (their CV blocks) + the winner's _refit rows → at least 2 distinct base config names.
    base_configs = {c for c in distinct_configs if not c.endswith("_refit")}
    assert len(base_configs) >= 2, f"sweep must carry per-variant configs, got {distinct_configs}"

    # Both variants' fold-0 val rows are present and FILLED (each from its own model).
    val_fold0 = read["predictions"].filter_predictions(partition="val", fold_id="0")
    assert len(val_fold0) == 2, "two variants → two fold-0 val rows"
    assert all(len(np.asarray(r["y_pred"])) > 0 for r in val_fold0), "each variant's val arrays are filled"


# ---------------------------------------------------------------------------
# (e) MULTI-TARGET — round-trips WITH shape.
# ---------------------------------------------------------------------------


def _multi_target_dataset(n_targets: int = 3) -> SpectroDataset:
    """A 3-target SpectroDataset from the regression corpus (y = [y0, 2*y0+1, cos(y0)])."""
    base = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = [int(s) for s in base.index_column("sample", {"partition": "train"})]
    test = [int(s) for s in base.index_column("sample", {"partition": "test"})]

    def _xy(ids: list[int]):
        x = np.asarray(base.x_rows(ids, layout="2d"))
        y_block = np.asarray(base.y({"sample": ids}), dtype=float).reshape(len(ids), -1)
        stored = base.index_column("sample", {"sample": ids})
        row_of = {int(s): r for r, s in enumerate(stored)}
        y0 = y_block[[row_of[int(s)] for s in ids], 0]
        cols = [y0, y0 * 2.0 + 1.0, np.cos(y0)][:n_targets]
        return x, np.column_stack(cols)

    x_train, y_train = _xy(train)
    x_test, y_test = _xy(test)
    dataset = SpectroDataset("multi_target_regression")
    headers = [str(i) for i in range(x_train.shape[1])]
    dataset.add_samples(x_train, {"partition": "train"}, headers=headers, header_unit="nm")
    dataset.add_samples(x_test, {"partition": "test"}, headers=headers, header_unit="nm")
    dataset.add_targets(np.vstack([y_train, y_test]))
    return dataset


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_multi_target_round_trips_shape(tmp_path: Path) -> None:
    """A multi-target run carries SHAPE in the parquet so 2D y_pred/y_true recover their target width."""
    import polars as pl

    results_root = tmp_path / "results"
    dataset = _multi_target_dataset(3)
    pipeline = [KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset, workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    run_dir = sorted(results_root.iterdir())[0]

    # The parquet carries the 2D shape + target_width on the direct-block rows.
    df = pl.read_parquet(run_dir / "predictions.parquet")
    filled = df.filter(pl.col("arrays_present"))
    assert filled.height > 0
    sample = filled.row(0, named=True)
    assert len(sample["y_pred_shape"]) == 2 and sample["y_pred_shape"][1] == 3
    assert sample["target_width"] == 3

    # The reader reshapes back to 2D and the values round-trip == the RunResult.
    read = read_native_results(run_dir)
    rt_val = read["predictions"].filter_predictions(partition="val", fold_id="0")
    og_val = result.predictions.filter_predictions(partition="val", fold_id="0")
    rt_pred = np.asarray(rt_val[0]["y_pred"])
    assert rt_pred.ndim == 2 and rt_pred.shape[1] == 3, "multi-target y_pred recovers its (n, 3) shape"
    assert np.allclose(rt_pred, np.asarray(og_val[0]["y_pred"]))
    assert np.allclose(np.asarray(rt_val[0]["y_true"]), np.asarray(og_val[0]["y_true"]))


# ---------------------------------------------------------------------------
# (f) MODEL ARTIFACTS — capture + persist + verify-then-load round-trip (P3 Slice 2c-i).
# ---------------------------------------------------------------------------


def _refit_x(dataset_key: str) -> np.ndarray:
    """The held-out test feature matrix (2D) — the X a rehydrated estimator predicts in the round-trip."""
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    test_ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    return np.asarray(base.x_rows(test_ids, layout="2d"))


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_model_artifact_round_trip(tmp_path: Path) -> None:
    """In-process + native results ON: the manifest flags + lists a model artifact with a fingerprint +
    size_bytes; the reader rehydrates the estimator; the rehydrated estimator's predictions MATCH the
    in-memory captured model's, exactly (round-trip)."""
    results_root = tmp_path / "results"
    pipeline = [SNV(), {"y_processing": MinMaxScaler()}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )

    # The RunResult carries the captured fitted REFIT estimator(s) — at least the single model node's.
    captured = result._dagml_refit_artifacts  # noqa: SLF001
    assert len(captured) == 1, "one model node → one captured REFIT artifact"
    in_mem = captured[0]
    assert in_mem["estimator"] is not None and in_mem["y_transform"] is not None, "y_processing → a captured y_transform"

    run_dir = sorted(results_root.iterdir())[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["capabilities"]["has_model_artifacts"] is True
    assert len(manifest["artifacts"]) == 1
    ref = manifest["artifacts"][0]
    # ArtifactRef-identical fields (NOT "content_hash"): backend / uri / content_fingerprint / size_bytes.
    # backend is the SERIALIZATION backend (dag-ml ArtifactBackend, per ADR-16) — "joblib", NOT the ML framework.
    assert ref["backend"] == "joblib"
    assert ref["uri"].startswith("artifacts/") and ref["uri"].endswith(".joblib")
    assert len(ref["content_fingerprint"]) == 64  # sha256 hex
    assert ref["size_bytes"] == (run_dir / ref["uri"]).stat().st_size
    assert (run_dir / ref["uri"]).exists()

    # READER rehydrates the estimator; its predictions MATCH the in-memory captured model's, exactly.
    read = read_native_results(run_dir)
    assert len(read["artifacts"]) == 1
    rehydrated = read["artifacts"][0]
    x_test = _refit_x("regression")
    expected = np.asarray(in_mem["estimator"].predict(x_test), dtype=float)
    actual = np.asarray(rehydrated["estimator"].predict(x_test), dtype=float)
    assert np.array_equal(expected, actual), "the rehydrated estimator reproduces the in-memory model's predictions exactly"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_stacking_replay_manifest(tmp_path: Path) -> None:
    """A native stacking run persists the base/meta artifact order needed for raw-X bundle replay."""
    from sklearn.linear_model import Ridge

    from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

    results_root = tmp_path / "results"
    pipeline = [
        KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42),
        {
            "branch": [
                [SNV(), {"model": PLSRegression(n_components=5)}],
                [MSC(), {"model": PLSRegression(n_components=5)}],
            ]
        },
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0, random_state=42)},
    ]
    result = run_via_dagml(
        pipeline, dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
        random_state=42,
    )
    assert len(result._dagml_refit_artifacts) == 3  # noqa: SLF001

    run_dir = sorted(results_root.iterdir())[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    replay = manifest["stacking_replay"]
    assert replay["schema_version"] == 1
    assert replay["producer_node"] == "merge:stack"
    assert replay["meta_artifact_id"] == "artifact:merge:stack:nirs4all:refit:variant:base"
    assert replay["meta_feature_construction"] == {
        "kind": "base_prediction_column_stack",
        "producer_order": "sorted_prediction_input_base_key",
        "prediction_space": "original_target",
        "column_blocks": "one block per base producer, preserving target column order",
    }
    assert replay["base_producers"] == [
        {
            "artifact_id": "artifact:branch:0.node:1:nirs4all:refit:variant:base",
            "producer_node": "branch:0.node:1",
            "meta_feature_key": "branch:0.node:1.oof",
            "column_block": "prediction_values",
            "branch_index": 0,
        },
        {
            "artifact_id": "artifact:branch:1.node:1:nirs4all:refit:variant:base",
            "producer_node": "branch:1.node:1",
            "meta_feature_key": "branch:1.node:1.oof",
            "column_block": "prediction_values",
            "branch_index": 1,
        },
    ]

    artifact_refs = {artifact["artifact_id"]: artifact for artifact in manifest["artifacts"]}
    assert artifact_refs["artifact:branch:0.node:1:nirs4all:refit:variant:base"]["producer_node"] == "branch:0.node:1"
    assert artifact_refs["artifact:branch:1.node:1:nirs4all:refit:variant:base"]["producer_node"] == "branch:1.node:1"
    assert artifact_refs["artifact:merge:stack:nirs4all:refit:variant:base"]["producer_node"] == "merge:stack"

    read = read_native_results(run_dir)
    rehydrated = {artifact["artifact_id"]: artifact for artifact in read["artifacts"]}
    assert rehydrated["artifact:branch:0.node:1:nirs4all:refit:variant:base"]["producer_node"] == "branch:0.node:1"
    assert rehydrated["artifact:merge:stack:nirs4all:refit:variant:base"]["producer_node"] == "merge:stack"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_model_artifact_tamper_raises_before_load(tmp_path: Path) -> None:
    """A content_fingerprint mismatch on a model artifact raises ValueError BEFORE joblib.load (verify-then-load)."""
    results_root = tmp_path / "results"
    run_via_dagml(
        [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}],
        dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    run_dir = sorted(results_root.iterdir())[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    artifact_path = run_dir / manifest["artifacts"][0]["uri"]
    # Corrupt the artifact bytes so they no longer match the recorded content_fingerprint.
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="content_fingerprint mismatch"):
        read_native_results(run_dir)


def test_native_results_subprocess_has_no_model_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SUBPROCESS mechanism (Mechanism A) cannot reach the child-process models → has_model_artifacts:false,
    NO loadable artifacts[] entries (it never fakes a payload)."""
    if not _DAGML_CLI.exists():
        pytest.skip(f"dag-ml-cli binary not built at {_DAGML_CLI}")
    # Force the subprocess path (Mechanism A) so the models are fit in a child process this one can't capture.
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0")
    results_root = tmp_path / "results"
    result = run_via_dagml(
        [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}],
        dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    assert result._dagml_refit_artifacts == [], "the subprocess mechanism captures no fitted estimators"  # noqa: SLF001
    run_dir = sorted(results_root.iterdir())[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["capabilities"]["has_model_artifacts"] is False
    assert manifest["artifacts"] == []
    assert not (run_dir / "artifacts").exists(), "no artifacts/ payload when nothing was captured"
    # The reader returns no artifacts (and never tries to load a non-existent payload).
    assert read_native_results(run_dir)["artifacts"] == []


# ---------------------------------------------------------------------------
# (g) REP-FUSION captures + persists + rehydrates its model artifact(s) (P3 Slice 2c-i, MUST-FIX 1).
# ---------------------------------------------------------------------------


_REP_PHYS, _REP_REPS, _REP_FEAT = 12, 3, 20  # 12 physical samples × 3 equal replicates × 20 features


def _equal_rep_dataset() -> SpectroDataset:
    """A synthetic EQUAL-rep dataset (mirrors the cli_runner rep-fusion fixture): _REP_PHYS samples × _REP_REPS reps."""
    import polars as pl

    base = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    train = [int(s) for s in base.index_column("sample", {"partition": "train"})]
    n_rows = _REP_PHYS * _REP_REPS
    x = np.asarray(base.x_rows(train, layout="2d"))[:n_rows, :_REP_FEAT]
    y = np.asarray(base.y({"sample": train}), dtype=float).ravel()[:n_rows]
    sample_ids = [f"p{phys}" for phys in range(_REP_PHYS) for _ in range(_REP_REPS)]
    dataset = SpectroDataset("rep_fusion_synth")
    headers = [str(i) for i in range(_REP_FEAT)]
    dataset.add_samples([x], {"partition": "train"}, headers=[headers], header_unit="nm")
    dataset.add_targets(y.reshape(-1, 1))
    dataset.add_metadata(pl.DataFrame({"sample_id": sample_ids}))
    return dataset


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
@pytest.mark.parametrize("rep_keyword", ["rep_to_pp", "rep_to_sources"])
def test_native_results_rep_fusion_captures_artifacts(tmp_path: Path, rep_keyword: str) -> None:
    """A rep_to_pp / rep_to_sources run captures + persists + rehydrates its model artifact(s) (MUST-FIX 1:
    no run path drops refit_artifacts). The rep-fusion path is driven via _run_rep_fusion (its in-memory
    SpectroDataset isn't a reloadable path), then the native-results writer persists the captured model."""
    from nirs4all.pipeline.dagml.native_results import write_native_results
    from nirs4all.pipeline.dagml.run_backend import _detect_rep_fusion, _run_rep_fusion

    dataset = _equal_rep_dataset()
    pipeline = [{rep_keyword: "sample_id"}, KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    rep_step = _detect_rep_fusion(pipeline)
    assert rep_step == {rep_keyword: "sample_id"}
    result = _run_rep_fusion(pipeline, rep_step, dataset, "UNUSED", str(_DAGML_CLI), sys.executable, tmp_path / "work", "rmse", "regression")

    # MUST-FIX 1: the rep-fusion run captures its fitted REFIT model (the path no longer drops them).
    captured = result._dagml_refit_artifacts  # noqa: SLF001
    assert len(captured) == 1, f"{rep_keyword} run captures its REFIT model artifact"

    results_root = tmp_path / "results"
    run_dir = write_native_results(result, result._dagml_score_set, str(results_root))  # noqa: SLF001
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["capabilities"]["has_model_artifacts"] is True
    assert len(manifest["artifacts"]) == 1
    assert manifest["artifacts"][0]["backend"] == "joblib"

    read = read_native_results(run_dir)
    assert len(read["artifacts"]) == 1
    # The rehydrated estimator reproduces the in-memory captured model's predictions exactly (round-trip).
    # The reshape fuses _REP_REPS replicates into the feature axis, so the fitted model expects
    # _REP_REPS * _REP_FEAT columns — feed a matrix of that width (serialization fidelity, not reshape math).
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, _REP_REPS * _REP_FEAT))
    expected = np.asarray(captured[0]["estimator"].predict(x), dtype=float)
    actual = np.asarray(read["artifacts"][0]["estimator"].predict(x), dtype=float)
    assert np.array_equal(expected, actual)


# ---------------------------------------------------------------------------
# (h) READER SECURITY — backend refusal (MUST-FIX 2) + path-traversal guard (MUST-FIX 3).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_native_results_reader_refuses_unknown_backend(tmp_path: Path) -> None:
    """MUST-FIX 2: the reader refuses an artifact whose backend is not the joblib serialization backend,
    BEFORE any joblib.load (only backend=='joblib' is loadable here)."""
    results_root = tmp_path / "results"
    run_via_dagml(
        [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}],
        dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    run_dir = sorted(results_root.iterdir())[0]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifacts"][0]["backend"] == "joblib"
    # Edit the manifest to claim a different (non-joblib) backend → the reader must refuse it.
    manifest["artifacts"][0]["backend"] = "torch"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported backend"):
        read_native_results(run_dir)


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
@pytest.mark.parametrize("evil_uri", ["/etc/passwd", "../../../etc/passwd", "..\\..\\evil", "file:///etc/passwd", "C:\\evil", "artifacts/\x85nel.joblib", "artifacts/\x07bell.joblib"])
def test_native_results_reader_refuses_non_portable_uri(tmp_path: Path, evil_uri: str) -> None:
    """MUST-FIX 3: an edited manifest with an absolute / ``..`` traversal / scheme uri raises ValueError
    BEFORE any read_bytes or joblib.load (mirrors dag-ml's portable-URI contract)."""
    results_root = tmp_path / "results"
    run_via_dagml(
        [SNV(), KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}],
        dataset_path("regression"), workdir=tmp_path / "work",
        dagml_cli=str(_DAGML_CLI), venv_python=sys.executable, results_path=str(results_root),
    )
    run_dir = sorted(results_root.iterdir())[0]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["uri"] = evil_uri
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative path|`\\.\\.`|scheme or colon|control characters"):
        read_native_results(run_dir)
