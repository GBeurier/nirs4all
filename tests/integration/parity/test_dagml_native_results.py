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
    assert {p.name for p in run_dir.iterdir()} == {"manifest.json", "score_set.json", "predictions.parquet"}

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
    assert manifest["capabilities"] == {"has_model_artifacts": False, "has_aggregate_predictions": False}
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
    assert {p.name for p in run_dirs[0].iterdir()} == {"manifest.json", "score_set.json", "predictions.parquet"}


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
        x = np.asarray(base.x_rows(ids, layout="2d"), dtype=float)
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
