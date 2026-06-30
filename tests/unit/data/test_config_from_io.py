"""Parity gate for the opt-in nirs4all-io input path (ADR-17 D-io, slice 1).

``DatasetConfigs.from_io(...)`` must produce a ``SpectroDataset`` equivalent to
the one the default ``DatasetConfigs(...)`` loader builds for the same input
(same partitions / X / y / wavelengths / task type). The opt-in path delegates
assembly to ``nirs4all-io`` (``nio.load``); the default path is unchanged.

These tests auto-skip when ``nirs4all-io`` is not installed, so the suite stays
green where the bridge is absent.
"""

import numpy as np
import pandas as pd
import pytest

from nirs4all.data.config import DatasetConfigs

# The opt-in path is only exercisable when the bridge is installed.
pytest.importorskip("nirs4all_io", reason="opt-in io path needs nirs4all-io installed")


def _assert_dataset_equivalent(ref, mine, partitions, *, compare_headers=True):
    """Data-plane equivalence between two SpectroDatasets over the given partitions.

    When ``compare_headers`` is False the header *strings* are not compared (only
    their count) — used for the in-memory-array shapes, where the default loader
    auto-names columns ``feature_i`` while the io path uses index headers ``str(i)``.
    These are synthetic, non-load-bearing labels (no real wavelengths exist for
    bare arrays), so the data plane (X / y / task_type / partitions) is what must match.
    """
    for part in partitions:
        sel = {"partition": part}
        ref_x = ref.x(sel, layout="2d", concat_source=True)
        mine_x = mine.x(sel, layout="2d", concat_source=True)
        assert ref_x.shape == mine_x.shape, f"{part} X shape: default {ref_x.shape} vs io {mine_x.shape}"
        np.testing.assert_allclose(
            np.asarray(ref_x, dtype=float), np.asarray(mine_x, dtype=float), rtol=1e-5,
            err_msg=f"{part} X values differ between default loader and io path",
        )
        np.testing.assert_allclose(
            ref.y(sel).ravel().astype(float), mine.y(sel).ravel().astype(float), rtol=1e-5,
            err_msg=f"{part} y values differ between default loader and io path",
        )
    assert ref.task_type.value == mine.task_type.value
    if compare_headers:
        assert ref.headers(0) == mine.headers(0)
    else:
        assert len(ref.headers(0)) == len(mine.headers(0))


def _write(path, df):
    df.to_csv(path, sep=";", index=False)


def test_from_io_matches_default_loader_dict_config(tmp_path):
    """Explicit train/test dict config: io path == default loader."""
    wl = [str(1000 + i * 5) for i in range(10)]
    rng = np.random.default_rng(0)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((20, 10)) * 2.0 + 0.3, columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(20) * 50}))
    _write(tmp_path / "Xval.csv", pd.DataFrame(rng.random((8, 10)) * 2.0 + 0.3, columns=wl))
    _write(tmp_path / "Yval.csv", pd.DataFrame({"y": rng.random(8) * 50}))
    cfg = {
        "train_x": str(tmp_path / "Xcal.csv"),
        "train_y": str(tmp_path / "Ycal.csv"),
        "test_x": str(tmp_path / "Xval.csv"),
        "test_y": str(tmp_path / "Yval.csv"),
        "delimiter": ";",
        "has_header": True,
    }

    ref = DatasetConfigs(dict(cfg)).get_dataset_at(0)          # default loader path (unchanged)
    mine = DatasetConfigs.from_io(dict(cfg)).get_dataset_at(0)  # opt-in nirs4all-io path

    _assert_dataset_equivalent(ref, mine, ("train", "test"))


def test_from_io_matches_default_loader_folder(tmp_path):
    """Classic Xcal/Ycal/Xval/Yval folder layout: io path == default loader."""
    wl = [str(1000 + i * 5) for i in range(12)]
    rng = np.random.default_rng(1)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((25, 12)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(25) * 30}))
    _write(tmp_path / "Xval.csv", pd.DataFrame(rng.random((9, 12)), columns=wl))
    _write(tmp_path / "Yval.csv", pd.DataFrame({"y": rng.random(9) * 30}))

    ref = DatasetConfigs(str(tmp_path)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(str(tmp_path)).get_dataset_at(0)

    _assert_dataset_equivalent(ref, mine, ("train", "test"))


# --------------------------------------------------------------------------- #
# Slice 2: more input shapes, each parity-gated against the default loader.    #
# --------------------------------------------------------------------------- #
def test_from_io_matches_default_loader_in_memory_arrays():
    """In-memory (X, y, split) tuple: io path == default dict-of-arrays loader.

    nio.load's in-memory shapes are ``X`` / ``(X, y)`` / ``(X, y, split)`` /
    ``{"X":..., "y":...}`` — the train/test split is expressed by a per-row label
    array. The default loader expresses the same split with explicit
    ``train_x``/``test_x`` array blocks; both must produce the same data plane.
    """
    rng = np.random.default_rng(2)
    n_train, n_test, n_feat = 20, 8, 10
    x = rng.random((n_train + n_test, n_feat)).astype(np.float32)
    y = (rng.random(n_train + n_test) * 50).astype(np.float32)
    split = np.array(["train"] * n_train + ["test"] * n_test)

    # io path: a single (X, y, split) tuple.
    mine = DatasetConfigs.from_io((x, y, split)).get_dataset_at(0)
    # default loader: explicit train/test array blocks.
    ref = DatasetConfigs(
        {"train_x": x[:n_train], "train_y": y[:n_train], "test_x": x[n_train:], "test_y": y[n_train:]}
    ).get_dataset_at(0)

    _assert_dataset_equivalent(ref, mine, ("train", "test"), compare_headers=False)


def test_from_io_matches_default_loader_config_json(tmp_path):
    """Config JSON file: io path == default config-file loader."""
    import json

    wl = [str(1000 + i * 5) for i in range(10)]
    rng = np.random.default_rng(3)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((22, 10)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(22) * 40}))
    _write(tmp_path / "Xval.csv", pd.DataFrame(rng.random((7, 10)), columns=wl))
    _write(tmp_path / "Yval.csv", pd.DataFrame({"y": rng.random(7) * 40}))
    cfg = {
        "train_x": str(tmp_path / "Xcal.csv"),
        "train_y": str(tmp_path / "Ycal.csv"),
        "test_x": str(tmp_path / "Xval.csv"),
        "test_y": str(tmp_path / "Yval.csv"),
        "delimiter": ";",
        "has_header": True,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    ref = DatasetConfigs(str(cfg_path)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(str(cfg_path)).get_dataset_at(0)

    _assert_dataset_equivalent(ref, mine, ("train", "test"))


def test_from_io_matches_default_loader_config_yaml(tmp_path):
    """Config YAML file: io path == default config-file loader."""
    import yaml

    wl = [str(1000 + i * 5) for i in range(10)]
    rng = np.random.default_rng(4)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((18, 10)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(18) * 40}))
    cfg = {
        "train_x": str(tmp_path / "Xcal.csv"),
        "train_y": str(tmp_path / "Ycal.csv"),
        "delimiter": ";",
        "has_header": True,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    ref = DatasetConfigs(str(cfg_path)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(str(cfg_path)).get_dataset_at(0)

    _assert_dataset_equivalent(ref, mine, ("train",))


def test_from_io_matches_default_loader_file_list(tmp_path):
    """Classic 4-file list: io path == default directory loader.

    A file list / glob is resolved + convention-matched by nio into the same
    classic Xcal/Ycal/Xval/Yval topology the default loader builds from the
    enclosing directory.
    """
    import glob

    wl = [str(1000 + i * 5) for i in range(12)]
    rng = np.random.default_rng(5)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((24, 12)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(24) * 30}))
    _write(tmp_path / "Xval.csv", pd.DataFrame(rng.random((9, 12)), columns=wl))
    _write(tmp_path / "Yval.csv", pd.DataFrame({"y": rng.random(9) * 30}))

    file_list = sorted(glob.glob(str(tmp_path / "*.csv")))

    ref = DatasetConfigs(str(tmp_path)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(file_list).get_dataset_at(0)

    _assert_dataset_equivalent(ref, mine, ("train", "test"))


# --------------------------------------------------------------------------- #
# Slice 2: constructor-level override threading, parity-gated.                 #
# --------------------------------------------------------------------------- #
def test_from_io_task_type_override_matches_default_loader(tmp_path):
    """Forcing task_type through from_io == forcing it through the constructor.

    The small-integer y auto-detects as classification; the override must flip
    both paths to regression identically.
    """
    wl = [str(1000 + i * 5) for i in range(8)]
    rng = np.random.default_rng(6)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((30, 8)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.integers(0, 3, 30)}))
    cfg = {
        "train_x": str(tmp_path / "Xcal.csv"),
        "train_y": str(tmp_path / "Ycal.csv"),
        "delimiter": ";",
        "has_header": True,
    }

    # Sanity: without the override both paths auto-detect classification.
    auto_ref = DatasetConfigs(dict(cfg)).get_dataset_at(0)
    assert auto_ref.task_type.value == "multiclass_classification"

    ref = DatasetConfigs(dict(cfg), task_type="regression").get_dataset_at(0)
    mine = DatasetConfigs.from_io(dict(cfg), task_type="regression").get_dataset_at(0)

    assert ref.task_type.value == "regression"
    _assert_dataset_equivalent(ref, mine, ("train",))


def test_from_io_repetition_aggregate_overrides_match_default_loader(tmp_path):
    """repetition / aggregate / signal_type overrides == the constructor's.

    Threads every constructor-level override and asserts both the data plane and
    the resulting repetition/aggregation state match the default loader.
    """
    wl = [str(1000 + i * 5) for i in range(8)]
    rng = np.random.default_rng(8)
    n = 24
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((n, 8)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(n) * 40}))
    _write(tmp_path / "Mcal.csv", pd.DataFrame({"sample_id": np.repeat(np.arange(n // 2), 2)}))
    cfg = {
        "train_x": str(tmp_path / "Xcal.csv"),
        "train_y": str(tmp_path / "Ycal.csv"),
        "train_m": str(tmp_path / "Mcal.csv"),
        "delimiter": ";",
        "has_header": True,
    }

    overrides = {
        "repetition": "sample_id",
        "aggregate_method": "median",
        "aggregate_exclude_outliers": True,
        "signal_type": "absorbance",
    }
    ref = DatasetConfigs(dict(cfg), **overrides).get_dataset_at(0)
    mine = DatasetConfigs.from_io(dict(cfg), **overrides).get_dataset_at(0)

    _assert_dataset_equivalent(ref, mine, ("train",))
    assert ref.repetition == mine.repetition == "sample_id"
    assert ref.aggregate == mine.aggregate
    assert ref.aggregate_method == mine.aggregate_method == "median"
    assert ref.aggregate_exclude_outliers == mine.aggregate_exclude_outliers is True
    assert ref.signal_type(0) == mine.signal_type(0)


# --------------------------------------------------------------------------- #
# Slice 2 (Codex gate): override PRECEDENCE must match __init__ exactly when a #
# config-level value and a constructor override interact.                      #
# --------------------------------------------------------------------------- #
def _config_with_groups(tmp_path, *, n=24):
    """Write a 1-source dataset whose metadata carries two grouping columns."""
    wl = [str(1000 + i * 5) for i in range(8)]
    rng = np.random.default_rng(20)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((n, 8)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(n) * 40}))
    _write(
        tmp_path / "Mcal.csv",
        pd.DataFrame({"sample_id": np.repeat(np.arange(n // 2), 2), "batch": np.repeat(np.arange(n // 4), 4)}),
    )
    return {
        "train_x": str(tmp_path / "Xcal.csv"),
        "train_y": str(tmp_path / "Ycal.csv"),
        "train_m": str(tmp_path / "Mcal.csv"),
        "delimiter": ";",
        "has_header": True,
    }


def _assert_agg_state_equal(ref, mine):
    assert ref.repetition == mine.repetition, f"repetition: default {ref.repetition} vs io {mine.repetition}"
    assert ref.aggregate == mine.aggregate, f"aggregate: default {ref.aggregate} vs io {mine.aggregate}"
    assert ref.aggregate_method == mine.aggregate_method
    assert ref.aggregate_exclude_outliers == mine.aggregate_exclude_outliers


def test_from_io_config_repetition_with_override_aggregate(tmp_path):
    """Config-level repetition is preserved when only aggregate is overridden.

    Default loader: repetition=sample_id (config) + aggregate=batch (override).
    The config-level repetition must NOT be dropped (which would make the None
    repetition wrongly auto-inherit from the overridden aggregate).
    """
    cfg = _config_with_groups(tmp_path)
    cfg["repetition"] = "sample_id"

    ref = DatasetConfigs(dict(cfg), aggregate="batch").get_dataset_at(0)
    mine = DatasetConfigs.from_io(dict(cfg), aggregate="batch").get_dataset_at(0)

    assert ref.repetition == "sample_id" and ref.aggregate == "batch"
    _assert_dataset_equivalent(ref, mine, ("train",))
    _assert_agg_state_equal(ref, mine)


def test_from_io_config_aggregate_with_override_repetition(tmp_path):
    """Config-level aggregate is preserved when only repetition is overridden."""
    cfg = _config_with_groups(tmp_path)
    cfg["aggregate"] = "batch"

    ref = DatasetConfigs(dict(cfg), repetition="sample_id").get_dataset_at(0)
    mine = DatasetConfigs.from_io(dict(cfg), repetition="sample_id").get_dataset_at(0)

    assert ref.repetition == "sample_id" and ref.aggregate == "batch"
    _assert_dataset_equivalent(ref, mine, ("train",))
    _assert_agg_state_equal(ref, mine)


def test_from_io_explicit_false_overrides_config_exclude_outliers_true(tmp_path):
    """An explicit aggregate_exclude_outliers=False cancels a config-level True.

    The constructor distinguishes an explicit False from the unset default; the
    io path must do the same so config-True + override-False yields False.
    """
    cfg = _config_with_groups(tmp_path)
    cfg["aggregate"] = "sample_id"
    cfg["aggregate_exclude_outliers"] = True

    ref = DatasetConfigs(dict(cfg), aggregate_exclude_outliers=False).get_dataset_at(0)
    mine = DatasetConfigs.from_io(dict(cfg), aggregate_exclude_outliers=False).get_dataset_at(0)

    assert ref.aggregate_exclude_outliers is False
    _assert_dataset_equivalent(ref, mine, ("train",))
    _assert_agg_state_equal(ref, mine)


def test_from_io_unset_exclude_outliers_inherits_config_true(tmp_path):
    """With no override, the io path inherits a config-level exclude_outliers=True."""
    cfg = _config_with_groups(tmp_path)
    cfg["aggregate"] = "sample_id"
    cfg["aggregate_exclude_outliers"] = True

    ref = DatasetConfigs(dict(cfg)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(dict(cfg)).get_dataset_at(0)

    assert ref.aggregate_exclude_outliers is True
    _assert_dataset_equivalent(ref, mine, ("train",))
    _assert_agg_state_equal(ref, mine)
