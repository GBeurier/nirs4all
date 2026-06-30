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


# --------------------------------------------------------------------------- #
# Slice 3 (ADR-17 D-io RESIDUAL): multi-source / joins / convention-gated      #
# shapes / nio.infer. Each is parity-gated where the default loader builds an  #
# equivalent, and golden-asserted where it does not (io-owned concepts).       #
# --------------------------------------------------------------------------- #
def test_from_io_multisource_key_alignment_matches_default_loader(tmp_path):
    """Two feature sources (nir + mir): io KEY-join == default positional list.

    PARITY + ALIGNMENT. nirs4all's default loader builds a multi-source
    SpectroDataset when ``train_x`` is a list of paths (positionally aligned,
    same row count). nio builds the same multi-source shape from two
    ``role: features`` sources joined 1:1 by a shared ``id`` key.

    To prove the join keys (not on-disk row position) drive alignment, ``mir``
    and ``y`` are written to disk in SHUFFLED ``id`` order relative to ``nir``,
    and every row carries a value that encodes its id (nir col == id, mir col ==
    id+0.5, y == id*10+0.25). The default-loader oracle is built from KEYLESS
    files written in nir's expected joined order. ``from_io`` must re-align mir/y
    to nir's order by key and so match the oracle row-for-row; a positional
    implementation that ignored the keys would read mir/y in shuffled order and
    FAIL.
    """
    wl_a = [str(1000 + i * 5) for i in range(6)]
    wl_b = [str(2000 + i * 5) for i in range(4)]
    n = 12
    ids = list(range(n))
    # each row's values encode its id, so a misaligned join is detectable.
    nir_vals = np.array([[float(i)] * 6 for i in ids])
    mir_vals = np.array([[float(i) + 0.5] * 4 for i in ids])
    yv = np.array([float(i) * 10 + 0.25 for i in ids])  # float -> regression (raw y survives)

    nir_k = pd.DataFrame(nir_vals, columns=wl_a)
    nir_k.insert(0, "id", ids)
    mir_k = pd.DataFrame(mir_vals, columns=wl_b)
    mir_k.insert(0, "id", ids)
    y_k = pd.DataFrame({"id": ids, "y": yv})

    # SHUFFLE mir + y rows on disk to a different id order than nir.
    perm_rng = np.random.default_rng(99)
    mir_k = mir_k.iloc[perm_rng.permutation(n)].reset_index(drop=True)
    y_k = y_k.iloc[perm_rng.permutation(n)].reset_index(drop=True)
    assert list(mir_k["id"]) != ids and list(y_k["id"]) != ids  # genuinely shuffled
    _write(tmp_path / "nir.csv", nir_k)
    _write(tmp_path / "mir.csv", mir_k)
    _write(tmp_path / "y.csv", y_k)

    # default-loader oracle: keyless files in nir's EXPECTED joined (id) order.
    _write(tmp_path / "nir_pos.csv", pd.DataFrame(nir_vals, columns=wl_a))
    _write(tmp_path / "mir_pos.csv", pd.DataFrame(mir_vals, columns=wl_b))
    _write(tmp_path / "y_pos.csv", pd.DataFrame({"y": yv}))

    default_cfg = {
        "train_x": [str(tmp_path / "nir_pos.csv"), str(tmp_path / "mir_pos.csv")],
        "train_y": str(tmp_path / "y_pos.csv"),
        "delimiter": ";",
        "has_header": True,
    }
    io_spec = {
        "sample_index": {"by": "id", "key": "id"},
        "sources": [
            {"id": "nir", "role": "features", "input": "nir.csv", "key": "id",
             "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}]},
            {"id": "mir", "role": "features", "input": "mir.csv", "key": "id",
             "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}],
             "join": {"to": "nir", "on": "id", "how": "1:1"}},
            {"id": "y", "role": "targets", "input": "y.csv", "key": "id",
             "columns": [{"role": "targets", "select": ["y"]}],
             "join": {"to": "nir", "on": "id", "how": "1:1", "coverage": "complete"}},
        ],
    }

    ref = DatasetConfigs(dict(default_cfg)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(io_spec, base_dir=str(tmp_path)).get_dataset_at(0)

    assert ref.n_sources == mine.n_sources == 2
    # per-source equivalence (concat_source=False yields one array per source)
    ref_srcs = ref.x({"partition": "train"}, layout="2d", concat_source=False)
    mine_srcs = mine.x({"partition": "train"}, layout="2d", concat_source=False)
    assert len(ref_srcs) == len(mine_srcs) == 2
    for s, (rs, ms) in enumerate(zip(ref_srcs, mine_srcs)):
        rs_a, ms_a = np.asarray(rs, dtype=float), np.asarray(ms, dtype=float)
        assert rs_a.shape == ms_a.shape, f"source {s} shape: default {rs_a.shape} vs io {ms_a.shape}"
        np.testing.assert_allclose(rs_a, ms_a, rtol=1e-5, err_msg=f"source {s} X values differ")
        assert ref.headers(s) == mine.headers(s), f"source {s} headers differ"

    # explicit key-alignment assertion: each row was re-ordered to nir's id order,
    # so the encoded values land on the diagonal (nir==id, mir==id+0.5, y==id*10+.25).
    nir_out = np.asarray(mine_srcs[0], dtype=float)[:, 0]
    mir_out = np.asarray(mine_srcs[1], dtype=float)[:, 0]
    y_out = np.asarray(mine.y({"partition": "train"}), dtype=float).ravel()
    np.testing.assert_allclose(nir_out, np.array(ids, dtype=float))
    np.testing.assert_allclose(mir_out, np.array(ids, dtype=float) + 0.5)
    np.testing.assert_allclose(y_out, np.array(ids, dtype=float) * 10 + 0.25)

    # and the combined data plane / y / task match the positional oracle row-for-row
    _assert_dataset_equivalent(ref, mine, ("train",))


def test_from_io_bare_xy_folder_matches_default_loader(tmp_path):
    """Single-partition ``X.csv`` / ``y.csv`` folder: io path == default loader.

    PARITY. A bare ``X.csv`` / ``y.csv`` folder is recognized by both the
    ``nirs4all-classic`` profile's bare fallback (``x``/``y`` stems) and nirs4all's
    own ``FolderParser``, so both build the same single-source dataset. Passing
    the built-in ``bare`` profile is harmless here and exercises the convention
    argument on the parity path.
    """
    wl = [str(1000 + i * 5) for i in range(10)]
    rng = np.random.default_rng(31)
    _write(tmp_path / "X.csv", pd.DataFrame(rng.random((14, 10)), columns=wl))
    _write(tmp_path / "y.csv", pd.DataFrame({"y": rng.random(14) * 30}))

    ref = DatasetConfigs(str(tmp_path)).get_dataset_at(0)
    mine = DatasetConfigs.from_io(str(tmp_path), conventions=["bare"]).get_dataset_at(0)

    assert ref.n_sources == mine.n_sources == 1
    _assert_dataset_equivalent(ref, mine, ("train",))


def test_from_io_bare_convention_extends_beyond_default_loader_golden(tmp_path):
    """The ``bare`` profile loads stems the default loader can't: GOLDEN.

    A folder with non-``x``/``y`` stems (``spectra.csv`` / ``target.csv``) is
    convention-gated: nio's default ``nirs4all-classic`` profile raises
    ``SpecError``, and nirs4all's own ``FolderParser`` recognizes no dataset there
    either — so there is NO parity oracle. The built-in ``bare`` profile resolves
    the wider stem vocabulary (``spectra``/``spectrum``/``features`` -> features,
    ``target``/``targets``/``labels`` -> targets), letting ``from_io`` load a
    dataset the default loader cannot. Assert both the gating and the loaded shape.
    """
    nio_spec = pytest.importorskip("nirs4all_io.spec.enums")
    wl = [str(1000 + i * 5) for i in range(10)]
    rng = np.random.default_rng(32)
    _write(tmp_path / "spectra.csv", pd.DataFrame(rng.random((12, 10)), columns=wl))
    _write(tmp_path / "target.csv", pd.DataFrame({"y": rng.random(12) * 30}))

    # default conventions cannot match these stems -> SpecError (convention-gated)
    with pytest.raises(nio_spec.SpecError):
        DatasetConfigs.from_io(str(tmp_path)).get_dataset_at(0)

    # the bare profile resolves them; the default loader has no equivalent, so
    # this is a golden assertion on the io-materialized shape.
    ds = DatasetConfigs.from_io(str(tmp_path), conventions=["bare"]).get_dataset_at(0)
    assert ds.n_sources == 1
    assert ds.num_samples == 12
    assert np.asarray(ds.x({"partition": "train"}, layout="2d", concat_source=True)).shape == (12, 10)
    assert np.asarray(ds.y({"partition": "train"})).shape == (12, 1)


def test_from_io_relational_lookup_join_golden(tmp_path):
    """Vendor corpus + reference table (m:1 lookup): GOLDEN (no default oracle).

    A relational join of a measurement corpus (two concatenated batches, each
    keyed by ``site``) with a reference dimension table (``site -> region``) is
    an io-owned concept: the default loader builds no such dataset from a single
    config, so there is no parity oracle.

    The join must broadcast ``region`` **by key**, not by row position. To prove
    that, the per-row ``site`` order differs from the reference table's row order
    (``S3,S1,S2``), the two batches present sites in different orders, and each
    feature row carries a distinct marker in its first column. We then assert the
    EXACT per-row broadcast result — a row-misaligned m:1 join (e.g. one that
    pasted the reference rows positionally) would pair the wrong region with each
    row and fail this test, where a set-only assertion would still pass.
    """
    wl = [str(1000 + i * 5) for i in range(4)]
    zeros = [0.0, 0.0, 0.0]

    # batch_a: sites S1,S2,S3 ; batch_b: sites S2,S1,S3 (different order).
    # col0 carries a unique per-row marker (1..6) so feature row order is checkable.
    batch_a = pd.DataFrame({wl[0]: [1.0, 2.0, 3.0], wl[1]: zeros, wl[2]: zeros, wl[3]: zeros,
                            "protein": [11.0, 12.0, 13.0], "site": ["S1", "S2", "S3"]})
    batch_b = pd.DataFrame({wl[0]: [4.0, 5.0, 6.0], wl[1]: zeros, wl[2]: zeros, wl[3]: zeros,
                            "protein": [22.0, 21.0, 23.0], "site": ["S2", "S1", "S3"]})
    # reference table rows are in a DIFFERENT order (S3,S1,S2) than first appearance,
    # so a positional paste would scramble the region->row mapping.
    sites = pd.DataFrame({"site": ["S3", "S1", "S2"], "region": ["east", "north", "south"]})
    _write(tmp_path / "batch_a.csv", batch_a)
    _write(tmp_path / "batch_b.csv", batch_b)
    _write(tmp_path / "sites.csv", sites)

    spec = {
        "sources": [
            {"id": "m", "role": "mixed", "input": ["batch_a.csv", "batch_b.csv"], "merge": "concat_samples",
             "columns": [
                 {"role": "features", "select": wl},
                 {"role": "targets", "select": ["protein"]},
                 {"role": "metadata", "select": ["site"]},
             ]},
            {"id": "sites", "kind": "lookup", "input": "sites.csv",
             "columns": [{"role": "metadata", "select": ["region"]}],
             "join": {"to": "m", "on": "site", "how": "m:1", "coverage": "complete"}},
        ],
    }

    ds = DatasetConfigs.from_io(spec, base_dir=str(tmp_path)).get_dataset_at(0)

    # concat_samples over two 3-row batches -> 6 samples, single feature source.
    assert ds.n_sources == 1
    assert ds.num_samples == 6
    X = np.asarray(ds.x({"partition": "train"}, layout="2d", concat_source=True), dtype=float)
    assert X.shape == (6, 4)
    # feature rows are in concat order (batch_a markers 1,2,3 then batch_b 4,5,6).
    assert X[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # EXACT per-row broadcast: each row's region is the by-key lookup of its site,
    # NOT a positional paste of the reference table's row order.
    assert "region" in ds.metadata_columns
    site = np.asarray(ds.metadata_column("site", {"partition": "train"})).tolist()
    region = np.asarray(ds.metadata_column("region", {"partition": "train"})).tolist()
    assert site == ["S1", "S2", "S3", "S2", "S1", "S3"]
    assert region == ["north", "south", "east", "south", "north", "east"]


def test_from_io_infer_returns_scored_plan_golden(tmp_path):
    """``from_io_infer`` returns a scored DatasetPlan + materializes it: GOLDEN.

    ``nio.infer`` is a planning API: its confidence-scored ``DatasetPlan`` (an
    uncalibrated ranking, not a probability) has no default-loader equivalent, so
    there is no parity oracle. Assert the plan's key properties for a known
    classic folder, and that the configs it returns materialize the expected
    train/test shape.
    """
    wl = [str(1000 + i * 5) for i in range(10)]
    rng = np.random.default_rng(33)
    _write(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((20, 10)), columns=wl))
    _write(tmp_path / "Ycal.csv", pd.DataFrame({"y": rng.random(20) * 50}))
    _write(tmp_path / "Xval.csv", pd.DataFrame(rng.random((8, 10)), columns=wl))
    _write(tmp_path / "Yval.csv", pd.DataFrame({"y": rng.random(8) * 50}))

    configs, plan = DatasetConfigs.from_io_infer(str(tmp_path))

    # the plan recommends the classic train/test folder topology with a high score
    assert plan.structure is not None
    assert plan.structure.value == "train_test_folder"
    assert plan.structure.score >= 0.5
    assert plan.task_type is not None and plan.task_type.value == "regression"
    assert 0.0 <= plan.overall_score <= 1.0
    assert plan.resolved_spec is not None  # the spec load executed
    # scores are explicitly uncalibrated rankings, not probabilities (io C5).
    assert plan.calibration["method"] == "none"

    # and the returned configs materialize the inferred shape.
    ds = configs.get_dataset_at(0)
    assert ds.n_sources == 1
    assert ds.task_type.value == "regression"
    assert np.asarray(ds.x({"partition": "train"}, layout="2d", concat_source=True)).shape == (20, 10)
    assert np.asarray(ds.x({"partition": "test"}, layout="2d", concat_source=True)).shape == (8, 10)


def test_from_io_infer_task_type_override(tmp_path):
    """A constructor override threads through the infer path identically to from_io.

    The small-integer y infers as classification; ``task_type='regression'`` must
    flip the materialized dataset, exactly as it does on the plain ``from_io`` path.
    """
    wl = [str(1000 + i * 5) for i in range(8)]
    rng = np.random.default_rng(34)
    _write(tmp_path / "X.csv", pd.DataFrame(rng.random((30, 8)), columns=wl))
    _write(tmp_path / "y.csv", pd.DataFrame({"y": rng.integers(0, 3, 30)}))

    auto_cfg, _auto_plan = DatasetConfigs.from_io_infer(str(tmp_path), conventions=["bare"])
    assert auto_cfg.get_dataset_at(0).task_type.value == "multiclass_classification"

    forced_cfg, _ = DatasetConfigs.from_io_infer(str(tmp_path), conventions=["bare"], task_type="regression")
    assert forced_cfg.get_dataset_at(0).task_type.value == "regression"
