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


def _assert_dataset_equivalent(ref, mine, partitions):
    """Data-plane equivalence between two SpectroDatasets over the given partitions."""
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
    assert ref.headers(0) == mine.headers(0)


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
