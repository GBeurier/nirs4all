"""Observable regression witnesses for logging and generator audit findings."""

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from nirs4all.core.logging.formatters import FileFormatter, JsonFormatter
from nirs4all.core.logging.handlers import RotatingRunFileHandler
from nirs4all.pipeline.config._generator.presets import register_preset, resolve_presets_recursive
from nirs4all.pipeline.config.generator import count_combinations, expand_spec
from nirs4all.workspace.compat import _sqlite_has_prediction_arrays


def test_log_size_rotation_finishes_and_retains_written_record(tmp_path):
    handler = RotatingRunFileHandler(tmp_path, "run", max_bytes=1, compress_rotated=False)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            record = logging.LogRecord("audit", logging.INFO, "audit", 1, "kept", (), None)
            executor.submit(handler.emit, record).result(timeout=2)
        assert "kept" in (tmp_path / "run.log").read_text()
    finally:
        handler.close()


def test_log_retention_preserves_current_log(tmp_path):
    (tmp_path / "old.log").write_text("old")
    handler = RotatingRunFileHandler(tmp_path, "current", max_runs=1, max_age_days=None)
    try:
        assert (tmp_path / "current.log").is_file()
        assert not (tmp_path / "old.log").exists()
    finally:
        handler.close()


def test_file_formatter_does_not_change_other_handlers_record():
    record = logging.LogRecord("audit", logging.INFO, "audit", 1, "value %s", (3,), None)
    record.branch_name = "branch_a"
    formatter = FileFormatter()
    first = formatter.format(record)
    assert formatter.format(record) == first
    assert record.msg == "value %s"
    assert record.getMessage() == "value 3"
    assert "value 3 [branch=" not in JsonFormatter().format(record)


def test_workspace_probe_encodes_special_uri_characters(tmp_path):
    path = tmp_path / "store?#1.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE prediction_arrays (id TEXT)")
    assert _sqlite_has_prediction_arrays(path)
    assert not (tmp_path / "store").exists()


def test_preset_keeps_explicit_sibling_model():
    register_preset("audit_ridge_grid", {"_grid_": {"alpha": [1, 2]}})
    resolved = resolve_presets_recursive({"_preset_": "audit_ridge_grid", "model": "sklearn.linear_model.Ridge"})
    assert resolved["model"] == "sklearn.linear_model.Ridge"
    assert "_grid_" in resolved


def test_range_node_seed_is_reproducible():
    spec = {"_range_": [1, 1000], "count": 5, "_seed_": 42}
    assert expand_spec(spec) == expand_spec(spec)


@pytest.mark.parametrize("values", [[0.0, 0.6, 0.1], [0.0, 0.3, 0.1], [0.6, 0.0, -0.1]])
def test_float_range_count_matches_expansion(values):
    spec = {"_range_": values}
    assert count_combinations(spec) == len(expand_spec(spec))


@pytest.mark.parametrize("spec", [{"_log_range_": [1e-12, 1e-6, 4]},
                                  {"_sample_": {"distribution": "log_uniform", "from": 1e-14, "to": 1e-12, "num": 20}}])
def test_small_positive_sweep_values_remain_positive(spec):
    assert all(value > 0 for value in expand_spec(spec, seed=42))
