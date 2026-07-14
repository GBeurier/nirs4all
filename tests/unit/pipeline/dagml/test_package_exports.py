"""Locks for lightweight dagml package exports."""

from __future__ import annotations

import nirs4all.pipeline.dagml as dagml_pkg


def test_dagml_package_lists_internal_native_tuning_modules_without_import_side_effects() -> None:
    assert "conformal_contracts" in dagml_pkg.__all__
    assert "conformal_store" in dagml_pkg.__all__
    assert "tuning_contracts" in dagml_pkg.__all__
    assert "tuning_adapters" in dagml_pkg.__all__
    assert "tuning_projection" in dagml_pkg.__all__
    assert "pipeline_objective" in dagml_pkg.__all__
    assert "pipeline_objective_compiler" in dagml_pkg.__all__
