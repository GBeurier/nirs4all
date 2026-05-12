"""Tests for the exp31 multidataset evidence-gap manifest preflight."""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_exp31() -> ModuleType:
    return _load_module(
        "exp31_multidataset_evidence_gap_manifest_preflight",
        "exp31_multidataset_evidence_gap_manifest_preflight.py",
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _csv_from_records(records: list[dict[str, Any]]) -> str:
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for record in records:
        writer.writerow({key: ("" if record.get(key) is None else str(record.get(key))) for key in keys})
    return buffer.getvalue()


def _plant_leaf_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "regime_family": "plant_leaf_visnir_swir",
        "source": "future_lab",
        "task": "regression",
        "database_name": "ECOSIS_LeafTraits",
        "dataset": "Chla+b_documented_2026_leaf",
        "axis_unit": "nm",
        "axis_min_value": "450",
        "axis_max_value": "2400",
        "n_features_after_alignment": "196",
        "n_train_rows": "3734",
        "n_test_rows": "3116",
        "preprocessing_status_documented_source": "doi:10.1234/ecosis-leaf",
        "preprocessing_status_value": "reflectance",
        "instrument_class": "LeafIntegratingSphere",
    }
    base.update(overrides)
    return base


def _manure_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "regime_family": "manure_organic_mineral",
        "source": "future_lab",
        "task": "regression",
        "database_name": "MANURE21",
        "dataset": "All_manure_documented_2026",
        "axis_unit": "nm",
        "axis_min_value": "852.78",
        "axis_max_value": "2502.37",
        "n_features_after_alignment": "1003",
        "n_train_rows": "343",
        "n_test_rows": "147",
        "preprocessing_status_documented_source": "doi:10.1234/manure-acquisition",
        "preprocessing_status_value": "reflectance",
        "acquisition_geometry_documented_source": "doi:10.1234/manure-geometry",
        "acquisition_geometry_kind": "row_bound_real_metadata",
        "cup_diameter_mm": "30",
        "instrument_class": "FOSS_NIRSystems_5000",
    }
    base.update(overrides)
    return base


def _liquid_food_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "regime_family": "liquid_food",
        "source": "future_lab",
        "task": "regression",
        "database_name": "BEER",
        "dataset": "Beer_OriginalExtract_documented_2026",
        "axis_unit": "nm",
        "axis_min_value": "1100",
        "axis_max_value": "2250",
        "n_features_after_alignment": "576",
        "n_train_rows": "40",
        "n_test_rows": "20",
        "preprocessing_status_documented_source": "doi:10.1234/beer-acquisition",
        "preprocessing_status_value": "absorbance",
        "pathlength_documented_source": "doi:10.1234/beer-acquisition",
        "pathlength_mm": "5.0",
        "temperature_documented_source": "doi:10.1234/beer-acquisition",
        "temperature_field_value_or_range": "20+/-1C",
        "batch_documented_source": "doi:10.1234/beer-acquisition",
        "batch_field_or_descriptor": "fermentation_lot_id",
    }
    base.update(overrides)
    return base


def _mineral_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "regime_family": "mineral_incombustible",
        "source": "future_lab",
        "task": "regression",
        "database_name": "IncombustibleMaterial",
        "dataset": "TIC_documented_2026_mineral",
        "axis_unit": "nm",
        "axis_min_value": "868",
        "axis_max_value": "1764",
        "n_features_after_alignment": "254",
        "n_train_rows": "43",
        "n_test_rows": "19",
        "preprocessing_status_documented_source": "doi:10.1234/mineral-acquisition",
        "preprocessing_status_value": "reflectance",
        "acquisition_geometry_documented_source": "doi:10.1234/mineral-geometry",
        "acquisition_geometry_kind": "real_cohort_metadata_header",
        "instrument_class": "FieldSpec",
    }
    base.update(overrides)
    return base


def _wavenumber_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "regime_family": "wavenumber_domain",
        "source": "future_lab",
        "task": "regression",
        "database_name": "COLZA",
        "dataset": "N_woOutlier_documented_2026",
        "axis_unit": "cm-1",
        "axis_min_value": "3594.9",
        "axis_max_value": "12489.6",
        "n_features_after_alignment": "1154",
        "n_train_rows": "1205",
        "n_test_rows": "1207",
        "preprocessing_status_documented_source": "doi:10.1234/colza-readme",
        "preprocessing_status_value": "absorbance",
        "axis_unit_documented_source": "doi:10.1234/colza-readme",
        "axis_direction_documented_source": "doi:10.1234/colza-readme",
        "axis_conversion_contract_source": "doi:10.1234/colza-axis-conversion",
        "panel_breadth_documented_sources": "doi:10.1234/colza-readme, doi:10.1234/second-cohort",
    }
    base.update(overrides)
    return base


# -----------------------------
# Test 1: empty default
# -----------------------------


def test_empty_default_emits_five_blocked_pending_decisions(tmp_path: Path) -> None:
    exp31 = _load_exp31()

    result = exp31.run_preflight(tmp_path)
    decisions = result["per_family_decisions"]

    assert result["rows"] == []
    for family in exp31.REGIME_FAMILIES:
        assert decisions[family] == f"blocked_pending_{family}_evidence_no_stats_ml"


# -----------------------------
# Test 2: leakage rejection
# -----------------------------


def test_leakage_fields_force_rejection_regardless_of_completeness(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record = _plant_leaf_row(label="A", auc="0.9")
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([record]))

    result = exp31.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert row.status == "rejected_leakage_fields"
    assert "auc" in row.rejected_leakage_fields
    assert "label" in row.rejected_leakage_fields
    assert result["per_family_decisions"]["plant_leaf_visnir_swir"] == "blocked_pending_plant_leaf_visnir_swir_evidence_no_stats_ml"


# -----------------------------
# Test 3: positive ready per family
# -----------------------------


def test_each_family_has_positive_ready_decision(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    records = [
        _plant_leaf_row(),
        _manure_row(),
        _liquid_food_row(),
        _mineral_row(),
        _wavenumber_row(),
    ]
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records(records))

    result = exp31.run_preflight(tmp_path, manifest)
    decisions = result["per_family_decisions"]

    for family in exp31.REGIME_FAMILIES:
        assert decisions[family] == f"ready_for_phase_m3_mechanism_design_{family}", (
            family,
            decisions[family],
            [(row.row_index, row.status, row.recommendation_signal, row.missing_evidence_fields, row.cross_family_tokens) for row in result["rows"]],
        )
    statuses = {row.status for row in result["rows"]}
    assert statuses == {f"accepted_for_{family}" for family in exp31.REGIME_FAMILIES}


# -----------------------------
# Test 4: negative blocked per family
# -----------------------------


def test_each_family_has_negative_blocked_when_required_field_dropped(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    builder_for_family = {
        "plant_leaf_visnir_swir": (_plant_leaf_row, "instrument_class"),
        "manure_organic_mineral": (_manure_row, "acquisition_geometry_documented_source"),
        "liquid_food": (_liquid_food_row, "pathlength_mm"),
        "mineral_incombustible": (_mineral_row, "instrument_class"),
        "wavenumber_domain": (_wavenumber_row, "axis_conversion_contract_source"),
    }
    for family, (builder, drop_field) in builder_for_family.items():
        record = builder(**{drop_field: ""})
        manifest = tmp_path / f"manifest_{family}.csv"
        _write(manifest, _csv_from_records([record]))

        result = exp31.run_preflight(tmp_path, manifest)
        row = result["rows"][0]

        assert row.status == f"blocked_missing_{family}_evidence_no_stats_ml", (family, row.status, row.missing_evidence_fields)
        assert drop_field in row.missing_evidence_fields
        assert result["per_family_decisions"][family] == f"blocked_pending_{family}_evidence_no_stats_ml"


# -----------------------------
# Test 5: cross-family token leakage
# -----------------------------


def test_cross_family_identity_token_blocks(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record = _liquid_food_row(dataset="Beer_OriginalExtract_with_manure_test")
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([record]))

    result = exp31.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert row.status == "blocked_cross_family_identity_token"
    assert "manure_organic_mineral:manure" in row.cross_family_tokens
    assert result["per_family_decisions"]["liquid_food"] == "blocked_pending_liquid_food_evidence_no_stats_ml"


# -----------------------------
# Test 6: reserved diesel tokens
# -----------------------------


def test_reserved_diesel_fuel_token_blocks_in_non_diesel_family(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record = _liquid_food_row(dataset="Beer_with_diesel_blend_documented_2026")
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([record]))

    result = exp31.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert row.status == "blocked_reserved_diesel_fuel_token"
    assert "diesel" in row.reserved_diesel_tokens
    assert result["per_family_decisions"]["liquid_food"] == "blocked_pending_liquid_food_evidence_no_stats_ml"


# -----------------------------
# Test 7: invalid enum values
# -----------------------------


def test_invalid_enum_values_blocked(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record_pp = _plant_leaf_row(preprocessing_status_value="not_in_enum")
    record_geom = _manure_row(acquisition_geometry_kind="not_in_enum")
    record_axis = _wavenumber_row(axis_unit="not_in_enum")
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([record_pp, record_geom, record_axis]))

    result = exp31.run_preflight(tmp_path, manifest)
    rows = result["rows"]

    assert all(row.status == "blocked_invalid_enum_value" for row in rows), [row.status for row in rows]
    assert any("preprocessing_status_value=not_in_enum" in row.enum_failures for row in rows)
    assert any("acquisition_geometry_kind=not_in_enum" in row.enum_failures for row in rows)
    assert any("axis_unit=not_in_enum" in row.enum_failures for row in rows)


# -----------------------------
# Test 8: invalid numeric fields
# -----------------------------


def test_invalid_numeric_fields_blocked(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record_swap = _plant_leaf_row(axis_min_value="2400", axis_max_value="450")
    record_pathlength = _liquid_food_row(pathlength_mm="not_a_number")
    record_axis_text = _plant_leaf_row(axis_min_value="abc", axis_max_value="def", dataset="alpine_documented_2026")
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([record_swap, record_pathlength, record_axis_text]))

    result = exp31.run_preflight(tmp_path, manifest)
    rows = result["rows"]

    assert all(row.status == "blocked_invalid_numeric_field" for row in rows), [(row.row_index, row.status, row.missing_evidence_fields, row.numeric_failures) for row in rows]
    assert any("axis_min_value_not_less_than_axis_max_value" in row.numeric_failures for row in rows)
    assert any("pathlength_mm_not_numeric" in row.numeric_failures for row in rows)
    assert any("axis_min_value_not_numeric" in row.numeric_failures for row in rows)


# -----------------------------
# Test 9: documented source not found
# -----------------------------


def test_documented_source_path_not_resolvable_blocks(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record = _plant_leaf_row(preprocessing_status_documented_source="docs/missing_source.txt")
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([record]))

    result = exp31.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert row.status == "blocked_documented_source_not_found"
    assert "preprocessing_status_documented_source=docs/missing_source.txt" in row.unresolved_documented_sources

    # Now create the file and confirm acceptance
    _write(tmp_path / "docs/missing_source.txt", "documented")
    result_ok = exp31.run_preflight(tmp_path, manifest)
    row_ok = result_ok["rows"][0]
    assert row_ok.status == "accepted_for_plant_leaf_visnir_swir"


# -----------------------------
# Test 10: CSV vs JSON parity
# -----------------------------


def test_csv_and_json_manifests_produce_identical_decisions(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    record = _liquid_food_row()
    csv_manifest = tmp_path / "manifest.csv"
    json_manifest = tmp_path / "manifest.json"
    _write(csv_manifest, _csv_from_records([record]))
    import json as _json
    _write(json_manifest, _json.dumps({"rows": [record]}))

    csv_result = exp31.run_preflight(tmp_path, csv_manifest)
    json_result = exp31.run_preflight(tmp_path, json_manifest)

    assert csv_result["per_family_decisions"] == json_result["per_family_decisions"]
    assert csv_result["rows"][0].status == json_result["rows"][0].status
    assert csv_result["rows"][0].recommendation_signal == json_result["rows"][0].recommendation_signal
    assert csv_result["manifest_source_kind"] == "csv"
    assert json_result["manifest_source_kind"] == "json"


# -----------------------------
# Test 11: --regime-family filter
# -----------------------------


def test_regime_family_filter_restricts_aggregation(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    records = [_plant_leaf_row(), _manure_row()]
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records(records))

    result = exp31.run_preflight(tmp_path, manifest, regime_family_filter=["plant_leaf_visnir_swir"])
    decisions = result["per_family_decisions"]
    statuses = {row.status for row in result["rows"]}

    assert decisions["plant_leaf_visnir_swir"] == "ready_for_phase_m3_mechanism_design_plant_leaf_visnir_swir"
    for family in exp31.REGIME_FAMILIES:
        if family == "plant_leaf_visnir_swir":
            continue
        assert decisions[family] == f"blocked_pending_{family}_evidence_no_stats_ml"
    assert "filtered_out" in statuses


# -----------------------------
# Test 12: markdown contract strings
# -----------------------------


def test_markdown_contract_contains_required_strings(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records([_plant_leaf_row()]))
    result = exp31.run_preflight(tmp_path, manifest)
    markdown = exp31.render_markdown(
        result,
        report_path=tmp_path / "report.md",
        csv_path=tmp_path / "report.csv",
        manifest=manifest,
    )

    for required in ("no statistics", "no PCA", "no calibration", "no ML", "no DL", "no labels", "no targets", "no splits"):
        assert required in markdown, required
    assert "nirs4all/` is not modified" in markdown


# -----------------------------
# Test 13: CSV columns and row count
# -----------------------------


def test_csv_columns_and_row_count_match_manifest(tmp_path: Path) -> None:
    exp31 = _load_exp31()
    records = [_plant_leaf_row(), _manure_row(), _liquid_food_row()]
    manifest = tmp_path / "manifest.csv"
    _write(manifest, _csv_from_records(records))
    csv_path = tmp_path / "exp31.csv"

    result = exp31.run_preflight(tmp_path, manifest)
    exp31.write_csv(result["rows"], csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(DictReader(handle))

    assert len(rows) == len(records)
    mandatory_columns = {
        "status",
        "regime_family",
        "source",
        "task",
        "database_name",
        "dataset",
        "axis_unit",
        "axis_min_value",
        "axis_max_value",
        "n_features_after_alignment",
        "n_train_rows",
        "n_test_rows",
        "preprocessing_status_value",
        "missing_evidence_fields",
        "rejected_leakage_fields",
        "recommendation_signal",
    }
    assert mandatory_columns.issubset(set(rows[0].keys()))
