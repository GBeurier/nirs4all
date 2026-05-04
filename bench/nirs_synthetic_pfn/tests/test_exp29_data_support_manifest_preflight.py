"""Tests for the P2-08 data-support manifest preflight."""

from __future__ import annotations

import importlib.util
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType


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


def _load_exp29() -> ModuleType:
    return _load_module(
        "exp29_data_support_manifest_preflight",
        "exp29_data_support_manifest_preflight.py",
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_blocked_empty_current_state(tmp_path: Path) -> None:
    exp29 = _load_exp29()

    result = exp29.run_preflight(tmp_path)
    markdown = exp29.render_markdown(
        result,
        report_path=tmp_path / "exp29.md",
        csv_path=None,
        manifest=None,
    )

    assert result["recommendation"] == exp29.BLOCKED_RECOMMENDATION
    assert result["rows"] == []
    assert "no stats" in markdown
    assert "no calibration" in markdown
    assert "no generator profile" in markdown
    assert "nirs4all/` is not modified" in markdown


def test_wider_support_manifest_row_unblocks(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        "source,task,database_name,dataset,wavelengths\n"
        "future_manifest,regression,DIESEL,DIESEL_wide,700;750;900;1550;1600\n",
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.READY_RECOMMENDATION
    assert len(result["wider_real_support_rows"]) == 1
    assert row.extends_outside_750_1550_after_alignment is True
    assert row.off_support_count_after_alignment == 2
    assert row.recommendation_signal == "wider_real_support_available"


def test_row_bound_geometry_manifest_row_unblocks(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        """
        {
          "rows": [
            {
              "source": "future_manifest",
              "task": "regression",
              "database_name": "DIESEL",
              "dataset": "DIESEL_geometry",
              "wavelengths": "750;900;1550",
              "row_binding_key": "sample_id",
              "metadata_source": "Mtrain.csv",
              "geometry_metadata_kind": "real_cohort_metadata_header",
              "source_detector_distance_mm": "4.0",
              "collection_angle_deg": "45"
            }
          ]
        }
        """,
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.READY_RECOMMENDATION
    assert len(result["row_bound_geometry_rows"]) == 1
    assert row.extends_outside_750_1550_after_alignment is False
    assert row.row_bound_geometry_metadata_present is True
    assert row.binding_fields_missing == ""
    assert "source_detector_distance_mm" in row.parsed_geometry_fields


def test_geometry_with_binding_key_but_without_real_row_bound_enum_does_not_unblock(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        """
        {
          "rows": [
            {
              "source": "future_manifest",
              "task": "regression",
              "database_name": "DIESEL",
              "dataset": "DIESEL_geometry",
              "row_binding_key": "sample_id",
              "metadata_source": "Mtrain.csv",
              "source_detector_distance_mm": "4.0"
            }
          ]
        }
        """,
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.BLOCKED_RECOMMENDATION
    assert row.status == "blocked_no_wavelength_or_row_bound_geometry_support"
    assert row.row_bound_geometry_metadata_present is False
    assert row.parsed_geometry_fields == "source_detector_distance_mm"


def test_generic_geometry_without_row_binding_does_not_unblock(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        "source,task,database_name,dataset,wavelengths,geometry_metadata_kind,source_detector_distance_mm\n"
        "future_manifest,regression,DIESEL,DIESEL_generic,750;900;1550,generic,4.0\n",
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.BLOCKED_RECOMMENDATION
    assert len(result["generic_geometry_rows"]) == 1
    assert row.generic_geometry_present is True
    assert row.row_bound_geometry_metadata_present is False
    assert row.recommendation_signal == "generic_geometry_available_not_row_bound"


def test_label_target_split_fields_are_rejected_and_do_not_unblock(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        "source,task,database_name,dataset,wavelengths,label,target,split,adversarial_score,auc,transfer_score\n"
        "future_manifest,regression,DIESEL,DIESEL_leaky,700;750;1550;1600,A,1.0,train,0.9,0.8,0.7\n",
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.BLOCKED_RECOMMENDATION
    assert row.status == "rejected_leakage_fields"
    assert row.extends_outside_750_1550_after_alignment is False
    assert row.off_support_count_after_alignment == 0
    assert row.rejected_leakage_fields == "adversarial_score;auc;label;split;target;transfer_score"
    assert row.recommendation_signal == "rejected_no_label_target_split_downstream_adversarial_auc_or_metric_inputs"


def test_non_fuel_wider_support_manifest_does_not_unblock(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        "source,task,database_name,dataset,wavelengths\n"
        "future_manifest,regression,WHEAT,WHEAT_wide,700;750;900;1550;1600\n",
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.BLOCKED_RECOMMENDATION
    assert row.status == "blocked_not_diesel_fuel_manifest_data_support"
    assert row.extends_outside_750_1550_after_alignment is False
    assert row.off_support_count_after_alignment == 2
    assert row.recommendation_signal == "not_diesel_fuel_manifest_data_support"


def test_summary_min_max_count_fields_parse_consistently_for_csv_and_json(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    csv_manifest = tmp_path / "manifest.csv"
    json_manifest = tmp_path / "manifest.json"
    _write(
        csv_manifest,
        "source,task,database_name,dataset,wavelength_min,wavelength_max,n_wavelengths_after_alignment,off_support_count_after_alignment\n"
        "future_manifest,regression,DIESEL,DIESEL_summary,700,1600,5,2\n",
    )
    _write(
        json_manifest,
        """
        {
          "rows": [
            {
              "source": "future_manifest",
              "task": "regression",
              "database_name": "DIESEL",
              "dataset": "DIESEL_summary",
              "wavelength_min": 700,
              "wavelength_max": 1600,
              "n_wavelengths_after_alignment": 5,
              "off_support_count_after_alignment": 2
            }
          ]
        }
        """,
    )

    csv_result = exp29.run_preflight(tmp_path, csv_manifest)
    json_result = exp29.run_preflight(tmp_path, json_manifest)
    csv_row = csv_result["rows"][0]
    json_row = json_result["rows"][0]

    assert csv_result["recommendation"] == exp29.READY_RECOMMENDATION
    assert json_result["recommendation"] == exp29.READY_RECOMMENDATION
    assert csv_row.wavelength_min == json_row.wavelength_min == 700.0
    assert csv_row.wavelength_max == json_row.wavelength_max == 1600.0
    assert csv_row.n_wavelengths_after_alignment == json_row.n_wavelengths_after_alignment == 5
    assert csv_row.support_count_after_alignment == json_row.support_count_after_alignment == 3
    assert csv_row.off_support_count_after_alignment == json_row.off_support_count_after_alignment == 2
    assert csv_row.evidence == json_row.evidence == "manifest_wavelength_min_max_count_fields"


def test_summary_min_max_count_without_off_support_count_is_ambiguous_and_blocked(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        "source,task,database_name,dataset,wavelength_min,wavelength_max,n_wavelengths_after_alignment\n"
        "future_manifest,regression,DIESEL,DIESEL_ambiguous,700,1600,5\n",
    )

    result = exp29.run_preflight(tmp_path, manifest)
    row = result["rows"][0]

    assert result["recommendation"] == exp29.BLOCKED_RECOMMENDATION
    assert row.status == "blocked_no_wavelength_or_row_bound_geometry_support"
    assert row.evidence == "no_wavelength_source"


def test_header_alignment_csv_output_and_markdown_contract(tmp_path: Path) -> None:
    exp29 = _load_exp29()
    root = tmp_path
    _write(root / "wide/Xtrain.csv", "750;900;1550\n1;2;3\n")
    _write(root / "wide/Xtest.csv", "700;900;1600\n1;2;3\n")
    manifest = root / "manifest.csv"
    _write(
        manifest,
        "source,task,database_name,dataset,train_path,test_path\n"
        "future_manifest,regression,DIESEL,DIESEL_merge,wide/Xtrain.csv,wide/Xtest.csv\n",
    )
    csv_path = tmp_path / "exp29.csv"
    report_path = tmp_path / "exp29.md"

    result = exp29.run_preflight(root, manifest)
    exp29.write_csv(result["rows"], csv_path)
    markdown = exp29.render_markdown(
        result,
        report_path=report_path,
        csv_path=csv_path,
        manifest=manifest,
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(DictReader(handle))

    assert result["recommendation"] == exp29.READY_RECOMMENDATION
    assert records[0]["n_wavelengths_after_alignment"] == "5"
    assert records[0]["spectral_path_fields"] == "train_path;test_path"
    assert "ready_for_mechanistic_audit_design" in markdown
    assert "No readiness prerequisite is satisfied" not in markdown
