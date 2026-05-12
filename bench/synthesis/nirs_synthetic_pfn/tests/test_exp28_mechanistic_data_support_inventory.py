"""Tests for the P2-06 mechanistic data-support inventory."""

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


def _load_exp28() -> ModuleType:
    return _load_module(
        "exp28_mechanistic_data_support_inventory",
        "exp28_mechanistic_data_support_inventory.py",
    )


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_numeric_header_and_support_counts(tmp_path: Path) -> None:
    exp28 = _load_exp28()
    path = tmp_path / "Xtrain.csv"
    path.write_text("700;750;900;1550;1600\n1;2;3;4;5\n", encoding="utf-8")

    wavelengths = exp28._parse_numeric_header(path)
    support, off_support, extends = exp28._support_counts(wavelengths)

    assert wavelengths == [700.0, 750.0, 900.0, 1550.0, 1600.0]
    assert support == 3
    assert off_support == 2
    assert extends is True


def test_mismatched_train_test_headers_are_merged_for_support_inventory() -> None:
    exp28 = _load_exp28()

    wavelengths = exp28._merged_wavelength_headers([750.0, 900.0], [900.0, 1600.0])
    support, off_support, extends = exp28._support_counts(wavelengths)

    assert wavelengths == [750.0, 900.0, 1600.0]
    assert support == 2
    assert off_support == 1
    assert extends is True


def test_text_inventory_skips_hidden_cache_directories(tmp_path: Path) -> None:
    exp28 = _load_exp28()
    root = tmp_path
    _write_csv(
        root / "bench/nirs_synthetic_pfn/.mypy_cache/stale.json",
        '{"source_detector": "cache artifact"}\n',
    )
    _write_csv(
        root / "bench/nirs_synthetic_pfn/docs/geometry.md",
        "source_detector distance is a real doc term\n",
    )

    files = exp28._iter_text_files(root, ("bench/nirs_synthetic_pfn",))

    assert files == [root / "bench/nirs_synthetic_pfn/docs/geometry.md"]


def test_run_inventory_finds_wider_fuel_grid_and_generic_geometry(tmp_path: Path) -> None:
    exp28 = _load_exp28()
    root = tmp_path
    data_dir = root / "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_wide"
    _write_csv(data_dir / "Xtrain.csv", "700;750;900;1550;1600\n1;2;3;4;5\n")
    _write_csv(data_dir / "Xtest.csv", "700;750;900;1550;1600\n1;2;3;4;5\n")
    _write_csv(data_dir / "Ytrain.csv", "Y\n1\n")
    _write_csv(data_dir / "Ytest.csv", "Y\n1\n")
    _write_csv(
        root / "bench/AOM_v0/benchmarks/cohort_regression.csv",
        "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path\n"
        "DIESEL,DIESEL_wide,ok,,1,1,5,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_wide/Xtrain.csv,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_wide/Xtest.csv,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_wide/Ytrain.csv,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_wide/Ytest.csv\n",
    )
    _write_csv(
        root / "bench/AOM_v0/benchmarks/cohort_classification.csv",
        "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path\n",
    )
    _write_csv(
        root / "nirs4all/synthesis/measurement_modes.py",
        "path_length_mm = 1.0\nillumination_angle = 0.0\ncollection_angle = 45.0\nBeer-Lambert\n",
    )

    result = exp28.run_inventory(root)

    assert result["recommendation"] == "continue_mechanistic_with_data_supported_audit"
    assert len(result["wider_real_grid_rows"]) == 2
    assert len(result["generic_geometry_rows"]) == 1
    assert any(row.law_area == "measurement_mode_pathlength_scattering" for row in result["law_rows"])


def test_csv_and_markdown_contract_for_blocked_current_state(tmp_path: Path) -> None:
    exp28 = _load_exp28()
    root = tmp_path
    data_dir = root / "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_current"
    _write_csv(data_dir / "Xtrain.csv", "750;900;1550\n1;2;3\n")
    _write_csv(data_dir / "Xtest.csv", "750;900;1550\n1;2;3\n")
    _write_csv(data_dir / "Ytrain.csv", "Y\n1\n")
    _write_csv(data_dir / "Ytest.csv", "Y\n1\n")
    _write_csv(
        root / "bench/AOM_v0/benchmarks/cohort_regression.csv",
        "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path\n"
        "DIESEL,DIESEL_current,ok,,1,1,3,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_current/Xtrain.csv,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_current/Xtest.csv,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_current/Ytrain.csv,"
        "bench/tabpfn_paper/data/regression/DIESEL/DIESEL_current/Ytest.csv\n",
    )
    _write_csv(
        root / "bench/AOM_v0/benchmarks/cohort_classification.csv",
        "database_name,dataset,status,reason,n_train,n_test,p,train_path,test_path,ytrain_path,ytest_path\n",
    )
    _write_csv(
        root / "nirs4all/synthesis/domains.py",
        "petrochem_fuels = {'wavelength_range': (900, 1700), 'measurement_mode': 'transmission'}\n",
    )
    result = exp28.run_inventory(root)
    csv_path = tmp_path / "exp28.csv"
    report_path = tmp_path / "exp28.md"

    exp28.write_csv(result["rows"], csv_path)
    markdown = exp28.render_markdown(result, report_path=report_path, csv_path=csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(DictReader(handle))

    assert records
    assert result["recommendation"] == "blocked_pending_metadata_or_wider_real_cohort_no_stats_ml"
    assert "Rows extending outside `750-1550 nm` after real-grid alignment semantics: `0`" in markdown
    assert "Do not move to stats/ML yet" in markdown
    assert "No calibration" in markdown
    assert "generator profile" in markdown
