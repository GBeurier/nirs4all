"""Tests for the exp30 multidataset real spectral atlas (Phase M0/M1)."""

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


def _load_exp30() -> ModuleType:
    return _load_module(
        "exp30_multidataset_real_spectral_atlas",
        "exp30_multidataset_real_spectral_atlas.py",
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_dataset(
    root: Path,
    *,
    relative_path: str,
    x_header: str,
    x_rows: list[str],
    y_header: str,
    y_rows: list[str],
    metadata_header: str | None = None,
    metadata_rows: list[str] | None = None,
) -> Path:
    directory = root / relative_path
    _write(directory / "Xtrain.csv", x_header + "\n" + "\n".join(x_rows) + "\n")
    _write(directory / "Xtest.csv", x_header + "\n" + x_rows[0] + "\n")
    _write(directory / "Ytrain.csv", y_header + "\n" + "\n".join(y_rows) + "\n")
    _write(directory / "Ytest.csv", y_header + "\n" + y_rows[0] + "\n")
    if metadata_header is not None and metadata_rows is not None:
        _write(directory / "Mtrain.csv", metadata_header + "\n" + "\n".join(metadata_rows) + "\n")
        _write(directory / "Mtest.csv", metadata_header + "\n" + metadata_rows[0] + "\n")
    return directory


def test_axis_parser_handles_nm_suffix_semicolon_headers() -> None:
    exp30 = _load_exp30()
    values, separator, token_format = exp30._parse_axis("852.78_nm;853.34_nm;854.0_nm")
    assert separator == ";"
    assert token_format == "nm_suffix"
    assert values == [852.78, 853.34, 854.0]


def test_axis_parser_handles_quoted_int_comma_headers() -> None:
    exp30 = _load_exp30()
    values, separator, token_format = exp30._parse_axis('"1100";"1102";"1104"')
    assert separator == ";"
    assert token_format == "int_quoted"
    assert values == [1100.0, 1102.0, 1104.0]


def test_axis_unit_forces_wavenumber_for_n_woOutlier() -> None:
    exp30 = _load_exp30()
    axis_type, axis_unit, reason = exp30._axis_unit_and_type([12489.6, 12481.89], "numeric", "N_woOutlier")
    assert axis_type == "wavenumber"
    assert axis_unit == "cm-1"
    assert "forced_by_panel_rule" in reason


def test_axis_unit_treats_above_4000_as_wavenumber() -> None:
    exp30 = _load_exp30()
    axis_type, axis_unit, reason = exp30._axis_unit_and_type([4500.0, 4400.0], "numeric", "Anything")
    assert axis_type == "wavenumber"
    assert axis_unit == "cm-1"
    assert "above_4000" in reason


def test_axis_unit_treats_visnir_swir_range_as_nm_when_no_explicit_suffix() -> None:
    exp30 = _load_exp30()
    axis_type, axis_unit, reason = exp30._axis_unit_and_type([350.0, 2500.0], "numeric", "Anything")
    assert axis_type == "nm"
    assert axis_unit == "nm"
    assert "VIS_NIR_SWIR" in reason


def test_axis_direction_detects_descending() -> None:
    exp30 = _load_exp30()
    assert exp30._axis_direction([12489.6, 11000.0, 3594.9]) == "descending"
    assert exp30._axis_direction([350.0, 2500.0]) == "ascending"
    assert exp30._axis_direction([100.0]) == "unknown"


def test_split_policy_detects_multiple_tokens() -> None:
    exp30 = _load_exp30()
    assert exp30._split_policy("All_manure_MgO_SPXY_strat_Manure_type") == "SPXY;stratified"
    assert exp30._split_policy("ALPINE_P_291_KS") == "KennardStone"
    assert exp30._split_policy("Beer_OriginalExtract_60_YbaseSplit") == "YbaseSplit"
    assert exp30._split_policy("An_spxyG70_30_byCultivar_NeoSpectra") == "SPXY;byCultivar"
    assert exp30._split_policy("nothing_obvious") == ""


def test_atlas_row_built_from_synthetic_dataset_with_metadata(tmp_path: Path) -> None:
    exp30 = _load_exp30()
    _make_dataset(
        tmp_path,
        relative_path="FAKE/dataset_a",
        x_header="500;510;520",
        x_rows=["0.1;0.2;0.3", "0.15;0.25;0.35"],
        y_header="target",
        y_rows=["1.0", "2.0"],
        metadata_header="Sample_ID;site;temperature_c",
        metadata_rows=["S1;A;20", "S2;B;21"],
    )
    panel = (("FAKE", "dataset_a", "FAKE/dataset_a"),)

    rows = exp30.build_atlas(tmp_path, panel)
    row = rows[0]

    assert row.status == "ok"
    assert row.train_rows == 2
    assert row.test_rows == 1
    assert row.n_features == 3
    assert row.axis_type == "nm"
    assert row.axis_separator == ";"
    assert row.axis_first == 500.0
    assert row.axis_last == 520.0
    assert row.axis_resolution_median == 10.0
    assert row.has_metadata is True
    assert row.metadata_columns_count == 3
    assert row.metadata_columns == "Sample_ID;site;temperature_c"
    assert row.target_column == "target"
    assert row.preprocessing_evidence == "unknown"
    assert row.documentary_evidence_source == ""


def test_atlas_row_detects_negative_values_and_target_sentinels(tmp_path: Path) -> None:
    exp30 = _load_exp30()
    _make_dataset(
        tmp_path,
        relative_path="FAKE/dataset_b",
        x_header="500;600;700",
        x_rows=["0.1;-0.05;0.3"],
        y_header="target",
        y_rows=["-999"],
    )
    panel = (("FAKE", "dataset_b", "FAKE/dataset_b"),)

    rows = exp30.build_atlas(tmp_path, panel)
    row = rows[0]

    assert row.status == "ok"
    assert row.has_negative_values is True
    assert row.sentinel_train_rows == 1
    assert "negative_x_values_observed_in_sampled_rows" in row.notes
    assert "target_sentinel_rows_present" in row.notes


def test_atlas_row_reports_missing_directory(tmp_path: Path) -> None:
    exp30 = _load_exp30()
    panel = (("FAKE", "missing", "FAKE/missing"),)

    rows = exp30.build_atlas(tmp_path, panel)
    row = rows[0]

    assert row.status == "directory_not_found"
    assert row.train_rows == 0
    assert row.preprocessing_evidence == "unknown"


def test_atlas_row_reports_missing_required_files(tmp_path: Path) -> None:
    exp30 = _load_exp30()
    directory = tmp_path / "FAKE/dataset_partial"
    directory.mkdir(parents=True)
    (directory / "Xtrain.csv").write_text("500;600\n0.1;0.2\n", encoding="utf-8")
    panel = (("FAKE", "dataset_partial", "FAKE/dataset_partial"),)

    rows = exp30.build_atlas(tmp_path, panel)
    row = rows[0]

    assert row.status == "required_files_missing"
    assert "Xtrain" in row.files_present
    assert "Xtest" in row.files_missing


def test_csv_and_markdown_output_contract(tmp_path: Path) -> None:
    exp30 = _load_exp30()
    _make_dataset(
        tmp_path,
        relative_path="FAKE/dataset_a",
        x_header="500;600;700",
        x_rows=["0.1;0.2;0.3"],
        y_header="target",
        y_rows=["1.0"],
    )
    _make_dataset(
        tmp_path,
        relative_path="FAKE/dataset_w",
        x_header="12489.6;12481.89;12474.17",
        x_rows=["0.2;0.21;0.22"],
        y_header="y",
        y_rows=["3.5"],
    )
    panel = (
        ("FAKE", "dataset_a", "FAKE/dataset_a"),
        ("FAKE", "dataset_w", "FAKE/dataset_w"),
    )
    csv_path = tmp_path / "atlas.csv"
    report_path = tmp_path / "atlas.md"

    rows = exp30.build_atlas(tmp_path, panel)
    exp30.write_csv(rows, csv_path)
    markdown = exp30.render_markdown(
        rows,
        panel_root=tmp_path,
        report_path=report_path,
        csv_path=csv_path,
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(DictReader(handle))

    assert len(records) == 2
    assert {row["axis_type"] for row in records} == {"nm", "wavenumber"}
    assert "Multidataset Real Spectral Atlas" in markdown
    assert "no labels-as-oracle" in markdown
    assert "no targets-as-oracle" in markdown
    assert "no splits-as-oracle" in markdown
    assert "nirs4all/` is not modified" in markdown
    assert "Phase M1 Distinguishability Checklist" in markdown


def test_real_panel_atlas_runs_against_repo() -> None:
    exp30 = _load_exp30()
    repo_root = Path(__file__).resolve().parents[3]
    panel_root = repo_root / "bench/tabpfn_paper/data/regression"
    if not panel_root.exists():
        return
    rows = exp30.build_atlas(panel_root)
    assert len(rows) == len(exp30.PANEL)
    ok_rows = [row for row in rows if row.status == "ok"]
    assert ok_rows, "expected at least one inspectable panel dataset"
    n_woOutlier = next((row for row in rows if row.dataset == "N_woOutlier"), None)
    if n_woOutlier is not None and n_woOutlier.status == "ok":
        assert n_woOutlier.axis_type == "wavenumber"
        assert n_woOutlier.axis_unit == "cm-1"
        assert n_woOutlier.axis_direction == "descending"
        assert n_woOutlier.preprocessing_evidence == "absorbance"
        assert n_woOutlier.documentary_evidence_source.endswith("README.txt")
    species_row = next(
        (row for row in rows if row.dataset == "Chla+b_spxyG_species"),
        None,
    )
    if species_row is not None and species_row.status == "ok":
        assert species_row.sentinel_train_rows >= 1
