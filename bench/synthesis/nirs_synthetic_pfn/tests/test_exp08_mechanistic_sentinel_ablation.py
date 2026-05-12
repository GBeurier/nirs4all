"""Schema and report tests for the R2a sentinel mechanistic ablation."""

from __future__ import annotations

import importlib.util
import sys
from csv import DictReader
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from nirsyntheticpfn.adapters.builder_adapter import R2A_MECHANISTIC_PROFILES


def _load_exp08_module() -> ModuleType:
    name = "exp08_mechanistic_sentinel_ablation"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "experiments/exp08_mechanistic_sentinel_ablation.py"
    experiments_dir = str(path.parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _write_empty_cohorts(root: Path) -> None:
    cohort_dir = root / "bench/AOM_v0/benchmarks"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "database_name,dataset,status,reason,n_train,n_test,p,"
        "train_path,test_path,ytrain_path,ytest_path\n"
    )
    (cohort_dir / "cohort_regression.csv").write_text(header, encoding="utf-8")
    (cohort_dir / "cohort_classification.csv").write_text(header, encoding="utf-8")


def _row_audit_kwargs(
    *,
    input_seed: int | None = 31415,
    transform_params: str = "{}",
) -> dict[str, Any]:
    return {
        "audit_oracle": False,
        "audit_label_inputs_used": False,
        "audit_target_inputs_used": False,
        "audit_split_inputs_used": False,
        "audit_source_oracle_used": False,
        "audit_learned": False,
        "audit_real_stat_capture": False,
        "audit_thresholds_modified": False,
        "audit_metrics_modified": False,
        "audit_imputed": False,
        "audit_replays_real_rows": False,
        "profile_input_seed": input_seed,
        "profile_scope": "bench_only_r2a_sentinel_mechanistic_ablation",
        "profile_transform_params": transform_params,
    }


def test_run_ablation_returns_blocked_no_real_data_when_cohorts_are_empty(tmp_path: Path) -> None:
    exp08 = _load_exp08_module()
    _write_empty_cohorts(tmp_path)

    result = exp08.run_ablation(
        root=tmp_path,
        profiles=["r2a_baseline", "r2a_pathlength_drift"],
        n_synthetic_samples=12,
        max_real_samples=12,
        max_sentinel_datasets=2,
        seed=1234,
    )

    assert result["status"] == "blocked_no_real_data"
    assert result["rows"] == []
    assert "r2a_baseline" in result["profiles"]
    assert "r2a_pathlength_drift" in result["profiles"]
    assert result["real_runnable_count"] == 0
    assert result["real_sentinel_candidate_count"] == 0
    assert result["real_selected_count"] == 0
    assert result["sentinel_tokens"] == list(exp08.DEFAULT_SENTINEL_TOKENS)


def _make_real_dataset(
    *,
    source: str = "AOM_regression",
    task: str = "regression",
    database_name: str,
    dataset: str,
) -> Any:
    from nirsyntheticpfn.evaluation.realism import RealDataset

    return RealDataset(
        source=source,
        task=task,
        database_name=database_name,
        dataset=dataset,
        train_path="train.csv",
        test_path="test.csv",
        ytrain_path="ytrain.csv",
        ytest_path="ytest.csv",
        n_train_declared=10,
        n_test_declared=4,
        p_declared=8,
    )


def test_default_sentinel_tokens_cover_primary_and_secondary_groups() -> None:
    exp08 = _load_exp08_module()
    tokens = set(exp08.DEFAULT_SENTINEL_TOKENS)
    # Primary spectroscopic sentinels
    assert {"BEER", "DIESEL", "CORN"}.issubset(tokens)
    # Secondary milk-like
    assert "MILK" in tokens
    # Secondary soil-like
    assert {"LUCAS", "PHOSPHORUS", "MANURE", "SOIL"}.issubset(tokens)
    # Secondary fruit-like
    assert {"BERRY", "PEACH", "PLUMS", "FRUIT"}.issubset(tokens)


def test_select_sentinel_datasets_matches_case_insensitive_across_fields() -> None:
    exp08 = _load_exp08_module()
    candidates = [
        _make_real_dataset(database_name="beer_db", dataset="sample_a"),
        _make_real_dataset(database_name="lucas_topsoil", dataset="row_1"),
        _make_real_dataset(database_name="industrial", dataset="DieselFuel"),
        _make_real_dataset(database_name="random_db", dataset="other"),
        _make_real_dataset(database_name="berry_orchard", dataset="peach_blend"),
    ]

    selected = exp08._select_sentinel_datasets(
        candidates, exp08.DEFAULT_SENTINEL_TOKENS
    )
    selected_keys = {(d.database_name, d.dataset) for d in selected}
    assert ("beer_db", "sample_a") in selected_keys  # BEER
    assert ("lucas_topsoil", "row_1") in selected_keys  # LUCAS / SOIL
    assert ("industrial", "DieselFuel") in selected_keys  # DIESEL (in dataset)
    assert ("berry_orchard", "peach_blend") in selected_keys  # BERRY / PEACH
    assert ("random_db", "other") not in selected_keys


def test_select_sentinel_datasets_preserves_input_order() -> None:
    exp08 = _load_exp08_module()
    candidates = [
        _make_real_dataset(database_name="random_db", dataset="x"),
        _make_real_dataset(database_name="corn_db", dataset="ds1"),
        _make_real_dataset(database_name="beer_db", dataset="ds2"),
    ]
    selected = exp08._select_sentinel_datasets(candidates, ["BEER", "CORN"])
    assert [d.database_name for d in selected] == ["corn_db", "beer_db"]


def test_select_sentinel_datasets_empty_tokens_returns_empty() -> None:
    exp08 = _load_exp08_module()
    candidates = [
        _make_real_dataset(database_name="beer_db", dataset="ds"),
    ]
    assert exp08._select_sentinel_datasets(candidates, []) == []


def test_select_sentinel_datasets_primary_precedes_secondary_under_cap() -> None:
    exp08 = _load_exp08_module()
    # Cohort order intentionally puts a fruit (secondary) row first so a naive
    # input-order selection would prefer the fruit cohort over BEER/CORN.
    candidates = [
        _make_real_dataset(database_name="berry_orchard", dataset="ds_fruit_1"),
        _make_real_dataset(database_name="peach_grove", dataset="ds_fruit_2"),
        _make_real_dataset(database_name="lucas_topsoil", dataset="ds_soil"),
        _make_real_dataset(database_name="beer_archive", dataset="ds_beer"),
        _make_real_dataset(database_name="corn_field", dataset="ds_corn"),
    ]
    selected = exp08._select_sentinel_datasets(
        candidates, exp08.DEFAULT_SENTINEL_TOKENS
    )
    # Primary group first (preserving cohort order within group), then soil,
    # then fruit (preserving cohort order within group).
    assert [d.database_name for d in selected] == [
        "beer_archive",
        "corn_field",
        "lucas_topsoil",
        "berry_orchard",
        "peach_grove",
    ]
    # A small cap must keep only the primary sentinels.
    assert [d.database_name for d in selected[:2]] == ["beer_archive", "corn_field"]


def test_select_sentinel_datasets_multi_token_match_uses_best_priority() -> None:
    exp08 = _load_exp08_module()
    # This dataset matches both a secondary soil token (SOIL) and a primary
    # token (CORN) — it must be ranked alongside primaries.
    multi_match = _make_real_dataset(database_name="corn_soil_lab", dataset="row_1")
    soil_only = _make_real_dataset(database_name="lucas_topsoil", dataset="row_2")
    primary_only = _make_real_dataset(database_name="beer_archive", dataset="row_3")
    selected = exp08._select_sentinel_datasets(
        [soil_only, multi_match, primary_only], exp08.DEFAULT_SENTINEL_TOKENS
    )
    # Both ``multi_match`` (priority 0 via CORN) and ``primary_only`` (BEER) are
    # primary; cohort order within the group is preserved (multi_match first).
    assert [d.database_name for d in selected] == [
        "corn_soil_lab",
        "beer_archive",
        "lucas_topsoil",
    ]


def test_select_sentinel_datasets_custom_tokens_rank_after_known_groups() -> None:
    exp08 = _load_exp08_module()
    candidates = [
        _make_real_dataset(database_name="custom_xyz_db", dataset="row_1"),
        _make_real_dataset(database_name="berry_orchard", dataset="row_2"),
        _make_real_dataset(database_name="beer_archive", dataset="row_3"),
    ]
    # ``XYZ`` is not part of any default group: it must rank below known
    # primary and secondary groups even though it appears first in the cohort.
    selected = exp08._select_sentinel_datasets(
        candidates, ["BEER", "BERRY", "XYZ"]
    )
    assert [d.database_name for d in selected] == [
        "beer_archive",
        "berry_orchard",
        "custom_xyz_db",
    ]


def test_token_priority_map_assigns_expected_groups() -> None:
    exp08 = _load_exp08_module()
    mapping = exp08._token_priority_map(["BEER", "MILK", "soil", "PLUMS", "ZZZ"])
    # primary group is index 0, secondary_milk = 1, secondary_soil = 2,
    # secondary_fruit = 3, custom = len(groups).
    assert mapping["beer"] == 0
    assert mapping["milk"] == 1
    assert mapping["soil"] == 2
    assert mapping["plums"] == 3
    assert mapping["zzz"] == len(exp08.SENTINEL_PRIORITY_GROUPS)


def test_run_ablation_negative_cap_behaves_like_zero(
    tmp_path: Path, monkeypatch: Any
) -> None:
    exp08 = _load_exp08_module()
    _write_empty_cohorts(tmp_path)

    candidates = [
        _make_real_dataset(database_name="random_db", dataset="ds_a"),
        _make_real_dataset(database_name="other_db", dataset="ds_b"),
    ]

    monkeypatch.setattr(
        exp08,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp08,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp08.run_ablation(
        root=tmp_path,
        profiles=["r2a_baseline"],
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=-3,
        seed=1,
        sentinel_tokens=["BEER"],  # would filter to 0 if it were applied
    )
    assert result["real_runnable_count"] == 2
    assert result["real_sentinel_candidate_count"] == 2
    assert result["real_selected_count"] == 2


def test_run_ablation_primary_first_under_low_cap(
    tmp_path: Path, monkeypatch: Any
) -> None:
    exp08 = _load_exp08_module()
    _write_empty_cohorts(tmp_path)

    candidates = [
        # Secondary fruit appears first in the cohort: a non-priority selector
        # would (incorrectly) pick this row before the primaries below.
        _make_real_dataset(database_name="berry_orchard", dataset="ds_fruit"),
        _make_real_dataset(database_name="lucas_topsoil", dataset="ds_soil"),
        _make_real_dataset(database_name="beer_archive", dataset="ds_beer"),
        _make_real_dataset(database_name="diesel_lab", dataset="ds_diesel"),
        _make_real_dataset(database_name="corn_field", dataset="ds_corn"),
    ]

    monkeypatch.setattr(
        exp08,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp08,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp08.run_ablation(
        root=tmp_path,
        profiles=["r2a_baseline"],
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=3,
        seed=1,
    )
    assert result["real_runnable_count"] == 5
    assert result["real_sentinel_candidate_count"] == 5
    assert result["real_selected_count"] == 3
    selected_dbs = {row.dataset.split("/")[0] for row in result["rows"]}
    # All three primaries must be present; neither secondary should appear.
    assert selected_dbs == {"beer_archive", "diesel_lab", "corn_field"}


def test_default_max_sentinel_datasets_covers_all_primary_named_b2_rows() -> None:
    exp08 = _load_exp08_module()
    # Documented B2 named primary cohort budget: 2 BEER + 3 DIESEL + 2 CORN = 7,
    # plus one secondary slot for regression pressure.
    assert exp08.DEFAULT_MAX_SENTINEL_DATASETS == 8
    candidates = [
        _make_real_dataset(database_name="BERRY", dataset="secondary_earlier"),
        _make_real_dataset(database_name="BEER", dataset="beer_1"),
        _make_real_dataset(database_name="BEER", dataset="beer_2"),
        _make_real_dataset(database_name="CORN", dataset="corn_1"),
        _make_real_dataset(database_name="CORN", dataset="corn_2"),
        _make_real_dataset(database_name="DIESEL", dataset="diesel_1"),
        _make_real_dataset(database_name="DIESEL", dataset="diesel_2"),
        _make_real_dataset(database_name="DIESEL", dataset="diesel_3"),
        _make_real_dataset(database_name="MILK", dataset="secondary_later"),
        _make_real_dataset(database_name="random", dataset="not_selected"),
    ]

    selected = exp08._select_sentinel_datasets(
        candidates,
        exp08.DEFAULT_SENTINEL_TOKENS,
    )[: exp08.DEFAULT_MAX_SENTINEL_DATASETS]

    assert [row.database_name for row in selected] == [
        "BEER",
        "BEER",
        "CORN",
        "CORN",
        "DIESEL",
        "DIESEL",
        "DIESEL",
        "MILK",
    ]


def test_run_ablation_zero_cap_keeps_all_runnable_without_token_filter(
    tmp_path: Path, monkeypatch: Any
) -> None:
    exp08 = _load_exp08_module()
    _write_empty_cohorts(tmp_path)

    candidates = [
        _make_real_dataset(database_name="random_db", dataset="ds_a"),
        _make_real_dataset(database_name="other_db", dataset="ds_b"),
        _make_real_dataset(database_name="beer_db", dataset="ds_c"),
    ]

    monkeypatch.setattr(
        exp08,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    # Force every dataset to fall through the blocked path so the run
    # completes without needing real spectra files on disk.
    monkeypatch.setattr(
        exp08,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp08.run_ablation(
        root=tmp_path,
        profiles=["r2a_baseline"],
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=0,
        seed=1,
        sentinel_tokens=["BEER"],  # would filter to 1, but cap<=0 must ignore tokens
    )

    assert result["real_runnable_count"] == 3
    assert result["real_sentinel_candidate_count"] == 3
    assert result["real_selected_count"] == 3


def test_run_ablation_positive_cap_filters_by_tokens_then_truncates(
    tmp_path: Path, monkeypatch: Any
) -> None:
    exp08 = _load_exp08_module()
    _write_empty_cohorts(tmp_path)

    candidates = [
        _make_real_dataset(database_name="random_db", dataset="ds_a"),
        _make_real_dataset(database_name="beer_db", dataset="ds_b"),
        _make_real_dataset(database_name="corn_db", dataset="ds_c"),
        _make_real_dataset(database_name="diesel_db", dataset="ds_d"),
    ]

    monkeypatch.setattr(
        exp08,
        "discover_local_real_datasets",
        lambda root: (list(candidates), []),
    )
    monkeypatch.setattr(
        exp08,
        "load_real_spectra",
        lambda dataset, root: (_ for _ in ()).throw(
            RuntimeError("synthetic-test: no real spectra available")
        ),
    )

    result = exp08.run_ablation(
        root=tmp_path,
        profiles=["r2a_baseline"],
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=2,
        seed=1,
        sentinel_tokens=["BEER", "CORN", "DIESEL"],
    )

    assert result["real_runnable_count"] == 4
    # 3 sentinel-token matches; cap=2 truncates after filtering.
    assert result["real_sentinel_candidate_count"] == 3
    assert result["real_selected_count"] == 2


def test_ablation_row_schema_and_to_dict_keys() -> None:
    exp08 = _load_exp08_module()
    row = exp08.AblationRow(
        status="compared",
        source="AOM_regression",
        task="regression",
        dataset="DB/DS",
        synthetic_preset="grain",
        mechanistic_profile="r2a_pathlength_drift",
        profile_enabled=True,
        profile_seed=1234,
        **_row_audit_kwargs(transform_params='{"factor_distribution":"uniform_0p85_1p15","factor_max":1.1,"factor_min":0.9}'),
        n_real_samples=16,
        n_synthetic_samples=16,
        n_wavelengths=64,
        adversarial_auc=0.91,
        pca_overlap=0.42,
        nearest_neighbor_ratio=1.5,
        derivative_log10_gap=0.3,
        blocked_reason="",
    )
    data = row.to_dict()
    expected_keys = {
        "status",
        "source",
        "task",
        "dataset",
        "synthetic_preset",
        "mechanistic_profile",
        "profile_enabled",
        "profile_seed",
        "audit_oracle",
        "audit_label_inputs_used",
        "audit_target_inputs_used",
        "audit_split_inputs_used",
        "audit_source_oracle_used",
        "audit_learned",
        "audit_real_stat_capture",
        "audit_thresholds_modified",
        "audit_metrics_modified",
        "audit_imputed",
        "audit_replays_real_rows",
        "profile_input_seed",
        "profile_scope",
        "profile_transform_params",
        "n_real_samples",
        "n_synthetic_samples",
        "n_wavelengths",
        "adversarial_auc",
        "pca_overlap",
        "nearest_neighbor_ratio",
        "derivative_log10_gap",
        "blocked_reason",
    }
    assert set(data.keys()) == expected_keys


def test_write_csv_produces_header_and_rows(tmp_path: Path) -> None:
    exp08 = _load_exp08_module()
    rows = [
        exp08.AblationRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="DB/DS",
            synthetic_preset="grain",
            mechanistic_profile="r2a_baseline",
            profile_enabled=False,
            profile_seed=None,
            **_row_audit_kwargs(transform_params='{"effect":"identity_control"}'),
            n_real_samples=16,
            n_synthetic_samples=16,
            n_wavelengths=64,
            adversarial_auc=0.83,
            pca_overlap=0.41,
            nearest_neighbor_ratio=1.4,
            derivative_log10_gap=0.2,
            blocked_reason="",
        ),
        exp08.AblationRow(
            status="compared",
            source="AOM_regression",
            task="regression",
            dataset="DB/DS",
            synthetic_preset="grain",
            mechanistic_profile="r2a_pathlength_drift",
            profile_enabled=True,
            profile_seed=99,
            **_row_audit_kwargs(
                input_seed=2024,
                transform_params='{"factor_distribution":"uniform_0p85_1p15","factor_max":1.15,"factor_min":0.85}',
            ),
            n_real_samples=16,
            n_synthetic_samples=16,
            n_wavelengths=64,
            adversarial_auc=0.81,
            pca_overlap=0.45,
            nearest_neighbor_ratio=1.3,
            derivative_log10_gap=0.18,
            blocked_reason="",
        ),
    ]
    csv_path = tmp_path / "out.csv"

    exp08.write_csv(rows, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert len(records) == 2
    assert records[0]["mechanistic_profile"] == "r2a_baseline"
    assert records[0]["audit_oracle"] == "False"
    assert records[0]["profile_transform_params"] == '{"effect":"identity_control"}'
    assert records[1]["mechanistic_profile"] == "r2a_pathlength_drift"
    assert records[1]["profile_input_seed"] == "2024"
    assert records[1]["profile_transform_params"] == '{"factor_distribution":"uniform_0p85_1p15","factor_max":1.15,"factor_min":0.85}'


def test_write_csv_empty_rows_still_produces_stable_header(tmp_path: Path) -> None:
    exp08 = _load_exp08_module()
    csv_path = tmp_path / "empty.csv"

    exp08.write_csv([], csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        records = list(reader)
    assert records == []
    assert reader.fieldnames == exp08._csv_fieldnames()
    assert "audit_oracle" in reader.fieldnames
    assert "profile_transform_params" in reader.fieldnames


def test_render_markdown_does_not_claim_b2_or_b3_pass(tmp_path: Path) -> None:
    exp08 = _load_exp08_module()
    result = {
        "status": "done",
        "rows": [
            exp08.AblationRow(
                status="compared",
                source="AOM_regression",
                task="regression",
                dataset="DB/DS",
                synthetic_preset="grain",
                mechanistic_profile="r2a_baseline",
                profile_enabled=False,
                profile_seed=None,
                **_row_audit_kwargs(transform_params='{"effect":"identity_control"}'),
                n_real_samples=16,
                n_synthetic_samples=16,
                n_wavelengths=64,
                adversarial_auc=0.91,
                pca_overlap=0.42,
                nearest_neighbor_ratio=1.4,
                derivative_log10_gap=0.21,
                blocked_reason="",
            ),
        ],
        "real_runnable_count": 5,
        "real_sentinel_candidate_count": 3,
        "real_selected_count": 1,
        "profiles": ["r2a_baseline", "r2a_pathlength_drift"],
        "sentinel_tokens": list(exp08.DEFAULT_SENTINEL_TOKENS),
    }

    md = exp08.render_markdown(
        result=result,
        report_path=tmp_path / "r2a.md",
        csv_path=tmp_path / "r2a.csv",
        profiles=["r2a_baseline", "r2a_pathlength_drift"],
        n_synthetic_samples=16,
        max_real_samples=16,
        max_sentinel_datasets=1,
        seed=1234,
        sentinel_tokens=list(exp08.DEFAULT_SENTINEL_TOKENS),
    )

    assert "real_sentinel_candidate_count" in md
    assert "Real sentinel candidates after token filter: 3" in md
    assert "BEER" in md
    assert "--sentinel-tokens" in md

    assert "Report-only" in md
    assert "B2/B3/B4/B5" in md
    assert "non-gate" in md.lower() or "non gate" in md.lower()
    assert "real_synthetic_scorecards" not in md
    assert "adversarial_auc.md" not in md
    for flag in (
        "oracle=false",
        "label_inputs_used=false",
        "target_inputs_used=false",
        "split_inputs_used=false",
        "source_oracle_used=false",
        "learned=false",
        "real_stat_capture=false",
        "thresholds_modified=false",
        "metrics_modified=false",
        "imputed=false",
        "replays_real_rows=false",
    ):
        assert flag in md, f"missing audit flag {flag!r}"
    assert "profile_input_seed" in md
    assert "profile_transform_params" in md


def test_default_report_paths_do_not_collide_with_existing_gate_reports() -> None:
    exp08 = _load_exp08_module()
    # Gate report names (frozen by docs/06).
    forbidden = {
        "real_synthetic_scorecards.md",
        "real_synthetic_scorecards.csv",
        "adversarial_auc.md",
        "adversarial_auc.csv",
        "transfer_validation.md",
        "transfer_validation.csv",
        "minimal_ablation_attribution.md",
        "minimal_ablation_attribution.csv",
        "encoder_tabpfn_gate.md",
        "encoder_tabpfn_gate.csv",
        "nirs_icl_gate_precheck.md",
        "nirs_icl_gate_precheck.csv",
        "integration_gate_status.md",
    }
    assert exp08.DEFAULT_REPORT.name not in forbidden
    assert exp08.DEFAULT_CSV.name not in forbidden
    assert exp08.DEFAULT_REPORT.name.startswith("r2a_")
    assert exp08.DEFAULT_CSV.name.startswith("r2a_")


def test_aggregate_reports_smoke_failures_against_provisional_threshold() -> None:
    exp08 = _load_exp08_module()
    rows = [
        exp08.AblationRow(
            status="compared",
            source="s",
            task="t",
            dataset=f"DB/D{i}",
            synthetic_preset="grain",
            mechanistic_profile="r2a_baseline",
            profile_enabled=False,
            profile_seed=None,
            **_row_audit_kwargs(transform_params='{"effect":"identity_control"}'),
            n_real_samples=10,
            n_synthetic_samples=10,
            n_wavelengths=32,
            adversarial_auc=auc,
            pca_overlap=0.5,
            nearest_neighbor_ratio=1.0,
            derivative_log10_gap=0.1,
            blocked_reason="",
        )
        for i, auc in enumerate([0.5, 0.9, 1.0])
    ]
    summary = exp08._aggregate_by_profile(rows)
    assert summary["r2a_baseline"]["n"] == 3
    assert summary["r2a_baseline"]["fail"] == 2  # 0.9 and 1.0 exceed smoke threshold 0.85
    assert np.isclose(summary["r2a_baseline"]["mean"], np.mean([0.5, 0.9, 1.0]))


def test_run_ablation_validates_profile_names(tmp_path: Path) -> None:
    exp08 = _load_exp08_module()
    _write_empty_cohorts(tmp_path)

    import pytest

    with pytest.raises(ValueError):
        exp08.run_ablation(
            root=tmp_path,
            profiles=["not_a_profile"],
            n_synthetic_samples=8,
            max_real_samples=8,
            max_sentinel_datasets=1,
            seed=1,
        )


def test_module_exposes_known_profile_set() -> None:
    exp08 = _load_exp08_module()
    assert "r2a_baseline" in R2A_MECHANISTIC_PROFILES
    # exp08 always prepends r2a_baseline to user-supplied profiles
    result = exp08.run_ablation(
        root=Path("/tmp/this/path/should/not/exist/r2a"),
        profiles=["r2a_pathlength_drift"],
        n_synthetic_samples=4,
        max_real_samples=4,
        max_sentinel_datasets=1,
        seed=1,
    )
    assert result["profiles"][0] == "r2a_baseline"
    assert "r2a_pathlength_drift" in result["profiles"]
