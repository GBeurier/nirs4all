"""Phase B2 real/synthetic scorecards for local benchmark cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from exp00_smoke_prior_dataset import PRESET_SOURCE_OVERRIDES, PRESETS, _preset_source
from nirsyntheticpfn.adapters.builder_adapter import build_synthetic_dataset_run
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_domain, canonicalize_prior_config
from nirsyntheticpfn.evaluation.prior_checks import PHASE_A_GATE_OVERRIDE
from nirsyntheticpfn.evaluation.realism import (
    PROVISIONAL_THRESHOLDS,
    CohortInventory,
    ComparisonSpace,
    RealDataset,
    ScorecardRow,
    align_to_real_grid,
    apply_real_marginal_calibration,
    blocked_scorecard_row,
    compare_real_synthetic,
    discover_local_real_datasets,
    downsample_rows,
    fit_real_marginal_calibration,
    is_index_fallback_grid,
    load_real_spectra,
    sanitize_finite_spectra,
    synthetic_only_row,
    write_scorecard_csv,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config
from nirs4all.synthesis.instruments import get_instrument_archetype
from nirs4all.synthesis.prior import MatrixType

DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv")
MIN_ON_DEMAND_SPECTRAL_RESOLUTION_NM = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-synthetic-samples", type=int, default=80)
    parser.add_argument("--max-real-samples", type=int, default=80)
    parser.add_argument(
        "--max-real-datasets",
        type=int,
        default=0,
        help="0 means every runnable local cohort row.",
    )
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    root = _repo_root()
    result = run_scorecards(
        root=root,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_real_datasets=args.max_real_datasets,
        seed=args.seed,
    )
    write_scorecard_csv(result["rows"], args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            result=result,
            report_path=args.report,
            csv_path=args.csv,
            n_synthetic_samples=args.n_synthetic_samples,
            max_real_samples=args.max_real_samples,
            max_real_datasets=args.max_real_datasets,
            seed=args.seed,
            git_status=_git_status_summary(root),
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.csv)


def run_scorecards(
    *,
    root: Path,
    n_synthetic_samples: int,
    max_real_samples: int,
    max_real_datasets: int,
    seed: int,
) -> dict[str, Any]:
    real_datasets, inventories = discover_local_real_datasets(root)
    selected = real_datasets if max_real_datasets <= 0 else real_datasets[:max_real_datasets]
    synthetic_run_count = 0
    rows: list[ScorecardRow] = []
    load_failures: list[dict[str, Any]] = []

    if not selected:
        first_preset = PRESETS[0][0]
        first_run = _build_default_synthetic_run(
            preset=first_preset,
            n_samples=n_synthetic_samples,
            seed=seed,
        )
        synthetic_run_count += 1
        rows.append(synthetic_only_row(
            synthetic_X=first_run.X,
            synthetic_wavelengths=first_run.wavelengths,
            synthetic_preset=first_preset,
            blocked_reason="blocked_no_real_data: no runnable local benchmark rows loaded",
            synthetic_mapping_strategy="synthetic_only",
            synthetic_mapping_reason="no real dataset was selected",
        ))
        return {
            "status": "blocked_no_real_data",
            "rows": rows,
            "inventories": inventories,
            "real_runnable_count": len(real_datasets),
            "real_selected_count": 0,
            "synthetic_run_count": synthetic_run_count,
            "load_failures": load_failures,
        }

    available_presets = [domain_alias for domain_alias, _, _ in PRESETS]
    for dataset in selected:
        original_preset, mapping_strategy, mapping_reason = select_synthetic_preset_for_dataset(dataset)
        preset = original_preset
        remap_audit: dict[str, Any] = {
            "enabled": False,
            "original_preset": original_preset,
            "selected_preset": original_preset,
            "reason": "not_evaluated",
            "is_index_fallback": False,
            "oracle": False,
            "label_inputs_used": False,
            "target_inputs_used": False,
            "split_inputs_used": False,
            "source_oracle_used": False,
            "thresholds_modified": False,
            "metrics_modified": False,
            "imputed": False,
            "replays_real_rows": False,
        }
        sanitation_real_audit: dict[str, Any] = {"action": "not_run", "side": "real"}
        sanitation_synthetic_audit: dict[str, Any] = {"action": "not_run", "side": "synthetic"}
        covariance_metadata: dict[str, Any] = _covariance_disabled_metadata()
        marginal_calibration_metadata: dict[str, Any] = {"enabled": False, "reason": "not_run"}
        dataset_name = f"{dataset.database_name}/{dataset.dataset}"
        try:
            real_X_raw, real_wavelengths_raw = load_real_spectra(dataset, root=root)
            if is_index_fallback_grid(real_wavelengths_raw):
                preset, remap_audit = grid_compatible_preset_fallback(
                    original_preset=original_preset,
                    real_wavelengths=real_wavelengths_raw,
                    available_presets=available_presets,
                    allow_cross_domain_fallback=False,
                )
                sanitation_real_audit = {
                    "action": "skipped_due_to_wavelength_grid_unknown",
                    "side": "real",
                    "thresholds_modified": False,
                    "metrics_modified": False,
                    "imputed": False,
                    "oracle": False,
                    "label_inputs_used": False,
                    "target_inputs_used": False,
                    "split_inputs_used": False,
                    "source_oracle_used": False,
                }
                blocked_reason = (
                    "wavelength_grid_unknown: real wavelength headers were not parsed as a "
                    "physical wavelength grid; refusing synthetic generation, clipping, and scoring"
                )
                rows.append(blocked_scorecard_row(
                    source=dataset.source,
                    task=dataset.task,
                    dataset=dataset_name,
                    synthetic_preset=preset,
                    blocked_reason=blocked_reason,
                    comparison_space="uncalibrated_raw",
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=_audited_mapping_reason(
                        mapping_reason=mapping_reason,
                        generation_metadata={
                            "preset": preset,
                            "skipped": True,
                            "reason": blocked_reason,
                        },
                        calibration_metadata={
                            "marginal": marginal_calibration_metadata,
                            "covariance": covariance_metadata,
                        },
                        sanitation_audit={
                            "real": sanitation_real_audit,
                            "synthetic": sanitation_synthetic_audit,
                        },
                        remap_audit=remap_audit,
                    ),
                ))
                load_failures.append({
                    "source": dataset.source,
                    "task": dataset.task,
                    "dataset": dataset_name,
                    "failure_class": "wavelength_grid_unknown",
                    "reason": blocked_reason,
                    "paths": {
                        "train_path": dataset.train_path,
                        "test_path": dataset.test_path,
                        "ytrain_path": dataset.ytrain_path,
                        "ytest_path": dataset.ytest_path,
                    },
                })
                continue
            sanitized_real, sanitized_wavelengths, sanitation_real_audit, real_blocked_reason = (
                sanitize_finite_spectra(
                    real_X_raw,
                    real_wavelengths_raw,
                    side="real",
                )
            )
            if sanitized_real is not None:
                sanitized_real = downsample_rows(
                    sanitized_real,
                    max_rows=max_real_samples,
                    random_state=_stable_dataset_seed(seed, dataset, "real_downsample"),
                )
                sanitation_real_audit["downsampled_to"] = int(sanitized_real.shape[0])
            if real_blocked_reason is not None or sanitized_real is None:
                rows.append(blocked_scorecard_row(
                    source=dataset.source,
                    task=dataset.task,
                    dataset=dataset_name,
                    synthetic_preset=preset,
                    blocked_reason=f"non_finite_spectra: {real_blocked_reason}",
                    comparison_space="uncalibrated_raw",
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=_audited_mapping_reason(
                        mapping_reason=mapping_reason,
                        generation_metadata={"preset": preset, "skipped": True},
                        calibration_metadata={
                            "marginal": marginal_calibration_metadata,
                            "covariance": covariance_metadata,
                        },
                        sanitation_audit={
                            "real": sanitation_real_audit,
                            "synthetic": sanitation_synthetic_audit,
                        },
                        remap_audit=remap_audit,
                    ),
                ))
                load_failures.append({
                    "source": dataset.source,
                    "task": dataset.task,
                    "dataset": dataset_name,
                    "failure_class": "non_finite_spectra",
                    "reason": real_blocked_reason or "sanitation returned None",
                    "paths": {
                        "train_path": dataset.train_path,
                        "test_path": dataset.test_path,
                        "ytrain_path": dataset.ytrain_path,
                        "ytest_path": dataset.ytest_path,
                    },
                })
                continue
            assert sanitized_real is not None and sanitized_wavelengths is not None
            real_X = sanitized_real
            real_wavelengths = sanitized_wavelengths
            if is_index_fallback_grid(real_wavelengths):
                preset, remap_audit = grid_compatible_preset_fallback(
                    original_preset=original_preset,
                    real_wavelengths=real_wavelengths,
                    available_presets=available_presets,
                    allow_cross_domain_fallback=False,
                )
                blocked_reason = (
                    "wavelength_grid_unknown: real wavelength headers were not parsed as a "
                    "physical wavelength grid; refusing synthetic generation, clipping, and scoring"
                )
                rows.append(blocked_scorecard_row(
                    source=dataset.source,
                    task=dataset.task,
                    dataset=dataset_name,
                    synthetic_preset=preset,
                    blocked_reason=blocked_reason,
                    comparison_space="uncalibrated_raw",
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=_audited_mapping_reason(
                        mapping_reason=mapping_reason,
                        generation_metadata={
                            "preset": preset,
                            "skipped": True,
                            "reason": blocked_reason,
                        },
                        calibration_metadata={
                            "marginal": marginal_calibration_metadata,
                            "covariance": covariance_metadata,
                        },
                        sanitation_audit={
                            "real": sanitation_real_audit,
                            "synthetic": sanitation_synthetic_audit,
                        },
                        remap_audit=remap_audit,
                    ),
                ))
                load_failures.append({
                    "source": dataset.source,
                    "task": dataset.task,
                    "dataset": dataset_name,
                    "failure_class": "wavelength_grid_unknown",
                    "reason": blocked_reason,
                    "paths": {
                        "train_path": dataset.train_path,
                        "test_path": dataset.test_path,
                        "ytrain_path": dataset.ytrain_path,
                        "ytest_path": dataset.ytest_path,
                    },
                })
                continue
            preset, remap_audit = grid_compatible_preset_fallback(
                original_preset=original_preset,
                real_wavelengths=real_wavelengths,
                available_presets=available_presets,
                allow_cross_domain_fallback=True,
            )
            if remap_audit["reason"] == "domain_wavelength_support":
                blocked_reason = (
                    "domain_wavelength_support: semantic matrix-first preset "
                    f"{original_preset!r} has no supported physical wavelength overlap; "
                    "refusing cross-domain grid-compatible fallback"
                )
                rows.append(blocked_scorecard_row(
                    source=dataset.source,
                    task=dataset.task,
                    dataset=dataset_name,
                    synthetic_preset=preset,
                    blocked_reason=blocked_reason,
                    comparison_space="uncalibrated_raw",
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=_audited_mapping_reason(
                        mapping_reason=mapping_reason,
                        generation_metadata={
                            "preset": preset,
                            "skipped": True,
                            "reason": blocked_reason,
                        },
                        calibration_metadata={
                            "marginal": marginal_calibration_metadata,
                            "covariance": covariance_metadata,
                        },
                        sanitation_audit={
                            "real": sanitation_real_audit,
                            "synthetic": sanitation_synthetic_audit,
                        },
                        remap_audit=remap_audit,
                    ),
                ))
                load_failures.append({
                    "source": dataset.source,
                    "task": dataset.task,
                    "dataset": dataset_name,
                    "failure_class": "domain_wavelength_support",
                    "reason": blocked_reason,
                    "paths": {
                        "train_path": dataset.train_path,
                        "test_path": dataset.test_path,
                        "ytrain_path": dataset.ytrain_path,
                        "ytest_path": dataset.ytest_path,
                    },
                })
                continue
            synthetic_run, generation_metadata = build_on_demand_synthetic_run_for_dataset(
                dataset=dataset,
                preset=preset,
                real_wavelengths=real_wavelengths,
                n_samples=n_synthetic_samples,
                seed=seed,
            )
            synthetic_run_count += 1
            sanitized_syn, sanitized_syn_wl, sanitation_synthetic_audit, syn_blocked_reason = (
                sanitize_finite_spectra(
                    synthetic_run.X,
                    synthetic_run.wavelengths,
                    side="synthetic",
                )
            )
            if sanitized_syn is not None:
                sanitized_syn = downsample_rows(
                    sanitized_syn,
                    max_rows=max_real_samples,
                    random_state=_stable_dataset_seed(seed, dataset, "synthetic_downsample"),
                )
                sanitation_synthetic_audit["downsampled_to"] = int(sanitized_syn.shape[0])
            if syn_blocked_reason is not None or sanitized_syn is None:
                rows.append(blocked_scorecard_row(
                    source=dataset.source,
                    task=dataset.task,
                    dataset=dataset_name,
                    synthetic_preset=preset,
                    blocked_reason=f"non_finite_spectra: {syn_blocked_reason}",
                    comparison_space="uncalibrated_raw",
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=_audited_mapping_reason(
                        mapping_reason=mapping_reason,
                        generation_metadata=generation_metadata,
                        calibration_metadata={
                            "marginal": marginal_calibration_metadata,
                            "covariance": covariance_metadata,
                        },
                        sanitation_audit={
                            "real": sanitation_real_audit,
                            "synthetic": sanitation_synthetic_audit,
                        },
                        remap_audit=remap_audit,
                    ),
                ))
                load_failures.append({
                    "source": dataset.source,
                    "task": dataset.task,
                    "dataset": dataset_name,
                    "failure_class": "non_finite_spectra",
                    "reason": syn_blocked_reason or "synthetic sanitation returned None",
                    "paths": {
                        "train_path": dataset.train_path,
                        "test_path": dataset.test_path,
                        "ytrain_path": dataset.ytrain_path,
                        "ytest_path": dataset.ytest_path,
                    },
                })
                continue
            assert sanitized_syn is not None and sanitized_syn_wl is not None
            synthetic_X = sanitized_syn
            synthetic_wavelengths = sanitized_syn_wl
            # Authoritative lane: keep an uncalibrated copy of synthetic spectra and
            # score it before any marginal calibration is fitted or applied. This is
            # the gate-driving lane; calibrated_raw_diagnostic and snv lanes that
            # follow are diagnostics that cannot override an uncalibrated_raw failure.
            synthetic_X_uncalibrated = np.asarray(synthetic_X, dtype=float).copy()
            synthetic_wavelengths_uncalibrated = np.asarray(
                synthetic_wavelengths, dtype=float
            ).copy()
            uncal_real_aligned, uncal_synthetic_aligned, uncal_aligned_wavelengths = (
                align_to_real_grid(
                    real_X,
                    real_wavelengths,
                    synthetic_X_uncalibrated,
                    synthetic_wavelengths_uncalibrated,
                )
            )
            uncalibrated_marginal_metadata: dict[str, Any] = {
                "enabled": False,
                "reason": "uncalibrated_raw_lane_skips_marginal_calibration",
                "status": "disabled",
                "oracle": False,
                "label_inputs_used": False,
                "target_inputs_used": False,
                "split_inputs_used": False,
                "source_oracle_used": False,
                "thresholds_modified": False,
                "metrics_modified": False,
                "imputed": False,
                "replays_real_rows": False,
            }
            uncalibrated_audited_reason = _audited_mapping_reason(
                mapping_reason=mapping_reason,
                generation_metadata=generation_metadata,
                calibration_metadata={
                    "marginal": uncalibrated_marginal_metadata,
                    "covariance": covariance_metadata,
                },
                sanitation_audit={
                    "real": sanitation_real_audit,
                    "synthetic": sanitation_synthetic_audit,
                },
                remap_audit=remap_audit,
            )
            rows.append(compare_real_synthetic(
                real_X=uncal_real_aligned,
                real_wavelengths=uncal_aligned_wavelengths,
                synthetic_X=uncal_synthetic_aligned,
                synthetic_wavelengths=uncal_aligned_wavelengths,
                dataset=dataset_name,
                source=dataset.source,
                task=dataset.task,
                synthetic_preset=preset,
                comparison_space="uncalibrated_raw",
                synthetic_mapping_strategy=mapping_strategy,
                synthetic_mapping_reason=uncalibrated_audited_reason,
                random_state=_stable_dataset_seed(seed, dataset, "metrics:uncalibrated_raw"),
            ))
            calibration = fit_real_marginal_calibration(real_X, real_wavelengths)
            synthetic_X, marginal_calibration_metadata = apply_real_marginal_calibration(
                synthetic_X,
                synthetic_wavelengths,
                calibration,
            )
            (
                sanitized_calibrated_syn,
                sanitized_calibrated_wl,
                sanitation_synthetic_post_audit,
                syn_post_blocked_reason,
            ) = sanitize_finite_spectra(
                synthetic_X,
                synthetic_wavelengths,
                side="synthetic_post_marginal",
            )
            sanitation_synthetic_audit = _merge_synthetic_sanitation_audits(
                post_generation=sanitation_synthetic_audit,
                post_marginal=sanitation_synthetic_post_audit,
            )
            if syn_post_blocked_reason is not None or sanitized_calibrated_syn is None:
                rows.append(blocked_scorecard_row(
                    source=dataset.source,
                    task=dataset.task,
                    dataset=dataset_name,
                    synthetic_preset=preset,
                    blocked_reason=f"non_finite_spectra: {syn_post_blocked_reason}",
                    comparison_space="calibrated_raw_diagnostic",
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=_audited_mapping_reason(
                        mapping_reason=mapping_reason,
                        generation_metadata=generation_metadata,
                        calibration_metadata={
                            "marginal": marginal_calibration_metadata,
                            "covariance": covariance_metadata,
                        },
                        sanitation_audit={
                            "real": sanitation_real_audit,
                            "synthetic": sanitation_synthetic_audit,
                        },
                        remap_audit=remap_audit,
                    ),
                ))
                load_failures.append({
                    "source": dataset.source,
                    "task": dataset.task,
                    "dataset": dataset_name,
                    "failure_class": "non_finite_spectra",
                    "reason": syn_post_blocked_reason or "post-marginal sanitation returned None",
                    "paths": {
                        "train_path": dataset.train_path,
                        "test_path": dataset.test_path,
                        "ytrain_path": dataset.ytrain_path,
                        "ytest_path": dataset.ytest_path,
                    },
                })
                continue
            assert sanitized_calibrated_syn is not None and sanitized_calibrated_wl is not None
            synthetic_X = sanitized_calibrated_syn
            synthetic_wavelengths = sanitized_calibrated_wl
            real_aligned, synthetic_aligned, aligned_wavelengths = align_to_real_grid(
                real_X,
                real_wavelengths,
                synthetic_X,
                synthetic_wavelengths,
            )
            audited_mapping_reason = _audited_mapping_reason(
                mapping_reason=mapping_reason,
                generation_metadata=generation_metadata,
                calibration_metadata={
                    "marginal": marginal_calibration_metadata,
                    "covariance": covariance_metadata,
                },
                sanitation_audit={
                    "real": sanitation_real_audit,
                    "synthetic": sanitation_synthetic_audit,
                },
                remap_audit=remap_audit,
            )
            diagnostic_spaces: tuple[ComparisonSpace, ...] = (
                "calibrated_raw_diagnostic",
                "snv",
            )
            for comparison_space in diagnostic_spaces:
                rows.append(compare_real_synthetic(
                    real_X=real_aligned,
                    real_wavelengths=aligned_wavelengths,
                    synthetic_X=synthetic_aligned,
                    synthetic_wavelengths=aligned_wavelengths,
                    dataset=dataset_name,
                    source=dataset.source,
                    task=dataset.task,
                    synthetic_preset=preset,
                    comparison_space=comparison_space,
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=audited_mapping_reason,
                    random_state=_stable_dataset_seed(seed, dataset, f"metrics:{comparison_space}"),
                ))
        except Exception as exc:
            failure_class = _failure_class(exc)
            rows.append(blocked_scorecard_row(
                source=dataset.source,
                task=dataset.task,
                dataset=dataset_name,
                synthetic_preset=preset,
                blocked_reason=f"{failure_class}: {exc}",
                comparison_space="uncalibrated_raw",
                synthetic_mapping_strategy=mapping_strategy,
                synthetic_mapping_reason=_audited_mapping_reason(
                    mapping_reason=mapping_reason,
                    generation_metadata={"preset": preset, "exception": str(exc)},
                    calibration_metadata={
                        "marginal": marginal_calibration_metadata,
                        "covariance": covariance_metadata,
                    },
                    sanitation_audit={
                        "real": sanitation_real_audit,
                        "synthetic": sanitation_synthetic_audit,
                    },
                    remap_audit=remap_audit,
                ),
            ))
            load_failures.append({
                "source": dataset.source,
                "task": dataset.task,
                "dataset": dataset_name,
                "failure_class": failure_class,
                "reason": str(exc),
                "paths": {
                    "train_path": dataset.train_path,
                    "test_path": dataset.test_path,
                    "ytrain_path": dataset.ytrain_path,
                    "ytest_path": dataset.ytest_path,
                },
            })

    compared_count = sum(
        1
        for row in rows
        if row.status == "compared" and row.comparison_space == "uncalibrated_raw"
    )
    status = "done" if compared_count else "blocked_no_successful_comparisons"
    if not rows:
        status = "blocked_no_real_data"
        first_preset = PRESETS[0][0]
        first_run = _build_default_synthetic_run(
            preset=first_preset,
            n_samples=n_synthetic_samples,
            seed=seed,
        )
        synthetic_run_count += 1
        rows.append(synthetic_only_row(
            synthetic_X=first_run.X,
            synthetic_wavelengths=first_run.wavelengths,
            synthetic_preset=first_preset,
            blocked_reason="blocked_no_real_data: selected real rows failed to load",
            synthetic_mapping_strategy="synthetic_only",
            synthetic_mapping_reason="selected real rows failed before any comparison",
        ))
    return {
        "status": status,
        "rows": rows,
        "inventories": inventories,
        "real_runnable_count": len(real_datasets),
        "real_selected_count": len(selected),
        "synthetic_run_count": synthetic_run_count,
        "load_failures": load_failures,
    }


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    n_synthetic_samples: int,
    max_real_samples: int,
    max_real_datasets: int,
    seed: int,
    git_status: dict[str, Any],
) -> str:
    rows: list[ScorecardRow] = result["rows"]
    inventories: list[CohortInventory] = result["inventories"]
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-real-datasets {max_real_datasets} "
        f"--seed {seed}"
    )
    compared = [row for row in rows if row.status == "compared"]
    uncalibrated_compared = [
        row for row in compared
        if row.comparison_space == "uncalibrated_raw"
    ]
    calibrated_diagnostic_compared = [
        row for row in compared
        if row.comparison_space == "calibrated_raw_diagnostic"
    ]
    snv_compared = [
        row for row in compared
        if row.comparison_space == "snv"
    ]
    blocked_rows = [row for row in rows if row.status == "blocked"]
    blocked_for_evidence = result["status"] != "done"
    auc_failures = [
        row for row in uncalibrated_compared
        if (
            row.adversarial_auc is not None
            and row.adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
        )
    ]
    pca_failures = [
        row for row in uncalibrated_compared
        if (
            row.pca_overlap is not None
            and row.pca_overlap < PROVISIONAL_THRESHOLDS["pca_overlap_min"]
        )
    ]
    calibrated_diagnostic_auc_failures = [
        row for row in calibrated_diagnostic_compared
        if (
            row.adversarial_auc is not None
            and row.adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
        )
    ]
    snv_auc_failures = [
        row for row in snv_compared
        if (
            row.adversarial_auc is not None
            and row.adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
        )
    ]
    lines = [
        "# Real/Synthetic Scorecards",
        "",
        "## Objective",
        "",
        "Standardize Phase B2 scorecards for local real benchmark cohorts against A2 synthetic preset datasets.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Outputs",
        "",
        f"- Markdown: `{report_path}`",
        f"- CSV metrics summary: `{csv_path}`",
        "",
        "## Phase A Gate Override",
        "",
        f"- `phase_a_gate_override`: `{PHASE_A_GATE_OVERRIDE}`",
        "- A3 fitted-only real-fit gate remains failed/documented and is not hidden by this B2 report.",
        "- B2 comparisons are realism diagnostics only; they do not establish downstream transfer benefit.",
        "",
        "## Config",
        "",
        f"- Seed: {seed}",
        f"- A2 synthetic presets generated: {result['synthetic_run_count']}",
        f"- Synthetic samples per preset: {n_synthetic_samples}",
        f"- Real samples per scored dataset cap: {max_real_samples}",
        f"- Real dataset cap: {'all runnable rows' if max_real_datasets <= 0 else max_real_datasets}",
        "- Comparison spaces: uncalibrated_raw, calibrated_raw_diagnostic, snv",
        "- Thresholds are provisional smoke thresholds, not calibrated domain gates.",
        "- Primary decisions and gates use only `comparison_space == \"uncalibrated_raw\"`; the historical `comparison_space == \"raw\"` lane is retired and `calibrated_raw_diagnostic` plus `snv` are diagnostics that cannot override an uncalibrated_raw failure.",
        "- The uncalibrated_raw lane is scored on synthetic spectra before any marginal calibration is fitted or applied, so it remains the authoritative gate.",
        "- Marginal calibration is applied only to the calibrated_raw_diagnostic and snv lanes; calibration does not change metric definitions or thresholds.",
        "",
        "## Git Status",
        "",
        _git_status_section(git_status),
        "",
        "## Real Cohort Inventory",
        "",
        "| source | cohort path | exists | total rows | status ok rows | runnable rows | rows with missing paths |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for inventory in inventories:
        lines.append(
            f"| `{inventory.source}` | `{inventory.path}` | `{inventory.exists}` | "
            f"{inventory.total_rows} | {inventory.ok_rows} | {inventory.runnable_rows} | "
            f"{inventory.missing_rows} |"
        )

    lines.extend([
        "",
        "## Missing Paths",
        "",
        _missing_paths_section(inventories),
        "",
        "## Scorecard Summary",
        "",
        f"- Report status: `{result['status']}`",
        f"- Runnable real rows discovered: {result['real_runnable_count']}",
        f"- Selected real rows attempted: {result['real_selected_count']}",
        f"- Uncalibrated real/synthetic comparison rows written (authoritative): {len(uncalibrated_compared)}",
        f"- Calibrated raw diagnostic comparison rows written: {len(calibrated_diagnostic_compared)}",
        f"- SNV diagnostic comparison rows written: {len(snv_compared)}",
        f"- Blocked selected rows written: {len(blocked_rows)}",
        f"- Synthetic-only dry-run rows written: {sum(1 for row in rows if row.status == 'synthetic_only')}",
        f"- Load/score failures after selection: {len(result['load_failures'])}",
        "",
        "## Synthetic Mapping Strategy",
        "",
        _mapping_strategy_section(rows),
        "",
        "## Real Marginal Calibration",
        "",
        _calibration_section(rows),
        "",
        "## Finite Sanitation",
        "",
        _sanitation_section(rows),
        "",
        "## Grid Remapping",
        "",
        _grid_remap_section(rows),
        "",
        "## Source Overrides",
        "",
        _source_override_section(rows),
        "",
        "## Audit Flags",
        "",
        _audit_flags_section(),
        "",
    ])
    if blocked_for_evidence:
        lines.extend([
            "No successful real/synthetic comparison was produced. Blocked rows are retained in the CSV, ",
            "so the run must not be interpreted as real/synthetic evidence.",
            "",
        ])
    elif max_real_datasets > 0 and result["real_runnable_count"] > result["real_selected_count"]:
        lines.extend([
            "The full local cohort is available beyond this capped run. Rerun with ",
            "`--max-real-datasets 0` to score every runnable row.",
            "",
        ])

    uncalibrated_aucs = [
        float(row.adversarial_auc)
        for row in uncalibrated_compared
        if row.adversarial_auc is not None
    ]
    uncalibrated_mean_auc = float(np.mean(uncalibrated_aucs)) if uncalibrated_aucs else None

    lines.extend([
        "## Diagnostic Outcome",
        "",
        (
            "No uncalibrated_raw real/synthetic diagnostic outcome is available because no uncalibrated_raw comparison was produced."
            if blocked_for_evidence
            else (
                f"- Uncalibrated_raw adversarial AUC smoke failures: {len(auc_failures)}/{len(uncalibrated_compared)} compared rows."
            )
        ),
    ])
    if not blocked_for_evidence:
        lines.extend([
            f"- Uncalibrated_raw mean adversarial AUC: {_fmt(uncalibrated_mean_auc)} (lower is better; provisional smoke threshold {PROVISIONAL_THRESHOLDS['adversarial_auc_smoke']}).",
            f"- Uncalibrated_raw PCA overlap smoke failures: {len(pca_failures)}/{len(uncalibrated_compared)} compared rows.",
            "- Uncalibrated_raw gate remains authoritative: these scorecards do not claim realism success when uncalibrated_raw adversarial AUC exceeds the smoke threshold; calibrated_raw_diagnostic and snv cannot override.",
            "",
            "## R9 Gap Summary",
            "",
            _r9_gap_summary_section(uncalibrated_compared),
            "",
            "## Calibrated Raw Diagnostic",
            "",
            f"- Calibrated_raw_diagnostic adversarial AUC smoke failures: {len(calibrated_diagnostic_auc_failures)}/{len(calibrated_diagnostic_compared)} diagnostic rows.",
            "- Marginal calibration is applied only on the calibrated_raw_diagnostic copy; the uncalibrated_raw lane stays the authoritative gate.",
            "- Calibrated_raw_diagnostic results cannot override an uncalibrated_raw failure and must not be interpreted as B2 success when uncalibrated_raw fails.",
            "",
            "## SNV Diagnostic",
            "",
            f"- SNV adversarial AUC smoke failures: {len(snv_auc_failures)}/{len(snv_compared)} diagnostic rows.",
            "- SNV is applied only after wavelength-grid alignment and only on the calibrated comparison copies.",
            "- SNV diagnostics do not override uncalibrated_raw decisions and must not be interpreted as B2 success when uncalibrated_raw fails.",
            "",
        ])
    else:
        lines.append("")

    lines.extend([
        "## Metrics Table",
        "",
        "| space | dataset | task | synthetic preset | n real | n synthetic | adv AUC | PCA overlap | NN ratio | derivative gap | decision |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| `{row.comparison_space}` | `{row.dataset}` | `{row.task}` | `{row.synthetic_preset}` | "
            f"{row.n_real_samples} | {row.n_synthetic_samples} | "
            f"{_fmt(row.adversarial_auc)} | {_fmt(row.pca_overlap)} | "
            f"{_fmt(row.nearest_neighbor_ratio)} | {_fmt(row.derivative_log10_gap)} | "
            f"`{row.provisional_decision}` |"
        )

    lines.extend([
        "",
        "## Metric Route",
        "",
        "- Spectral mean/variance: reported as mean profile averages for real and synthetic spectra.",
        "- Derivative mean/variance: first derivative per sample, summarized by median.",
        "- Correlation length: first autocorrelation lag below 1/e, summarized by median.",
        "- SNR: high-pass residual estimate from `nirs4all.synthesis.validation.compute_snr`.",
        "- Baseline curvature: degree-3 polynomial residual standard deviation.",
        "- Peak density: peaks per 100 nm using the shared validation helper.",
        "- PCA overlap: 2D PCA histogram intersection when sklearn and sample counts permit.",
        "- Adversarial AUC: PCA plus logistic real/synthetic classifier when sample counts permit.",
        "- Nearest-neighbor ratio: real-to-synthetic nearest distance divided by real-to-real nearest distance.",
        "",
        "## Provisional Thresholds",
        "",
        "| metric | threshold | interpretation |",
        "|---|---:|---|",
        f"| adversarial AUC smoke | {PROVISIONAL_THRESHOLDS['adversarial_auc_smoke']} | lower is better |",
        f"| adversarial AUC stretch | {PROVISIONAL_THRESHOLDS['adversarial_auc_stretch']} | stronger future target |",
        f"| derivative log10 gap | {PROVISIONAL_THRESHOLDS['derivative_order_of_magnitude_gap']} | no order-of-magnitude gap |",
        f"| PCA overlap min | {PROVISIONAL_THRESHOLDS['pca_overlap_min']} | non-empty overlap |",
        f"| NN ratio max | {PROVISIONAL_THRESHOLDS['nearest_neighbor_ratio_max']} | synthetic not much farther than real neighbors |",
        "",
        "## Load Failures",
        "",
        _load_failures_section(result["load_failures"]),
        "",
        "## Decision",
        "",
        (
            "Blocked for real/synthetic evidence: no successful real/synthetic comparisons were produced."
            if blocked_for_evidence
            else _decision_text(
                compared_count=len(uncalibrated_compared),
                auc_failure_count=len(auc_failures),
                blocked_count=len(blocked_rows),
            )
        ),
        "",
        "## Raw Summary JSON",
        "",
        "```json",
        json.dumps(_to_builtin({
            "status": result["status"],
            "inventories": [inventory.__dict__ for inventory in inventories],
            "load_failures": result["load_failures"],
            "row_count": len(rows),
            "git_status": git_status,
        }), indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def select_synthetic_run_for_dataset(
    dataset: RealDataset,
    synthetic_runs: list[tuple[str, Any]],
) -> tuple[str, Any, str, str]:
    """Choose a deterministic synthetic preset from matrix terms, with stable hash fallback."""
    if not synthetic_runs:
        raise ValueError("synthetic_runs must not be empty")

    available = dict(synthetic_runs)
    text, tokens = _dataset_text_and_tokens(dataset)
    rules = _matrix_first_mapping_rules()
    for rule_id, preset, keywords in rules:
        matched = next((keyword for keyword in keywords if _contains_keyword(text, tokens, keyword)), None)
        if matched is None:
            continue
        if preset in available:
            return (
                preset,
                available[preset],
                "matrix_first_dataset",
                _matrix_mapping_reason(
                    rule_id=rule_id,
                    matched=matched,
                    preset=preset,
                    dataset=dataset,
                ),
            )
        return _stable_hash_mapping(
            dataset,
            synthetic_runs,
            reason_prefix=(
                f"matrix_first rule {rule_id!r} matched token {matched!r} "
                f"but preset {preset!r} is unavailable"
            ),
        )
    return _stable_hash_mapping(
        dataset,
        synthetic_runs,
        reason_prefix="no matrix-first dataset rule matched",
    )


def select_synthetic_preset_for_dataset(dataset: RealDataset) -> tuple[str, str, str]:
    """Choose a deterministic preset name without depending on generated run order."""
    preset, _, strategy, reason = select_synthetic_run_for_dataset(
        dataset,
        [(domain_alias, None) for domain_alias, _, _ in PRESETS],
    )
    return preset, strategy, reason


def build_on_demand_synthetic_run_for_dataset(
    *,
    dataset: RealDataset,
    preset: str,
    real_wavelengths: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Generate one synthetic run for a loaded real dataset and wavelength grid."""
    target_type, target_size = _preset_target(preset)
    run_seed = _stable_dataset_seed(seed, dataset, f"on_demand_synthetic:{preset}")
    source_overrides, source_override_metadata = source_overrides_for_dataset(
        dataset=dataset,
        preset=preset,
        real_wavelengths=real_wavelengths,
    )
    source, grid_metadata = synthetic_source_for_real_grid(
        preset=preset,
        target_type=target_type,
        target_size=target_size,
        seed=run_seed,
        real_wavelengths=real_wavelengths,
        source_overrides=source_overrides,
    )
    record = canonicalize_prior_config(
        source,
        allow_bench_wavelength_support_override="_bench_wavelength_support_override" in source,
    )
    run = build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=run_seed)
    generation_metadata = {
        "preset": preset,
        "strategy": "on_demand_per_real_dataset",
        "dataset_key": dataset.key,
        "seed": int(run_seed),
        "grid": grid_metadata,
        "source_overrides": source_override_metadata,
        "canonical_wavelength_policy": record.wavelength_policy,
        "builder_effective_grid": {
            "wavelength_min": float(run.wavelengths[0]),
            "wavelength_max": float(run.wavelengths[-1]),
            "n_wavelengths": int(run.wavelengths.size),
            "median_step": _finite_median_step(run.wavelengths),
        },
    }
    run.metadata["b2_on_demand"] = _to_builtin(generation_metadata)
    return run, generation_metadata


def synthetic_source_for_real_grid(
    *,
    preset: str,
    target_type: str,
    target_size: int,
    seed: int,
    real_wavelengths: np.ndarray,
    source_overrides: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Build an A2 source whose requested grid comes from a real wavelength grid.

    Unsupported physical tails are clipped to the preset domain and instrument
    overlap. A fully invalid or unsupported grid falls back to the preset's
    documented A2 grid instead of silently using dataset order.
    """
    source = _preset_source(
        preset,
        target_type=target_type,
        target_size=target_size,
        seed=seed,
    )
    if source_overrides:
        source.update(source_overrides)
    requested = _real_grid_request(real_wavelengths)
    if not requested["valid"]:
        metadata = {
            **requested,
            "grid_source": "preset_default_fallback",
            "fallback_reason": requested["reason"],
            "source_wavelength_range": list(_source_wavelength_range(source)),
            "source_spectral_resolution": _source_spectral_resolution(source),
        }
        source["_b2_real_grid_metadata"] = metadata
        return source, metadata

    low, high, reason = _supported_real_grid_range(
        preset=preset,
        instrument=str(source["instrument"]),
        real_low=float(requested["wavelength_min"]),
        real_high=float(requested["wavelength_max"]),
        median_step=float(requested["median_step"]),
        domain_range_override=_source_wavelength_support_override_range(source),
    )
    if low is None or high is None:
        metadata = {
            **requested,
            "grid_source": "preset_default_fallback",
            "fallback_reason": reason,
            "source_wavelength_range": list(_source_wavelength_range(source)),
            "source_spectral_resolution": _source_spectral_resolution(source),
        }
        source["_b2_real_grid_metadata"] = metadata
        return source, metadata

    source_resolution = max(
        float(requested["median_step"]),
        MIN_ON_DEMAND_SPECTRAL_RESOLUTION_NM,
    )
    source["wavelength_range"] = (float(low), float(high))
    source["spectral_resolution"] = source_resolution
    grid_source = (
        "real_grid"
        if np.isclose(low, requested["wavelength_min"]) and np.isclose(high, requested["wavelength_max"])
        else "real_grid_clipped_to_supported_overlap"
    )
    metadata = {
        **requested,
        "grid_source": grid_source,
        "fallback_reason": "",
        "clip_reason": reason,
        "source_wavelength_range": [float(low), float(high)],
        "source_spectral_resolution": source_resolution,
        "min_source_spectral_resolution": MIN_ON_DEMAND_SPECTRAL_RESOLUTION_NM,
    }
    source["_b2_real_grid_metadata"] = metadata
    return source, metadata


def _dataset_text_and_tokens(dataset: RealDataset) -> tuple[str, set[str]]:
    text = " ".join([
        dataset.source,
        dataset.task,
        dataset.database_name,
        dataset.dataset,
    ]).lower()
    return text, set(re.findall(r"[a-z0-9]+", text))


def _matrix_first_mapping_rules() -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        ("milk_dairy_matrix", "dairy", ("milk", "dairy", "cheese")),
        ("diesel_fuel_matrix", "fuel", ("diesel", "fuel")),
        ("tablet_pharma_matrix", "tablets", ("tablet", "pharma", "escitalopram")),
        ("beer_beverage_matrix", "wine", ("beer",)),
        ("berry_puree_beverage_matrix", "juice", ("fruitpuree", "puree", "juice", "berry")),
        ("leaf_plant_matrix", "forage", ("leaftraits", "darkresp", "ecosis", "arabidopsis", "leaf")),
        ("soil_mineral_matrix", "soil", ("lucas", "soil", "phosphorus", "quartz", "incombustible")),
        ("cassava_starch_root_proxy", "grain", ("cassava",)),
        ("corn_grain_matrix", "grain", ("corn",)),
        ("grain_cereal_matrix", "grain", ("rice", "wheat", "barley", "cereal", "amylose", "grain")),
        ("meat_matrix", "meat", ("beef", "meat", "marbling", "impurity")),
        ("oilseed_matrix", "oilseeds", ("colza", "oilseed")),
        ("baking_matrix", "baking", ("biscuit", "baking")),
        ("fruit_solid_matrix", "fruit", ("fruit", "peach", "plum", "grape", "pistacia")),
        ("powder_matrix", "powders", ("powder",)),
    ]


def _matrix_mapping_reason(
    *,
    rule_id: str,
    matched: str,
    preset: str,
    dataset: RealDataset,
) -> str:
    return (
        f"matrix_first rule {rule_id!r} matched token {matched!r} in "
        f"{dataset.database_name}/{dataset.dataset} -> {preset}; "
        "source_fields=source,task,database_name,dataset; "
        "oracle=false; non_oracle=true; no_target_or_label=true; "
        "y_labels_splits_targets_not_used=true"
    )


def _contains_keyword(text: str, tokens: set[str], keyword: str) -> bool:
    if keyword == "oil":
        return keyword in tokens
    return keyword in tokens or keyword in text


def source_overrides_for_dataset(
    *,
    dataset: RealDataset,
    preset: str,
    real_wavelengths: np.ndarray | None = None,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Return bench-only source overrides for physical matrix/mode metadata."""
    _, tokens = _dataset_text_and_tokens(dataset)
    overrides = dict(PRESET_SOURCE_OVERRIDES.get(preset, {}))
    reasons: list[str] = []
    rules: list[str] = []
    if overrides:
        reasons.append(f"preset {preset!r} bench source defaults")
        rules.append(f"preset_{preset}_bench_defaults")
    if preset == "dairy" and ({"milk", "dairy"} & tokens):
        overrides.update({
            "matrix_type": _supported_matrix_type("emulsion", fallback="liquid"),
            "particle_size": 7.5,
        })
        components = _domain_component_subset(preset, ("water", "lactose", "casein", "lipid"))
        if components:
            overrides["components"] = components
        reasons.append("milk/dairy matrix uses generic dairy emulsion components")
        rules.append("milk_dairy_matrix_physics")
    if preset == "fuel" and ({"diesel", "fuel"} & tokens):
        overrides.update({
            "matrix_type": "liquid",
            "particle_size": 2.0,
        })
        components = _domain_component_subset(preset, ("alkane", "aromatic", "oil", "methanol"))
        if components:
            overrides["components"] = components
        reasons.append("diesel/fuel matrix uses generic liquid fuel components")
        rules.append("diesel_fuel_matrix_physics")
    if preset == "grain" and "corn" in tokens:
        overrides.update({
            "matrix_type": "granular",
            "particle_size": 250.0,
        })
        components = _domain_component_subset(preset, ("starch", "protein", "moisture", "lipid"))
        if components:
            overrides["components"] = components
        reasons.append("corn/grain matrix uses generic granular grain components")
        rules.append("corn_grain_matrix_physics")
    if preset == "wine" and ({"beer", "wine"} & tokens):
        overrides.update({
            "measurement_mode": "transmittance",
            "matrix_type": "liquid",
            "particle_size": 5.0,
        })
        components = _domain_component_subset(preset, ("water", "ethanol", "glucose", "fructose"))
        if components:
            overrides["components"] = components
        reasons.append("beer/wine beverage matrix keeps liquid transmittance source")
        rules.append("beer_wine_liquid_transmittance")
        support_override = _wavelength_support_override_for_real_grid(
            preset=preset,
            real_wavelengths=real_wavelengths,
            rule="beer_wine_real_grid_support",
        )
        if support_override is not None:
            overrides["_bench_wavelength_support_override"] = support_override
            reasons.append("beer/wine wavelength support override requested for real grid")
            rules.append("beer_wine_real_grid_support")
    if preset == "juice" and ({"fruitpuree", "puree"} & tokens):
        overrides.update({
            "measurement_mode": "transmittance",
            "matrix_type": "emulsion",
            "particle_size": 5.0,
        })
        reasons.append("puree context uses supported emulsion matrix override")
        rules.append("puree_emulsion_matrix_physics")
    instrument_audit = _instrument_override_for_dataset(tokens)
    if instrument_audit["applied"]:
        overrides.update({
            "instrument": instrument_audit["canonical"],
            "instrument_category": instrument_audit["instrument_category"],
        })
        reasons.append(str(instrument_audit["reason"]))
        rules.append(str(instrument_audit["rule"]))
    metadata = {
        "enabled": bool(overrides),
        "overrides": overrides,
        "reasons": reasons,
        "reason": "; ".join(reasons) if reasons else "no source override",
        "rules": rules,
        "rule": rules[0] if rules else "",
        "instrument": instrument_audit,
        "scope": "bench_only_matrix_mode",
        "non_oracle": True,
        "no_target_or_label": True,
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "imputed": False,
        "replays_real_rows": False,
    }
    return overrides, metadata


def _domain_component_subset(preset: str, candidates: tuple[str, ...]) -> list[str]:
    domain_config = get_domain_config(canonicalize_domain(preset))
    allowed: set[str] = set()
    for component in domain_config.typical_components:
        try:
            allowed.add(get_component(str(component)).name)
        except ValueError:
            continue
    selected: list[str] = []
    for candidate in candidates:
        try:
            canonical = get_component(candidate).name
        except ValueError:
            continue
        if canonical in allowed:
            selected.append(canonical)
    return selected


def _supported_matrix_type(preferred: str, *, fallback: str) -> str:
    supported = {matrix.value for matrix in MatrixType}
    if preferred in supported:
        return preferred
    if fallback in supported:
        return fallback
    raise ValueError(f"neither matrix type {preferred!r} nor {fallback!r} is supported")


def _instrument_override_for_dataset(tokens: set[str]) -> dict[str, Any]:
    candidates: list[tuple[str, str, str]] = []
    if "micronir" in tokens:
        candidates.append(("MicroNIR", "viavi_micronir", "instrument_token_micronir"))
    if "neospectra" in tokens or "neoscanner" in tokens:
        candidates.append(("NeoSpectra", "siware_neoscanner", "instrument_token_neospectra"))
    if not candidates:
        return {
            "applied": False,
            "reason": "no instrument token matched",
            "rule": "",
            "token": "",
            "canonical": "",
            "instrument_category": "",
            "non_oracle": True,
            "no_target_or_label": True,
            "imputed": False,
        }
    token, canonical, rule = candidates[0]
    try:
        instrument = get_instrument_archetype(canonical)
    except Exception:
        return {
            "applied": False,
            "reason": f"instrument token {token!r} canonical {canonical!r} is unavailable",
            "rule": rule,
            "token": token,
            "canonical": canonical,
            "instrument_category": "",
            "non_oracle": True,
            "no_target_or_label": True,
            "imputed": False,
        }
    return {
        "applied": True,
        "reason": f"instrument token {token!r} -> canonical {canonical!r}",
        "rule": rule,
        "token": token,
        "canonical": canonical,
        "instrument_category": instrument.category.value,
        "wavelength_range_nm": list(instrument.wavelength_range),
        "non_oracle": True,
        "no_target_or_label": True,
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "imputed": False,
        "replays_real_rows": False,
    }


def _wavelength_support_override_for_real_grid(
    *,
    preset: str,
    real_wavelengths: np.ndarray | None,
    rule: str,
) -> dict[str, Any] | None:
    if real_wavelengths is None:
        return None
    requested = _real_grid_request(real_wavelengths)
    if not requested["valid"]:
        return None
    domain_range = get_domain_config(canonicalize_domain(preset)).wavelength_range
    requested_range = (
        float(requested["wavelength_min"]),
        float(requested["wavelength_max"]),
    )
    if domain_range[0] <= requested_range[0] and requested_range[1] <= domain_range[1]:
        return None
    return {
        "enabled": True,
        "scope": "bench_only_real_grid_support",
        "reason": (
            "B2 compare-space audit requires preserving the real wavelength support "
            f"{requested_range} for preset {preset!r}"
        ),
        "rule": rule,
        "domain_range": requested_range,
        "source_fields": [
            "source",
            "task",
            "database_name",
            "dataset",
            "preset",
            "real_wavelengths",
        ],
        "non_oracle": True,
        "no_target_or_label": True,
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "imputed": False,
        "replays_real_rows": False,
    }


def _source_wavelength_support_override_range(
    source: dict[str, object],
) -> tuple[float, float] | None:
    raw_override = source.get("_bench_wavelength_support_override")
    if not isinstance(raw_override, dict) or raw_override.get("enabled") is not True:
        return None
    raw_range = raw_override.get("domain_range")
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        return None
    return float(raw_range[0]), float(raw_range[1])


def _stable_hash_mapping(
    dataset: RealDataset,
    synthetic_runs: list[tuple[str, Any]],
    *,
    reason_prefix: str,
) -> tuple[str, Any, str, str]:
    ordered_runs = sorted(synthetic_runs, key=lambda item: item[0])
    key = "|".join([
        dataset.source,
        dataset.task,
        dataset.database_name,
        dataset.dataset,
    ])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    position = int(digest[:12], 16) % len(ordered_runs)
    preset, run = ordered_runs[position]
    return (
        preset,
        run,
        "stable_hash_fallback",
        f"{reason_prefix}; sha256({key})[:12]={digest[:12]} -> {preset}",
    )


def _build_default_synthetic_run(*, preset: str, n_samples: int, seed: int) -> Any:
    target_type, target_size = _preset_target(preset)
    source = _preset_source(
        preset,
        target_type=target_type,
        target_size=target_size,
        seed=seed,
    )
    record = canonicalize_prior_config(source)
    return build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=seed)


def _preset_target(preset: str) -> tuple[str, int]:
    for domain_alias, target_type, target_size in PRESETS:
        if domain_alias == preset:
            return target_type, target_size
    raise ValueError(f"unknown synthetic preset {preset!r}")


def _real_grid_request(real_wavelengths: np.ndarray) -> dict[str, Any]:
    wavelengths = np.asarray(real_wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 3:
        return {
            "valid": False,
            "reason": f"real wavelength grid must be one-dimensional with >=3 points, got {wavelengths.shape}",
        }
    if not np.isfinite(wavelengths).all():
        return {
            "valid": False,
            "reason": "real wavelength grid contains non-finite values",
        }
    diffs = np.diff(wavelengths)
    finite_positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if finite_positive_diffs.size != diffs.size:
        return {
            "valid": False,
            "reason": "real wavelength grid must be strictly increasing with finite positive steps",
        }
    return {
        "valid": True,
        "reason": "",
        "real_n_wavelengths": int(wavelengths.size),
        "wavelength_min": float(wavelengths[0]),
        "wavelength_max": float(wavelengths[-1]),
        "median_step": float(np.median(finite_positive_diffs)),
    }


def _source_wavelength_range(source: dict[str, object]) -> tuple[float, float]:
    raw = source["wavelength_range"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"invalid source wavelength_range {raw!r}")
    return float(raw[0]), float(raw[1])


def _source_spectral_resolution(source: dict[str, object]) -> float:
    raw = source["spectral_resolution"]
    if not isinstance(raw, (int, float, str)):
        raise ValueError(f"invalid source spectral_resolution {raw!r}")
    return float(raw)


def _supported_real_grid_range(
    *,
    preset: str,
    instrument: str,
    real_low: float,
    real_high: float,
    median_step: float,
    domain_range_override: tuple[float, float] | None = None,
) -> tuple[float | None, float | None, str]:
    domain_range = (
        domain_range_override
        if domain_range_override is not None
        else get_domain_config(canonicalize_domain(preset)).wavelength_range
    )
    instrument_range = get_instrument_archetype(instrument).wavelength_range
    low = max(real_low, float(domain_range[0]), float(instrument_range[0]))
    high = min(real_high, float(domain_range[1]), float(instrument_range[1]))
    if not low < high:
        return None, None, (
            "real grid has no supported overlap with "
            f"domain_range={domain_range} and instrument_range={instrument_range}"
        )
    if high - low < max(median_step * 2.0, 1e-9):
        return None, None, (
            "real grid supported overlap has fewer than three median-step points "
            f"for range={(low, high)} and median_step={median_step}"
        )
    if np.isclose(low, real_low) and np.isclose(high, real_high):
        return float(low), float(high), "real grid used without clipping"
    return float(low), float(high), (
        "real grid clipped to A2 preset domain/instrument support: "
        f"real_range={(real_low, real_high)}, supported_range={(low, high)}, "
        f"domain_range={domain_range}, instrument_range={instrument_range}"
    )


def _finite_median_step(wavelengths: np.ndarray) -> float | None:
    diffs = np.diff(np.asarray(wavelengths, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _stable_dataset_seed(seed: int, dataset: RealDataset, purpose: str) -> int:
    key = "|".join([
        str(seed),
        purpose,
        dataset.source,
        dataset.task,
        dataset.database_name,
        dataset.dataset,
    ])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def _audited_mapping_reason(
    *,
    mapping_reason: str,
    generation_metadata: dict[str, Any],
    calibration_metadata: dict[str, Any],
    sanitation_audit: dict[str, Any] | None = None,
    remap_audit: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "mapping_reason": mapping_reason,
        "generation": generation_metadata,
        "calibration": calibration_metadata,
    }
    if sanitation_audit is not None:
        payload["sanitation"] = sanitation_audit
    if remap_audit is not None:
        payload["grid_remap"] = remap_audit
    return json.dumps(_to_builtin(payload), sort_keys=True, separators=(",", ":"))


def _covariance_disabled_metadata() -> dict[str, Any]:
    return {
        "enabled": False,
        "reason": "disabled_by_default_auc_regression",
        "method": "low_rank_pca_variance_match_with_orth_shrinkage_blend",
        "status": "disabled",
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "replays_real_rows": False,
        "imputed": False,
    }


def _merge_synthetic_sanitation_audits(
    *,
    post_generation: dict[str, Any],
    post_marginal: dict[str, Any],
) -> dict[str, Any]:
    generation_action = str(post_generation.get("action", "not_recorded"))
    marginal_action = str(post_marginal.get("action", "not_recorded"))
    return {
        "side": "synthetic",
        "action": f"post_generation:{generation_action}|post_marginal:{marginal_action}",
        "stages": {
            "post_generation": post_generation,
            "post_marginal": post_marginal,
        },
        "dropped_rows": int(post_generation.get("dropped_rows", 0) or 0)
        + int(post_marginal.get("dropped_rows", 0) or 0),
        "dropped_cols": int(post_generation.get("dropped_cols", 0) or 0)
        + int(post_marginal.get("dropped_cols", 0) or 0),
        "n_rows_after": int(post_marginal.get("n_rows_after", post_generation.get("n_rows_after", 0)) or 0),
        "n_cols_after": int(post_marginal.get("n_cols_after", post_generation.get("n_cols_after", 0)) or 0),
        "finite_policy": "drop_nonfinite_no_imputation",
        "thresholds_modified": False,
        "metrics_modified": False,
        "imputed": False,
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
    }


def grid_compatible_preset_fallback(
    *,
    original_preset: str,
    real_wavelengths: np.ndarray,
    available_presets: list[str],
    allow_cross_domain_fallback: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Pick an alternate A2 preset whose supported overlap covers the real grid.

    Numeric grids whose original preset has no >=3-point real overlap are
    remapped deterministically to the first alternative preset (in the supplied
    order) whose domain x instrument overlap can host >=3 real wavelengths.
    Index-fallback grids (parsed via ``np.arange`` because their CSV header is
    not numeric) are never remapped: their physical wavelength scale is
    unknown, so an alternative preset would not be more meaningful than the
    original.
    """
    is_index = is_index_fallback_grid(real_wavelengths)
    audit: dict[str, Any] = {
        "enabled": True,
        "original_preset": original_preset,
        "selected_preset": original_preset,
        "is_index_fallback": bool(is_index),
        "allow_cross_domain_fallback": bool(allow_cross_domain_fallback),
        "reason": "no_remap_needed",
        "tried_presets": [],
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "imputed": False,
        "replays_real_rows": False,
    }
    if is_index:
        audit["reason"] = "wavelength_grid_unknown"
        return original_preset, audit
    if _preset_has_real_grid_overlap(original_preset, real_wavelengths):
        return original_preset, audit
    audit["tried_presets"].append(original_preset)
    if not allow_cross_domain_fallback:
        audit["reason"] = "domain_wavelength_support"
        return original_preset, audit
    for candidate in available_presets:
        if candidate == original_preset:
            continue
        audit["tried_presets"].append(candidate)
        if _preset_has_real_grid_overlap(candidate, real_wavelengths):
            audit["selected_preset"] = candidate
            audit["reason"] = "grid_compatible_fallback"
            return candidate, audit
    audit["reason"] = "no_grid_compatible_alternative"
    return original_preset, audit


def _preset_has_real_grid_overlap(preset: str, real_wavelengths: np.ndarray) -> bool:
    requested = _real_grid_request(real_wavelengths)
    if not requested["valid"]:
        return False
    target_type, target_size = _preset_target(preset)
    source = _preset_source(
        preset,
        target_type=target_type,
        target_size=target_size,
        seed=0,
    )
    low, high, _ = _supported_real_grid_range(
        preset=preset,
        instrument=str(source["instrument"]),
        real_low=float(requested["wavelength_min"]),
        real_high=float(requested["wavelength_max"]),
        median_step=float(requested["median_step"]),
    )
    if low is None or high is None:
        return False
    real_arr = np.asarray(real_wavelengths, dtype=float)
    mask = (real_arr >= float(low)) & (real_arr <= float(high))
    return int(mask.sum()) >= 3


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _missing_paths_section(inventories: list[CohortInventory]) -> str:
    missing: list[str] = []
    for inventory in inventories:
        missing.extend(inventory.missing_paths)
    if not missing:
        return "No missing paths among `status == ok` cohort rows."
    return "\n".join(f"- `{path}`" for path in sorted(set(missing)))


def _mapping_strategy_section(rows: list[ScorecardRow]) -> str:
    authoritative_rows = [
        row for row in rows if row.comparison_space == "uncalibrated_raw"
    ]
    if not authoritative_rows:
        return (
            "No uncalibrated_raw rows were produced, so no dataset mapping summary is available."
        )
    strategy_counts: dict[str, int] = {}
    preset_counts: dict[str, int] = {}
    for row in authoritative_rows:
        strategy_counts[row.synthetic_mapping_strategy] = (
            strategy_counts.get(row.synthetic_mapping_strategy, 0) + 1
        )
        preset_counts[row.synthetic_preset] = preset_counts.get(row.synthetic_preset, 0) + 1
    lines = [
        "Deterministic matrix-first dataset mapping is used first; stable SHA-256 fallback is used only when no matrix rule matches. No dataset-index round-robin selection is used.",
        "Rules use only source/task/database/dataset identifiers, never y values, labels, or split contents.",
        "",
        "Strategy counts over uncalibrated_raw rows:",
    ]
    lines.extend(
        f"- `{strategy}`: {count}"
        for strategy, count in sorted(strategy_counts.items())
    )
    lines.append("")
    lines.append("Synthetic preset counts over uncalibrated_raw rows:")
    lines.extend(
        f"- `{preset}`: {count}"
        for preset, count in sorted(preset_counts.items())
    )
    return "\n".join(lines)


def _calibration_section(rows: list[ScorecardRow]) -> str:
    compared_rows = [row for row in rows if row.status == "compared"]
    if not compared_rows:
        return (
            "No calibration was applied because no real/synthetic comparison row "
            "was produced."
        )
    grid_strategies: dict[str, int] = {}
    cov_enabled_count = 0
    cov_disabled_count = 0
    cov_ranks: list[int] = []
    cov_shrinkage: set[float] = set()
    cov_blend: set[float] = set()
    warnings: set[str] = set()
    for row in compared_rows:
        try:
            reason = json.loads(row.synthetic_mapping_reason)
            calibration = reason.get("calibration", {}) or {}
        except (TypeError, ValueError, AttributeError):
            continue
        marginal = calibration.get("marginal", {}) or {}
        covariance = calibration.get("covariance", {}) or {}
        strategy = str(marginal.get("grid_strategy", "not_recorded"))
        grid_strategies[strategy] = grid_strategies.get(strategy, 0) + 1
        warning = marginal.get("warning")
        if warning:
            warnings.add(str(warning))
        cov_warning = covariance.get("warning")
        if cov_warning:
            warnings.add(str(cov_warning))
        if covariance.get("enabled"):
            cov_enabled_count += 1
            rank = covariance.get("rank")
            if isinstance(rank, int):
                cov_ranks.append(rank)
            shrinkage = covariance.get("shrinkage")
            if isinstance(shrinkage, (int, float)):
                cov_shrinkage.add(float(shrinkage))
            blend = covariance.get("blend")
            if isinstance(blend, (int, float)):
                cov_blend.add(float(blend))
        else:
            cov_disabled_count += 1

    lines = [
        "Strong provisional marginal calibration is applied to synthetic spectra after the authoritative `uncalibrated_raw` lane is scored, and only before the `calibrated_raw_diagnostic` and `snv` diagnostic lanes; the `uncalibrated_raw` gate is computed on uncalibrated synthetic spectra.",
        "Covariance calibration is disabled by default because the R5 covariance calibration worsened adversarial AUC.",
        "Fit inputs are limited to `real_X` and real wavelengths; apply inputs are limited to synthetic X and synthetic wavelengths, with real calibration interpolated to that grid when needed.",
        "No y/target/labels/splits or source oracle inputs are used for calibration; metadata is marked `oracle=false`, `label_inputs_used=false`, `target_inputs_used=false`, `split_inputs_used=false`, and `source_oracle_used=false`.",
        "The marginal calibration uses per-wavelength robust affine scaling, quantile mapping, and high-pass residual scaling.",
        "Covariance metadata is still emitted with `enabled=false` and `reason=disabled_by_default_auc_regression`.",
        "Thresholds are not changed and metric definitions are not weakened by calibration (`thresholds_modified=false`, `metrics_modified=false`).",
        "These calibrations are intentionally strong and provisional; they must not be interpreted as proof of downstream transfer.",
        "",
        "Marginal calibration grid strategy counts over compared rows:",
    ]
    if grid_strategies:
        lines.extend(
            f"- `{strategy}`: {count}"
            for strategy, count in sorted(grid_strategies.items())
        )
    else:
        lines.append("- `not_recorded`: metadata unavailable")
    lines.append("")
    lines.append("Covariance calibration coverage over compared rows:")
    lines.append(f"- enabled: {cov_enabled_count}")
    lines.append(f"- disabled: {cov_disabled_count}")
    if cov_ranks:
        lines.append(
            f"- rank min/median/max: {min(cov_ranks)}/{int(np.median(cov_ranks))}/{max(cov_ranks)}"
        )
    if cov_shrinkage:
        lines.append(
            f"- shrinkage values used: {sorted(cov_shrinkage)}"
        )
    if cov_blend:
        lines.append(
            f"- blend values used: {sorted(cov_blend)}"
        )
    if warnings:
        lines.append("")
        lines.extend(f"- Warning: {warning}" for warning in sorted(warnings))
    return "\n".join(lines)


def _sanitation_section(rows: list[ScorecardRow]) -> str:
    actions_real: dict[str, int] = {}
    actions_synthetic: dict[str, int] = {}
    dropped_rows_total = 0
    dropped_cols_total = 0
    audited_rows = 0
    for row in rows:
        try:
            reason = json.loads(row.synthetic_mapping_reason)
            sanitation = reason.get("sanitation", {}) or {}
        except (TypeError, ValueError, AttributeError):
            continue
        real_audit = sanitation.get("real", {}) or {}
        syn_audit = sanitation.get("synthetic", {}) or {}
        action_real = str(real_audit.get("action", "not_run"))
        action_syn = str(syn_audit.get("action", "not_run"))
        actions_real[action_real] = actions_real.get(action_real, 0) + 1
        actions_synthetic[action_syn] = actions_synthetic.get(action_syn, 0) + 1
        dropped_rows_total += int(real_audit.get("dropped_rows", 0) or 0)
        dropped_cols_total += int(real_audit.get("dropped_cols", 0) or 0)
        audited_rows += 1
    if audited_rows == 0:
        return "No sanitation audit was recorded for any row."
    lines = [
        "Audit-aware non-finite sanitation is applied before alignment, calibration, and scoring; synthetic spectra are sanitized both after generation and after marginal calibration.",
        "No imputation is used; rows or wavelength columns containing non-finite values are dropped, and the policy is recorded per side/stage.",
        "Sanitation requires `>=8` finite samples and `>=3` finite wavelengths; otherwise the row is blocked rather than scored.",
        "Sanitation never modifies thresholds or metric definitions (`thresholds_modified=false`, `metrics_modified=false`, `imputed=false`).",
        "",
        "Real-side sanitation actions:",
    ]
    lines.extend(
        f"- `{action}`: {count}" for action, count in sorted(actions_real.items())
    )
    lines.append("")
    lines.append("Synthetic-side sanitation actions:")
    lines.extend(
        f"- `{action}`: {count}" for action, count in sorted(actions_synthetic.items())
    )
    lines.append("")
    lines.append(f"Total real rows dropped across audited rows: {dropped_rows_total}")
    lines.append(f"Total real columns dropped across audited rows: {dropped_cols_total}")
    return "\n".join(lines)


def _audit_flags_section() -> str:
    flags = [
        "oracle=false",
        "label_inputs_used=false",
        "target_inputs_used=false",
        "split_inputs_used=false",
        "source_oracle_used=false",
        "replays_real_rows=false",
        "thresholds_modified=false",
        "metrics_modified=false",
        "imputed=false",
    ]
    lines = [
        "All sanitation, marginal calibration, covariance calibration, and grid remap steps record the following audit flags:",
        "",
    ]
    lines.extend(f"- `{flag}`" for flag in flags)
    return "\n".join(lines)


def _source_override_section(rows: list[ScorecardRow]) -> str:
    audited = 0
    enabled_count = 0
    wavelength_override_count = 0
    wavelength_applied_count = 0
    rule_counts: dict[str, int] = {}
    for row in rows:
        try:
            reason_payload = json.loads(row.synthetic_mapping_reason)
            source_overrides = (
                reason_payload.get("generation", {}).get("source_overrides", {}) or {}
            )
        except (TypeError, ValueError, AttributeError):
            continue
        if not source_overrides:
            continue
        audited += 1
        if source_overrides.get("enabled"):
            enabled_count += 1
        for rule in source_overrides.get("rules", []) or []:
            rule_name = str(rule)
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        overrides = source_overrides.get("overrides", {}) or {}
        support_override = overrides.get("_bench_wavelength_support_override")
        if isinstance(support_override, dict) and support_override.get("enabled") is True:
            wavelength_override_count += 1
        wavelength_policy = reason_payload.get("generation", {}).get(
            "canonical_wavelength_policy", {}
        ) or {}
        support_audit = wavelength_policy.get("bench_wavelength_support_override", {})
        if isinstance(support_audit, dict) and support_audit.get("applied") is True:
            wavelength_applied_count += 1

    if audited == 0:
        return "No source override audit was recorded for any row."

    lines = [
        "Bench source overrides are emitted only from the B2 scorecard path and are recorded under `generation.source_overrides`.",
        "Rules use dataset source/task/database/dataset tokens plus physical real wavelengths when needed; they do not read y values, labels, splits, targets, or performance metrics.",
        "The wavelength support override remains opt-in at canonicalization time and disabled by default outside explicit B2 on-demand generation.",
        "",
        f"Source override audits recorded: {audited}",
        f"Source override enabled rows: {enabled_count}",
        f"Wavelength support override requested rows: {wavelength_override_count}",
        f"Wavelength support override applied rows: {wavelength_applied_count}",
    ]
    if rule_counts:
        lines.append("")
        lines.append("Source override rule counts:")
        lines.extend(f"- `{rule}`: {count}" for rule, count in sorted(rule_counts.items()))
    return "\n".join(lines)


def _r9_gap_summary_section(raw_compared: list[ScorecardRow]) -> str:
    if not raw_compared:
        return (
            "R9 is a partial diagnostic improvement, not a B2 pass.\n"
            "No uncalibrated_raw comparison rows are available, so R9 gap status cannot be evaluated."
        )

    auc_failure_count = sum(
        1
        for row in raw_compared
        if (
            row.adversarial_auc is not None
            and row.adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
        )
    )
    pca_failure_count = sum(
        1
        for row in raw_compared
        if (
            row.pca_overlap is not None
            and row.pca_overlap < PROVISIONAL_THRESHOLDS["pca_overlap_min"]
        )
    )
    gap_prefixes = ("BEER/", "DIESEL/", "CORN/")
    gap_rows = [
        row for row in raw_compared
        if row.dataset.upper().startswith(gap_prefixes)
        and (
            row.provisional_decision != "provisional_pass"
            or (
                row.adversarial_auc is not None
                and row.adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
            )
            or (
                row.pca_overlap is not None
                and row.pca_overlap < PROVISIONAL_THRESHOLDS["pca_overlap_min"]
            )
        )
    ]

    lines = [
        "R9 is a partial diagnostic improvement, not a B2 pass.",
        f"Raw adversarial AUC gaps remain: {auc_failure_count}/{len(raw_compared)} compared rows.",
        f"Raw PCA overlap gaps remain: {pca_failure_count}/{len(raw_compared)} compared rows.",
        "BEER, DIESEL, and CORN rows remain named gaps when they appear below; no downstream transfer or realism-pass claim is made for them.",
        "",
        "Named persistent gaps over raw rows:",
    ]
    if not gap_rows:
        lines.append("- None among BEER/DIESEL/CORN in this run.")
        return "\n".join(lines)
    for row in sorted(gap_rows, key=lambda item: item.dataset):
        lines.append(
            f"- `{row.dataset}`: preset `{row.synthetic_preset}`, "
            f"adv AUC {_fmt(row.adversarial_auc)}, PCA overlap {_fmt(row.pca_overlap)}, "
            f"decision `{row.provisional_decision}`"
        )
    return "\n".join(lines)


def _grid_remap_section(rows: list[ScorecardRow]) -> str:
    audited = 0
    reasons: dict[str, int] = {}
    remapped: list[tuple[str, str, str]] = []
    index_fallback_count = 0
    for row in rows:
        try:
            reason_payload = json.loads(row.synthetic_mapping_reason)
            remap = reason_payload.get("grid_remap", {}) or {}
        except (TypeError, ValueError, AttributeError):
            continue
        if not remap:
            continue
        audited += 1
        reason = str(remap.get("reason", "not_recorded"))
        reasons[reason] = reasons.get(reason, 0) + 1
        if remap.get("is_index_fallback"):
            index_fallback_count += 1
        if reason == "grid_compatible_fallback":
            remapped.append((row.dataset, str(remap.get("original_preset")), str(remap.get("selected_preset"))))
    if audited == 0:
        return "No grid remap audit was recorded for any row."
    lines = [
        "Grid-compatible preset remapping is applied only when the original preset has fewer than three real-grid overlap points, the real wavelengths are numeric, and the dataset mapping came from stable hash fallback.",
        "Index-fallback grids (those parsed via `np.arange` because their CSV header is not numeric) are blocked with `wavelength_grid_unknown` because their physical wavelength scale is unknown.",
        "Semantic matrix-first matches with no supported physical wavelength overlap are blocked with `domain_wavelength_support` instead of being remapped cross-domain for wavelength compatibility.",
        "Every remap records `original_preset`, `selected_preset`, and a `reason` token; remapping never modifies thresholds or metric definitions.",
        "",
        "Remap reason counts:",
    ]
    lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(reasons.items()))
    lines.append(f"- index-fallback rows skipped: {index_fallback_count}")
    if remapped:
        lines.append("")
        lines.append("Datasets remapped to a grid-compatible preset:")
        seen: set[tuple[str, str, str]] = set()
        for entry in remapped:
            if entry in seen:
                continue
            seen.add(entry)
            lines.append(f"- `{entry[0]}`: `{entry[1]}` -> `{entry[2]}`")
    return "\n".join(lines)


def _load_failures_section(load_failures: list[dict[str, Any]]) -> str:
    if not load_failures:
        return "None."
    lines = []
    for failure in load_failures:
        lines.append(
            f"- `{failure['source']}` `{failure['dataset']}` "
            f"[{failure.get('failure_class', 'unknown')}]: {failure['reason']}"
        )
    return "\n".join(lines)


def _failure_class(exc: Exception) -> str:
    reason = str(exc)
    if "fewer than three overlapping points" in reason:
        return "wavelength_grid_overlap"
    if (
        "non-finite spectra" in reason
        or "spectra contain non-finite values" in reason
        or "non_finite_retention_below_threshold" in reason
    ):
        return "non_finite_spectra"
    if "feature mismatch" in reason:
        return "real_train_test_shape_mismatch"
    if "expected 2D spectra" in reason or "spectra need" in reason:
        return "invalid_spectra_shape"
    return exc.__class__.__name__


def _decision_text(*, compared_count: int, auc_failure_count: int, blocked_count: int) -> str:
    if compared_count > 0 and auc_failure_count == compared_count:
        return (
            "B2 scorecard route is runnable, but realism smoke success is not established: "
            "raw adversarial AUC failed for all raw compared rows, so synthetic spectra are trivially separable. "
            f"Blocked selected rows retained in CSV: {blocked_count}."
        )
    return (
        "B2 scorecard route is runnable and writes standardized markdown plus CSV metrics. "
        f"Blocked selected rows retained in CSV: {blocked_count}."
    )


def _git_status_summary(root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {
        "returncode": result.returncode,
        "line_count": len(lines),
        "lines": lines[:80],
        "truncated": len(lines) > 80,
    }


def _git_status_section(git_status: dict[str, Any]) -> str:
    if git_status["line_count"] == 0:
        return "Clean working tree."
    lines = [
        f"- `git status --short` lines: {git_status['line_count']}",
        "- First entries:",
    ]
    lines.extend(f"  - `{line}`" for line in git_status["lines"])
    if git_status.get("truncated"):
        lines.append("  - _truncated_")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    main()
