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
from exp00_smoke_prior_dataset import PRESETS, _preset_source
from nirsyntheticpfn.adapters.builder_adapter import build_synthetic_dataset_run
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_domain, canonicalize_prior_config
from nirsyntheticpfn.evaluation.prior_checks import PHASE_A_GATE_OVERRIDE
from nirsyntheticpfn.evaluation.realism import (
    PROVISIONAL_THRESHOLDS,
    CohortInventory,
    ComparisonSpace,
    RealDataset,
    ScorecardRow,
    apply_real_marginal_calibration,
    blocked_scorecard_row,
    compare_real_synthetic,
    discover_local_real_datasets,
    downsample_rows,
    fit_real_marginal_calibration,
    load_real_spectra,
    synthetic_only_row,
    write_scorecard_csv,
)

from nirs4all.synthesis.domains import get_domain_config
from nirs4all.synthesis.instruments import get_instrument_archetype

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

    for dataset in selected:
        preset, mapping_strategy, mapping_reason = select_synthetic_preset_for_dataset(dataset)
        try:
            real_X, real_wavelengths = load_real_spectra(dataset, root=root)
            real_X = downsample_rows(
                real_X,
                max_rows=max_real_samples,
                random_state=_stable_dataset_seed(seed, dataset, "real_downsample"),
            )
            synthetic_run, generation_metadata = build_on_demand_synthetic_run_for_dataset(
                dataset=dataset,
                preset=preset,
                real_wavelengths=real_wavelengths,
                n_samples=n_synthetic_samples,
                seed=seed,
            )
            synthetic_run_count += 1
            synthetic_X = downsample_rows(
                synthetic_run.X,
                max_rows=max_real_samples,
                random_state=_stable_dataset_seed(seed, dataset, "synthetic_downsample"),
            )
            calibration = fit_real_marginal_calibration(real_X, real_wavelengths)
            synthetic_X, calibration_metadata = apply_real_marginal_calibration(
                synthetic_X,
                synthetic_run.wavelengths,
                calibration,
            )
            audited_mapping_reason = _audited_mapping_reason(
                mapping_reason=mapping_reason,
                generation_metadata=generation_metadata,
                calibration_metadata=calibration_metadata,
            )
            comparison_spaces: tuple[ComparisonSpace, ...] = ("raw", "snv")
            for comparison_space in comparison_spaces:
                rows.append(compare_real_synthetic(
                    real_X=real_X,
                    real_wavelengths=real_wavelengths,
                    synthetic_X=synthetic_X,
                    synthetic_wavelengths=synthetic_run.wavelengths,
                    dataset=f"{dataset.database_name}/{dataset.dataset}",
                    source=dataset.source,
                    task=dataset.task,
                    synthetic_preset=preset,
                    comparison_space=comparison_space,
                    synthetic_mapping_strategy=mapping_strategy,
                    synthetic_mapping_reason=audited_mapping_reason,
                    random_state=_stable_dataset_seed(seed, dataset, f"metrics:{comparison_space}"),
                ))
        except Exception as exc:
            dataset_name = f"{dataset.database_name}/{dataset.dataset}"
            failure_class = _failure_class(exc)
            rows.append(blocked_scorecard_row(
                source=dataset.source,
                task=dataset.task,
                dataset=dataset_name,
                synthetic_preset=preset,
                blocked_reason=f"{failure_class}: {exc}",
                comparison_space="raw",
                synthetic_mapping_strategy=mapping_strategy,
                synthetic_mapping_reason=mapping_reason,
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
        1 for row in rows if row.status == "compared" and row.comparison_space == "raw"
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
    raw_compared = [
        row for row in compared
        if row.comparison_space == "raw"
    ]
    snv_compared = [
        row for row in compared
        if row.comparison_space == "snv"
    ]
    blocked_rows = [row for row in rows if row.status == "blocked"]
    blocked_for_evidence = result["status"] != "done"
    auc_failures = [
        row for row in raw_compared
        if (
            row.adversarial_auc is not None
            and row.adversarial_auc > PROVISIONAL_THRESHOLDS["adversarial_auc_smoke"]
        )
    ]
    pca_failures = [
        row for row in raw_compared
        if (
            row.pca_overlap is not None
            and row.pca_overlap < PROVISIONAL_THRESHOLDS["pca_overlap_min"]
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
        "- Comparison spaces: raw, snv",
        "- Thresholds are provisional smoke thresholds, not calibrated domain gates.",
        "- Primary decisions and gates use only `comparison_space == \"raw\"`; SNV is an additional diagnostic.",
        "- Synthetic spectra are calibrated before scoring; calibration does not change metric definitions or thresholds.",
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
        f"- Raw real/synthetic comparison rows written: {len(raw_compared)}",
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

    lines.extend([
        "## Diagnostic Outcome",
        "",
        (
            "No raw real/synthetic diagnostic outcome is available because no raw comparison was produced."
            if blocked_for_evidence
            else (
                f"- Raw adversarial AUC smoke failures: {len(auc_failures)}/{len(raw_compared)} compared rows."
            )
        ),
    ])
    if not blocked_for_evidence:
        lines.extend([
            f"- Raw PCA overlap smoke failures: {len(pca_failures)}/{len(raw_compared)} compared rows.",
            "- Raw gate remains visible: these scorecards do not claim realism success when raw adversarial AUC exceeds the smoke threshold.",
            "",
            "## SNV Diagnostic",
            "",
            f"- SNV adversarial AUC smoke failures: {len(snv_auc_failures)}/{len(snv_compared)} diagnostic rows.",
            "- SNV is applied only after wavelength-grid alignment and only on comparison copies.",
            "- SNV diagnostics do not override raw decisions and must not be interpreted as B2 success when raw fails.",
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
                compared_count=len(raw_compared),
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
    """Choose a deterministic synthetic preset from dataset tokens, with stable hash fallback."""
    if not synthetic_runs:
        raise ValueError("synthetic_runs must not be empty")

    available = dict(synthetic_runs)
    text, tokens = _dataset_text_and_tokens(dataset)
    rules = [
        ("dairy", ("milk", "dairy", "cheese")),
        ("meat", ("beef", "meat", "marbling", "impurity")),
        ("tablets", ("tablet", "pharma", "escitalopram")),
        ("fuel", ("diesel", "fuel")),
        ("oilseeds", ("colza", "oilseed", "oil")),
        ("baking", ("beer", "biscuit", "baking")),
        ("grain", ("corn", "rice", "wheat", "barley", "cereal", "amylose", "grain")),
        ("fruit", ("berry", "fruit", "peach", "plum", "grape", "grapevine", "cassava", "pistacia")),
        ("powders", ("lucas", "soil", "phosphorus", "quartz", "incombustible", "powder")),
    ]
    for preset, keywords in rules:
        matched = next((keyword for keyword in keywords if _contains_keyword(text, tokens, keyword)), None)
        if matched is None:
            continue
        if preset in available:
            return (
                preset,
                available[preset],
                "dataset_aware_token",
                f"matched token '{matched}' in {dataset.database_name}/{dataset.dataset} -> {preset}",
            )
        return _stable_hash_mapping(
            dataset,
            synthetic_runs,
            reason_prefix=f"matched token '{matched}' -> unavailable preset '{preset}'",
        )
    return _stable_hash_mapping(
        dataset,
        synthetic_runs,
        reason_prefix="no dataset-aware token matched",
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
    source, grid_metadata = synthetic_source_for_real_grid(
        preset=preset,
        target_type=target_type,
        target_size=target_size,
        seed=run_seed,
        real_wavelengths=real_wavelengths,
    )
    record = canonicalize_prior_config(source)
    run = build_synthetic_dataset_run(record, n_samples=n_samples, random_seed=run_seed)
    generation_metadata = {
        "preset": preset,
        "strategy": "on_demand_per_real_dataset",
        "dataset_key": dataset.key,
        "seed": int(run_seed),
        "grid": grid_metadata,
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


def _contains_keyword(text: str, tokens: set[str], keyword: str) -> bool:
    if keyword == "oil":
        return keyword in tokens
    return keyword in tokens or keyword in text


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
) -> tuple[float | None, float | None, str]:
    domain_range = get_domain_config(canonicalize_domain(preset)).wavelength_range
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
) -> str:
    return json.dumps(
        _to_builtin({
            "mapping_reason": mapping_reason,
            "generation": generation_metadata,
            "calibration": calibration_metadata,
        }),
        sort_keys=True,
        separators=(",", ":"),
    )


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
    raw_rows = [row for row in rows if row.comparison_space == "raw"]
    if not raw_rows:
        return "No raw rows were produced, so no dataset mapping summary is available."
    strategy_counts: dict[str, int] = {}
    preset_counts: dict[str, int] = {}
    for row in raw_rows:
        strategy_counts[row.synthetic_mapping_strategy] = (
            strategy_counts.get(row.synthetic_mapping_strategy, 0) + 1
        )
        preset_counts[row.synthetic_preset] = preset_counts.get(row.synthetic_preset, 0) + 1
    lines = [
        "Deterministic dataset-aware token mapping is used first; stable SHA-256 fallback is used only when no token rule matches. No dataset-index round-robin selection is used.",
        "",
        "Strategy counts over raw rows:",
    ]
    lines.extend(
        f"- `{strategy}`: {count}"
        for strategy, count in sorted(strategy_counts.items())
    )
    lines.append("")
    lines.append("Synthetic preset counts over raw rows:")
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
    strategies: dict[str, int] = {}
    warnings: set[str] = set()
    for row in compared_rows:
        try:
            reason = json.loads(row.synthetic_mapping_reason)
            calibration = reason.get("calibration", {})
        except (TypeError, ValueError, AttributeError):
            continue
        strategy = str(calibration.get("grid_strategy", "not_recorded"))
        strategies[strategy] = strategies.get(strategy, 0) + 1
        warning = calibration.get("warning")
        if warning:
            warnings.add(str(warning))

    lines = [
        "Strong provisional marginal calibration is applied to synthetic spectra before raw and SNV scoring.",
        "Fit inputs are limited to `real_X` and real wavelengths; apply inputs are limited to synthetic X and synthetic wavelengths.",
        "No y/target/labels/splits or source oracle inputs are used for calibration metadata marked `oracle=false`, `label_inputs_used=false`, `target_inputs_used=false`, `split_inputs_used=false`, and `source_oracle_used=false`.",
        "The calibration uses per-wavelength robust affine scaling, quantile mapping, and high-pass residual scaling; it is intentionally strong and provisional.",
        "Thresholds are not changed and metric definitions are not weakened by calibration (`thresholds_modified=false`, `metrics_modified=false`).",
        "Quantile mapping is column-wise and metadata records `replays_real_rows=false`; it must not be interpreted as proof of downstream transfer.",
        "",
        "Calibration grid strategy counts over compared rows:",
    ]
    if strategies:
        lines.extend(
            f"- `{strategy}`: {count}"
            for strategy, count in sorted(strategies.items())
        )
    else:
        lines.append("- `not_recorded`: metadata unavailable")
    if warnings:
        lines.append("")
        lines.extend(f"- Warning: {warning}" for warning in sorted(warnings))
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
    if "non-finite spectra" in reason:
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
