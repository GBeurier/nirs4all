"""exp33 panel x-realism adversarial discriminator (multi-dataset runner).

Bench-only runner for the X-realism strategy. Walks
``bench/tabpfn_paper/data`` (or any directory passed via ``--root``),
locates every leaf directory containing an ``Xtrain.csv``, and runs the
single-dataset evaluator from
``exp32_hybrid_xrealism_discriminator.py`` on each one. Aggregates the
best AUC (closest to 0.5) per dataset and emits a panel-wide report.

The same strict scope as exp32 applies: no Y file is read, no
``nirs4all/`` import, no targets/labels/splits-as-oracle, no global
cross-dataset score is interpreted as a content-realism claim.
Adversarial AUC IS the tuning oracle.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any

EXP33_AUDIT_SCOPE = "bench_only_phase_r0_panel_xrealism_adversarial_discriminator"
DEFAULT_ROOT = Path("bench/tabpfn_paper/data")
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/xrealism_panel.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/xrealism_panel.csv")
DEFAULT_PCA_RANGE: tuple[int, ...] = (0, 5, 20, 80)


def _load_exp32() -> ModuleType:
    name = "exp32_hybrid_xrealism_discriminator"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PanelRow:
    status: str
    relative_path: str
    n_real: int
    n_features: int
    axis_min: float | None
    axis_max: float | None
    best_rf_n_pca_components_requested: int | None
    best_rf_auc_mean: float | None
    best_rf_auc_std: float | None
    best_lr_n_pca_components_requested: int | None
    best_lr_auc_mean: float | None
    best_lr_auc_std: float | None
    elapsed_seconds: float
    error_message: str
    audit_scope: str = EXP33_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_leaf_dataset_dirs(root: Path) -> list[Path]:
    leaves: list[Path] = []
    for path in sorted(root.rglob("Xtrain.csv")):
        leaves.append(path.parent)
    return leaves


def _evaluate_one(
    exp32: ModuleType,
    directory: Path,
    *,
    root: Path,
    pca_range: tuple[int, ...],
    n_synthetic_factor: float,
    n_splits: int,
    test_size: float,
    seed: int,
    baseline_degree: int,
    max_peaks: int,
    n_estimators: int,
    classifiers: tuple[str, ...],
    score_sampling_mode: str = "gaussian",
    noise_sampling_mode: str = "gaussian",
    empirical_jitter_fraction: float = 0.05,
    score_gmm_components: int = 8,
    score_gmm_covariance_type: str = "full",
    score_knn_mixup_k: int = 5,
    score_knn_mixup_dirichlet_alpha: float = 1.0,
    multiplicative_scattering_degree: int = 0,
    additive_baseline_shift_std: float = 0.0,
    subsample_rows: int | None = None,
) -> PanelRow:
    relative = directory.relative_to(root) if directory.is_relative_to(root) else directory
    started = time.perf_counter()
    try:
        result = exp32.evaluate_dataset(
            directory,
            pca_range=pca_range,
            n_synthetic_factor=n_synthetic_factor,
            n_splits=n_splits,
            test_size=test_size,
            seed=seed,
            baseline_degree=baseline_degree,
            max_peaks=max_peaks,
            n_estimators=n_estimators,
            classifiers=classifiers,
            score_sampling_mode=score_sampling_mode,
            noise_sampling_mode=noise_sampling_mode,
            empirical_jitter_fraction=empirical_jitter_fraction,
            score_gmm_components=score_gmm_components,
            score_gmm_covariance_type=score_gmm_covariance_type,
            score_knn_mixup_k=score_knn_mixup_k,
            score_knn_mixup_dirichlet_alpha=score_knn_mixup_dirichlet_alpha,
            multiplicative_scattering_degree=multiplicative_scattering_degree,
            additive_baseline_shift_std=additive_baseline_shift_std,
            subsample_rows=subsample_rows,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return PanelRow(
            status="error",
            relative_path=str(relative),
            n_real=0,
            n_features=0,
            axis_min=None,
            axis_max=None,
            best_rf_n_pca_components_requested=None,
            best_rf_auc_mean=None,
            best_rf_auc_std=None,
            best_lr_n_pca_components_requested=None,
            best_lr_auc_mean=None,
            best_lr_auc_std=None,
            elapsed_seconds=elapsed,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    elapsed = time.perf_counter() - started
    best_rf = result.get("best_rf")
    best_lr = result.get("best_lr")
    return PanelRow(
        status="ok",
        relative_path=str(relative),
        n_real=int(result["n_real"]),
        n_features=int(result["n_features"]),
        axis_min=float(result["axis_min"]),
        axis_max=float(result["axis_max"]),
        best_rf_n_pca_components_requested=int(best_rf.n_pca_components_requested) if best_rf is not None else None,
        best_rf_auc_mean=float(best_rf.auc_rf_mean) if best_rf is not None else None,
        best_rf_auc_std=float(best_rf.auc_rf_std) if best_rf is not None else None,
        best_lr_n_pca_components_requested=int(best_lr.n_pca_components_requested) if best_lr is not None else None,
        best_lr_auc_mean=float(best_lr.auc_lr_mean) if best_lr is not None else None,
        best_lr_auc_std=float(best_lr.auc_lr_std) if best_lr is not None else None,
        elapsed_seconds=elapsed,
        error_message="",
    )


def run_panel(
    root: Path,
    *,
    pca_range: tuple[int, ...] = DEFAULT_PCA_RANGE,
    n_synthetic_factor: float = 1.0,
    n_splits: int = 3,
    test_size: float = 0.3,
    seed: int = 20260501,
    baseline_degree: int = 3,
    max_peaks: int = 16,
    n_estimators: int = 100,
    classifiers: tuple[str, ...] = ("rf", "lr"),
    max_datasets: int | None = None,
    skip_under_n_samples: int = 20,
    skip_over_n_features: int | None = None,
    score_sampling_mode: str = "gaussian",
    noise_sampling_mode: str = "gaussian",
    empirical_jitter_fraction: float = 0.05,
    score_gmm_components: int = 8,
    score_gmm_covariance_type: str = "full",
    score_knn_mixup_k: int = 5,
    score_knn_mixup_dirichlet_alpha: float = 1.0,
    multiplicative_scattering_degree: int = 0,
    additive_baseline_shift_std: float = 0.0,
    subsample_rows: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    exp32 = _load_exp32()
    leaves = _iter_leaf_dataset_dirs(root)
    if max_datasets is not None:
        leaves = leaves[:max_datasets]

    rows: list[PanelRow] = []
    for index, directory in enumerate(leaves, start=1):
        if progress:
            print(f"[{index}/{len(leaves)}] {directory.relative_to(root) if directory.is_relative_to(root) else directory}", flush=True)
        try:
            x_train_path = directory / "Xtrain.csv"
            with x_train_path.open("r", encoding="utf-8", errors="ignore") as handle:
                header = handle.readline()
                row_count = sum(1 for line in handle if line.strip())
            sep = exp32._detect_separator(header)
            n_features_estimate = sum(1 for tok in header.rstrip("\n").split(sep) if tok.strip())
        except Exception as exc:
            rows.append(
                PanelRow(
                    status="error",
                    relative_path=str(directory.relative_to(root) if directory.is_relative_to(root) else directory),
                    n_real=0,
                    n_features=0,
                    axis_min=None,
                    axis_max=None,
                    best_rf_n_pca_components_requested=None,
                    best_rf_auc_mean=None,
                    best_rf_auc_std=None,
                    best_lr_n_pca_components_requested=None,
                    best_lr_auc_mean=None,
                    best_lr_auc_std=None,
                    elapsed_seconds=0.0,
                    error_message=f"header_read_failed: {type(exc).__name__}: {exc}",
                )
            )
            continue

        if row_count < skip_under_n_samples:
            rows.append(
                PanelRow(
                    status="skipped_too_few_samples",
                    relative_path=str(directory.relative_to(root) if directory.is_relative_to(root) else directory),
                    n_real=row_count,
                    n_features=n_features_estimate,
                    axis_min=None,
                    axis_max=None,
                    best_rf_n_pca_components_requested=None,
                    best_rf_auc_mean=None,
                    best_rf_auc_std=None,
                    best_lr_n_pca_components_requested=None,
                    best_lr_auc_mean=None,
                    best_lr_auc_std=None,
                    elapsed_seconds=0.0,
                    error_message=f"row_count={row_count} < skip_under_n_samples={skip_under_n_samples}",
                )
            )
            continue

        if skip_over_n_features is not None and n_features_estimate > skip_over_n_features:
            rows.append(
                PanelRow(
                    status="skipped_too_many_features",
                    relative_path=str(directory.relative_to(root) if directory.is_relative_to(root) else directory),
                    n_real=row_count,
                    n_features=n_features_estimate,
                    axis_min=None,
                    axis_max=None,
                    best_rf_n_pca_components_requested=None,
                    best_rf_auc_mean=None,
                    best_rf_auc_std=None,
                    best_lr_n_pca_components_requested=None,
                    best_lr_auc_mean=None,
                    best_lr_auc_std=None,
                    elapsed_seconds=0.0,
                    error_message=f"n_features={n_features_estimate} > skip_over_n_features={skip_over_n_features}",
                )
            )
            continue

        rows.append(
            _evaluate_one(
                exp32,
                directory,
                root=root,
                pca_range=pca_range,
                n_synthetic_factor=n_synthetic_factor,
                n_splits=n_splits,
                test_size=test_size,
                seed=seed,
                baseline_degree=baseline_degree,
                max_peaks=max_peaks,
                n_estimators=n_estimators,
                classifiers=classifiers,
                score_sampling_mode=score_sampling_mode,
                noise_sampling_mode=noise_sampling_mode,
                empirical_jitter_fraction=empirical_jitter_fraction,
                score_gmm_components=score_gmm_components,
                score_gmm_covariance_type=score_gmm_covariance_type,
                score_knn_mixup_k=score_knn_mixup_k,
                score_knn_mixup_dirichlet_alpha=score_knn_mixup_dirichlet_alpha,
                multiplicative_scattering_degree=multiplicative_scattering_degree,
                additive_baseline_shift_std=additive_baseline_shift_std,
                subsample_rows=subsample_rows,
            )
        )

    return {
        "status": "done",
        "root": str(root),
        "n_total_datasets_found": len(leaves),
        "rows": rows,
    }


def write_csv(rows: list[PanelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(PanelRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None) -> str:
    rows: list[PanelRow] = list(result["rows"])
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"
    ok_rows = [row for row in rows if row.status == "ok"]
    error_rows = [row for row in rows if row.status == "error"]
    skipped_rows = [row for row in rows if row.status.startswith("skipped")]

    lines: list[str] = [
        "# exp33 Panel X-Realism Adversarial Discriminator",
        "",
        f"- audit_scope: `{EXP33_AUDIT_SCOPE}`",
        f"- root: `{result['root']}`",
        f"- report: `{report_path}`",
        csv_line,
        f"- n_total_datasets_found: `{result['n_total_datasets_found']}`",
        f"- ok_rows: `{len(ok_rows)}`",
        f"- skipped_rows: `{len(skipped_rows)}`",
        f"- error_rows: `{len(error_rows)}`",
        "",
        "## Goal",
        "",
        "- Each leaf dataset is evaluated independently with the exp32 hybrid generator and adversarial AUC harness.",
        "- Win condition per dataset: RandomForest AUC near 0.5 across CV splits.",
        "- Forbidden inputs: no labels, no targets, no splits-as-oracle, no downstream metrics, no transfer scores. Adversarial AUC IS the tuning oracle (per `docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`).",
        "- No `nirs4all/` import; no Y file is read.",
        "- This panel report does not aggregate AUCs into a single global score; per-dataset AUCs stand alone.",
        "",
        "## Per-Dataset Results",
        "",
        "| dataset | n_real | n_features | best RF AUC (n_pca) | best LR AUC (n_pca) | elapsed (s) | status |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        rf_label = (
            f"`{row.best_rf_auc_mean:.4f} +/- {row.best_rf_auc_std:.4f}` "
            f"(n_pca=`{row.best_rf_n_pca_components_requested}`)"
            if row.best_rf_auc_mean is not None
            else "n/a"
        )
        lr_label = (
            f"`{row.best_lr_auc_mean:.4f} +/- {row.best_lr_auc_std:.4f}` "
            f"(n_pca=`{row.best_lr_n_pca_components_requested}`)"
            if row.best_lr_auc_mean is not None
            else "n/a"
        )
        lines.append(
            f"| `{row.relative_path}` | `{row.n_real}` | `{row.n_features}` | {rf_label} | {lr_label} | "
            f"`{row.elapsed_seconds:.2f}` | `{row.status}` |"
        )

    if error_rows:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                "| dataset | error |",
                "|---|---|",
            ]
        )
        for row in error_rows:
            safe_msg = row.error_message.replace("|", "/")
            lines.append(f"| `{row.relative_path}` | `{safe_msg}` |")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp33_panel_xrealism_discriminator.py \\",
            f"  --root {result['root']} \\",
            f"  --report {report_path} \\",
            f"  --csv {csv_path if csv_path is not None else 'bench/nirs_synthetic_pfn/reports/xrealism_panel.csv'}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_pca_range(spec: str) -> tuple[int, ...]:
    if not spec:
        return DEFAULT_PCA_RANGE
    return tuple(int(token) for token in spec.split(",") if token.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pca-range", type=str, default="")
    parser.add_argument("--n-synthetic-factor", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--baseline-degree", type=int, default=3)
    parser.add_argument("--max-peaks", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--classifiers", type=str, default="rf,lr")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--skip-under-n-samples", type=int, default=20)
    parser.add_argument("--skip-over-n-features", type=int, default=None)
    parser.add_argument("--score-sampling-mode", type=str, default="gaussian", choices=("gaussian", "empirical", "joint_bootstrap", "gmm", "copula", "knn_mixup"))
    parser.add_argument("--noise-sampling-mode", type=str, default="gaussian", choices=("gaussian", "empirical", "joint_bootstrap"))
    parser.add_argument("--empirical-jitter-fraction", type=float, default=0.05)
    parser.add_argument("--score-gmm-components", type=int, default=8)
    parser.add_argument("--score-gmm-covariance-type", type=str, default="full", choices=("full", "tied", "diag", "spherical"))
    parser.add_argument("--score-knn-mixup-k", type=int, default=5)
    parser.add_argument("--score-knn-mixup-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--multiplicative-scattering-degree", type=int, default=0)
    parser.add_argument("--additive-baseline-shift-std", type=float, default=0.0)
    parser.add_argument("--subsample-rows", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    pca_range = _parse_pca_range(args.pca_range)
    classifiers = tuple(token.strip() for token in args.classifiers.split(",") if token.strip())

    result = run_panel(
        args.root,
        pca_range=pca_range,
        n_synthetic_factor=args.n_synthetic_factor,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        baseline_degree=args.baseline_degree,
        max_peaks=args.max_peaks,
        n_estimators=args.n_estimators,
        classifiers=classifiers,
        max_datasets=args.max_datasets,
        skip_under_n_samples=args.skip_under_n_samples,
        skip_over_n_features=args.skip_over_n_features,
        score_sampling_mode=args.score_sampling_mode,
        noise_sampling_mode=args.noise_sampling_mode,
        empirical_jitter_fraction=args.empirical_jitter_fraction,
        score_gmm_components=args.score_gmm_components,
        score_gmm_covariance_type=args.score_gmm_covariance_type,
        score_knn_mixup_k=args.score_knn_mixup_k,
        score_knn_mixup_dirichlet_alpha=args.score_knn_mixup_dirichlet_alpha,
        multiplicative_scattering_degree=args.multiplicative_scattering_degree,
        additive_baseline_shift_std=args.additive_baseline_shift_std,
        subsample_rows=args.subsample_rows,
        progress=not args.no_progress,
    )
    if args.csv is not None:
        write_csv(list(result["rows"]), args.csv)
    markdown = render_markdown(result, report_path=args.report, csv_path=args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.report}")
    if args.csv is not None:
        print(f"wrote {args.csv}")
    summary = {
        "n_total_datasets_found": result["n_total_datasets_found"],
        "ok": sum(1 for row in result["rows"] if row.status == "ok"),
        "skipped": sum(1 for row in result["rows"] if row.status.startswith("skipped")),
        "errors": sum(1 for row in result["rows"] if row.status == "error"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
