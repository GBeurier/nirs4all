"""exp34 mode comparison runner (X-realism iteration helper).

Side-by-side comparison of generator configurations for a single
dataset. Built on the exp32 hybrid generator + adversarial AUC
harness. Used to iterate towards the win condition (RandomForest AUC
near 0.5 across CV splits).

Each variant is a named, fully-specified hybrid generator config. Runs
all variants at the same PCA rank, with the same discriminator splits
and seed, then dumps a Markdown comparison and a CSV row per variant.

No targets, no labels, no splits-as-oracle, no `nirs4all/` import.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any

EXP34_AUDIT_SCOPE = "bench_only_phase_r0_xrealism_mode_comparison"
DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/xrealism_mode_comparison.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/xrealism_mode_comparison.csv")


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
class Variant:
    name: str
    score_sampling_mode: str = "gaussian"
    noise_sampling_mode: str = "gaussian"
    empirical_jitter_fraction: float = 0.05
    score_gmm_components: int = 8
    score_gmm_covariance_type: str = "full"
    score_knn_mixup_k: int = 5
    score_knn_mixup_dirichlet_alpha: float = 1.0
    multiplicative_scattering_degree: int = 0
    additive_baseline_shift_std: float = 0.0


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant("v0_gaussian"),
    Variant("v2_gmm10_diag", score_sampling_mode="gmm", noise_sampling_mode="gaussian", score_gmm_components=10, score_gmm_covariance_type="diag"),
    Variant("v3_gmm10_diag_noise_jb", score_sampling_mode="gmm", noise_sampling_mode="joint_bootstrap", score_gmm_components=10, score_gmm_covariance_type="diag"),
    Variant("v4_copula_noise_jb", score_sampling_mode="copula", noise_sampling_mode="joint_bootstrap"),
    Variant("v5_joint_bootstrap_05", score_sampling_mode="joint_bootstrap", noise_sampling_mode="joint_bootstrap", empirical_jitter_fraction=0.05),
    Variant("v6_knn_mixup_k5_alpha1", score_sampling_mode="knn_mixup", noise_sampling_mode="joint_bootstrap", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v6_knn_mixup_k10_alpha2", score_sampling_mode="knn_mixup", noise_sampling_mode="joint_bootstrap", score_knn_mixup_k=10, score_knn_mixup_dirichlet_alpha=2.0),
)

EXTENDED_VARIANTS: tuple[Variant, ...] = (
    Variant("v0_gaussian"),
    Variant("v1_empirical_marginal", score_sampling_mode="empirical", noise_sampling_mode="empirical", empirical_jitter_fraction=0.05),
    Variant("v2_gmm10_diag", score_sampling_mode="gmm", noise_sampling_mode="gaussian", score_gmm_components=10, score_gmm_covariance_type="diag"),
    Variant("v2_gmm20_diag", score_sampling_mode="gmm", noise_sampling_mode="gaussian", score_gmm_components=20, score_gmm_covariance_type="diag"),
    Variant("v2_gmm10_full", score_sampling_mode="gmm", noise_sampling_mode="gaussian", score_gmm_components=10, score_gmm_covariance_type="full"),
    Variant("v3_gmm10_diag_noise_jb", score_sampling_mode="gmm", noise_sampling_mode="joint_bootstrap", score_gmm_components=10, score_gmm_covariance_type="diag"),
    Variant("v3_gmm10_diag_scatter2", score_sampling_mode="gmm", noise_sampling_mode="gaussian", score_gmm_components=10, score_gmm_covariance_type="diag", multiplicative_scattering_degree=2),
    Variant("v4_copula_noise_jb", score_sampling_mode="copula", noise_sampling_mode="joint_bootstrap"),
    Variant("v5_joint_bootstrap_05", score_sampling_mode="joint_bootstrap", noise_sampling_mode="joint_bootstrap", empirical_jitter_fraction=0.05),
    Variant("v5_joint_bootstrap_15", score_sampling_mode="joint_bootstrap", noise_sampling_mode="joint_bootstrap", empirical_jitter_fraction=0.15),
    Variant("v6_knn_mixup_k5_alpha1", score_sampling_mode="knn_mixup", noise_sampling_mode="joint_bootstrap", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v6_knn_mixup_k10_alpha2", score_sampling_mode="knn_mixup", noise_sampling_mode="joint_bootstrap", score_knn_mixup_k=10, score_knn_mixup_dirichlet_alpha=2.0),
    Variant("v6_knn_mixup_k20_alpha05", score_sampling_mode="knn_mixup", noise_sampling_mode="joint_bootstrap", score_knn_mixup_k=20, score_knn_mixup_dirichlet_alpha=0.5),
    Variant("v7_knn_mixup_k3_alpha1_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=3, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v7_knn_mixup_k5_alpha1_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v7_knn_mixup_k5_alpha05_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=0.5),
    Variant("v7_knn_mixup_k7_alpha05_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=7, score_knn_mixup_dirichlet_alpha=0.5),
)

KNN_VARIANTS: tuple[Variant, ...] = (
    Variant("v0_gaussian"),
    Variant("v6_knn_mixup_k5_alpha1_jb_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="joint_bootstrap", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v7_knn_mixup_k3_alpha1_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=3, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v7_knn_mixup_k5_alpha1_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=1.0),
    Variant("v7_knn_mixup_k5_alpha05_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=0.5),
    Variant("v7_knn_mixup_k7_alpha05_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=7, score_knn_mixup_dirichlet_alpha=0.5),
    Variant("v7_knn_mixup_k5_alpha2_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=5, score_knn_mixup_dirichlet_alpha=2.0),
    Variant("v7_knn_mixup_k10_alpha05_gauss_noise", score_sampling_mode="knn_mixup", noise_sampling_mode="gaussian", score_knn_mixup_k=10, score_knn_mixup_dirichlet_alpha=0.5),
)


@dataclass(frozen=True)
class ComparisonRow:
    dataset: str
    variant_name: str
    n_pca_components_requested: int
    n_pca_components_actual: int
    n_real: int
    n_synthetic: int
    n_features: int
    auc_rf_mean: float
    auc_rf_std: float
    auc_lr_mean: float
    auc_lr_std: float
    score_sampling_mode: str
    noise_sampling_mode: str
    empirical_jitter_fraction: float
    score_gmm_components: int
    multiplicative_scattering_degree: int
    additive_baseline_shift_std: float
    audit_scope: str = EXP34_AUDIT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_comparison(
    directory: Path,
    *,
    variants: tuple[Variant, ...] = DEFAULT_VARIANTS,
    n_pca_components: int = 80,
    n_synthetic_factor: float = 0.5,
    n_splits: int = 2,
    test_size: float = 0.3,
    seed: int = 20260501,
    baseline_degree: int = 3,
    max_peaks: int = 16,
    n_estimators: int = 50,
    classifiers: tuple[str, ...] = ("rf", "lr"),
    subsample_rows: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    exp32 = _load_exp32()
    rows: list[ComparisonRow] = []

    for index, variant in enumerate(variants, start=1):
        if progress:
            print(f"[{index}/{len(variants)}] running variant {variant.name}", flush=True)
        result = exp32.evaluate_dataset(
            directory,
            pca_range=(n_pca_components,),
            n_synthetic_factor=n_synthetic_factor,
            n_splits=n_splits,
            test_size=test_size,
            seed=seed,
            baseline_degree=baseline_degree,
            max_peaks=max_peaks,
            n_estimators=n_estimators,
            classifiers=classifiers,
            score_sampling_mode=variant.score_sampling_mode,
            noise_sampling_mode=variant.noise_sampling_mode,
            empirical_jitter_fraction=variant.empirical_jitter_fraction,
            score_gmm_components=variant.score_gmm_components,
            score_gmm_covariance_type=variant.score_gmm_covariance_type,
            score_knn_mixup_k=variant.score_knn_mixup_k,
            score_knn_mixup_dirichlet_alpha=variant.score_knn_mixup_dirichlet_alpha,
            multiplicative_scattering_degree=variant.multiplicative_scattering_degree,
            additive_baseline_shift_std=variant.additive_baseline_shift_std,
            subsample_rows=subsample_rows,
        )
        eval_row = list(result["rows"])[0]
        rows.append(
            ComparisonRow(
                dataset=str(directory),
                variant_name=variant.name,
                n_pca_components_requested=int(eval_row.n_pca_components_requested),
                n_pca_components_actual=int(eval_row.n_pca_components_actual),
                n_real=int(eval_row.n_real),
                n_synthetic=int(eval_row.n_synthetic),
                n_features=int(eval_row.n_features),
                auc_rf_mean=float(eval_row.auc_rf_mean),
                auc_rf_std=float(eval_row.auc_rf_std),
                auc_lr_mean=float(eval_row.auc_lr_mean),
                auc_lr_std=float(eval_row.auc_lr_std),
                score_sampling_mode=variant.score_sampling_mode,
                noise_sampling_mode=variant.noise_sampling_mode,
                empirical_jitter_fraction=variant.empirical_jitter_fraction,
                score_gmm_components=variant.score_gmm_components,
                multiplicative_scattering_degree=variant.multiplicative_scattering_degree,
                additive_baseline_shift_std=variant.additive_baseline_shift_std,
            )
        )
        if progress:
            print(
                f"  -> RF AUC {eval_row.auc_rf_mean:.4f} +/- {eval_row.auc_rf_std:.4f}; "
                f"LR AUC {eval_row.auc_lr_mean:.4f} +/- {eval_row.auc_lr_std:.4f}",
                flush=True,
            )

    return {
        "status": "done",
        "dataset": str(directory),
        "n_pca_components": int(n_pca_components),
        "n_variants": len(variants),
        "rows": rows,
    }


def write_csv(rows: list[ComparisonRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(ComparisonRow)], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(result: dict[str, Any], *, report_path: Path, csv_path: Path | None) -> str:
    rows: list[ComparisonRow] = list(result["rows"])
    csv_line = f"- csv: `{csv_path}`" if csv_path is not None else "- csv: `not_written`"
    if rows:
        sorted_rows = sorted(rows, key=lambda row: abs(row.auc_rf_mean - 0.5))
        best = sorted_rows[0]
    else:
        best = None
    lines: list[str] = [
        "# exp34 X-Realism Mode Comparison",
        "",
        f"- audit_scope: `{EXP34_AUDIT_SCOPE}`",
        f"- dataset: `{result['dataset']}`",
        f"- n_pca_components: `{result['n_pca_components']}`",
        f"- n_variants: `{result['n_variants']}`",
        f"- report: `{report_path}`",
        csv_line,
        "",
        "## Goal",
        "",
        "- Compare hybrid generator variants on the same dataset and PCA rank.",
        "- Win condition per variant: RandomForest AUC near 0.5 across CV splits.",
        "- Forbidden inputs (kept from prior doctrine): no labels, no targets, no splits-as-oracle, no downstream metrics, no transfer scores. Adversarial AUC IS the tuning oracle.",
        "- No `nirs4all/` import; no Y file is read.",
        "",
        "## Best Variant (closest to RF AUC 0.5)",
        "",
    ]
    if best is not None:
        lines.append(
            f"- `{best.variant_name}`: RF AUC `{best.auc_rf_mean:.4f} +/- {best.auc_rf_std:.4f}`, "
            f"LR AUC `{best.auc_lr_mean:.4f} +/- {best.auc_lr_std:.4f}`."
        )
    else:
        lines.append("- n/a (no variants ran).")

    lines.extend(
        [
            "",
            "## Per-Variant Results (sorted by |RF AUC - 0.5|)",
            "",
            "| variant | score_mode | noise_mode | gmm_k | scatter_deg | RF AUC | LR AUC |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    sorted_rows = sorted(rows, key=lambda row: abs(row.auc_rf_mean - 0.5))
    for row in sorted_rows:
        lines.append(
            f"| `{row.variant_name}` | `{row.score_sampling_mode}` | `{row.noise_sampling_mode}` | "
            f"`{row.score_gmm_components}` | `{row.multiplicative_scattering_degree}` | "
            f"`{row.auc_rf_mean:.4f} +/- {row.auc_rf_std:.4f}` | "
            f"`{row.auc_lr_mean:.4f} +/- {row.auc_lr_std:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=bench/nirs_synthetic_pfn/src python \\",
            "  bench/nirs_synthetic_pfn/experiments/exp34_mode_comparison.py \\",
            f"  --dataset {result['dataset']} \\",
            f"  --n-pca-components {result['n_pca_components']} \\",
            f"  --report {report_path} \\",
            f"  --csv {csv_path if csv_path is not None else 'bench/nirs_synthetic_pfn/reports/xrealism_mode_comparison.csv'}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--n-pca-components", type=int, default=80)
    parser.add_argument("--n-synthetic-factor", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--baseline-degree", type=int, default=3)
    parser.add_argument("--max-peaks", type=int, default=16)
    parser.add_argument("--variants-preset", type=str, default="default", choices=("default", "extended", "knn"))
    parser.add_argument("--subsample-rows", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.variants_preset == "default":
        variants = DEFAULT_VARIANTS
    elif args.variants_preset == "extended":
        variants = EXTENDED_VARIANTS
    elif args.variants_preset == "knn":
        variants = KNN_VARIANTS
    else:
        raise ValueError(f"unknown variants preset {args.variants_preset}")

    result = run_comparison(
        args.dataset,
        variants=variants,
        n_pca_components=args.n_pca_components,
        n_synthetic_factor=args.n_synthetic_factor,
        n_splits=args.n_splits,
        seed=args.seed,
        baseline_degree=args.baseline_degree,
        max_peaks=args.max_peaks,
        n_estimators=args.n_estimators,
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
    print(json.dumps({"n_variants": result["n_variants"]}, indent=2))


if __name__ == "__main__":
    main()
