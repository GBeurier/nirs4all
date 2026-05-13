"""Aggregate multi-seed benchmark results into paired tables and figures.

This module ingests heterogeneous result CSVs (the AOM_v0 wide schema and the
benchmark-harness schema) and produces:

- a unified long-format DataFrame keyed by ``dataset, variant, seed``;
- per-comparison paired tables (candidate vs reference) with bootstrap CIs,
  Wilcoxon / sign-test p-values (Holm-corrected over the pre-registered
  family), Cliff's delta and no-harm tails;
- seed-stability and runtime summaries;
- LaTeX tables and matplotlib PDF/PNG figures for the Talanta submission;
- companion markdown reports under ``paper_aom/review/``.

The script is designed to be run repeatedly while result generation is still
in flight: ``--partial`` aggregates whatever exists on disk and warns about
missing workspaces, while ``--strict`` requires all pre-registered workspaces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths and pre-registered comparison family
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = REPO_ROOT / "paper_aom" / "review"
TABLES_DIR = REPO_ROOT / "paper_aom" / "tables"
FIGURES_DIR = REPO_ROOT / "paper_aom" / "figures"
COHORT_MANIFEST = REVIEW_DIR / "cohort_manifest.csv"

# The full set of workspaces to look at. Order matters: schema mapping is
# inferred from the file's columns, so we keep one entry per logical run.
WORKSPACES: list[dict[str, str]] = [
    {
        "name": "aom_pls_full_seed0",
        "path": "bench/AOM_v0/benchmark_runs/full/results.csv",
        "schema": "aom_v0_wide",
        "required": False,
    },
    {
        "name": "aom_pls_paper_seed0",
        "path": "bench/scenarios/runs/paper_aom_aompls_seed0/results.csv",
        "schema": "aom_v0_wide",
        "required": False,
    },
    {
        "name": "aom_pls_paper_seeds012",
        "path": "bench/scenarios/runs/paper_aom_aompls_seeds012/results.csv",
        "schema": "aom_v0_wide",
        "required": True,
    },
    {
        "name": "aom_pls_da_seeds012",
        "path": "bench/AOM_v0/benchmark_runs/paper_aom_aompls_da_seeds012/results.csv",
        "schema": "aom_v0_wide",
        "required": True,
    },
    {
        "name": "aom_ridge_audit_seeds012",
        "path": "bench/AOM_v0/Ridge/benchmark_runs/da001_audit20_seeds012/results.csv",
        "schema": "harness",
        "required": False,
    },
    {
        "name": "aom_ridge_paper_seeds012",
        "path": "bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012/results.csv",
        "schema": "harness",
        "required": True,
    },
    {
        "name": "aom_ridge_cls_seeds012",
        "path": "bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_cls_seeds012/results.csv",
        "schema": "harness",
        "required": True,
    },
    {
        "name": "linear_hpo_seed0",
        "path": "bench/scenarios/runs/paper_aom_linear_hpo/results.csv",
        "schema": "linear_hpo",
        "required": True,
    },
]

PRIMARY_COMPARISONS: list[tuple[str, str, str, str]] = [
    # (label, candidate, reference, task)
    ("ASLS-AOM-compact-cv5 vs PLS-TabPFN-HPO", "ASLS-AOM-compact-cv5-numpy", "PLS-tabpfn-hpo-25trials", "regression"),
    ("ASLS-AOM-compact-cv5 vs PLS-default", "ASLS-AOM-compact-cv5-numpy", "PLS-default-cv5", "regression"),
    ("AOM-compact-cv5 vs PLS-default", "AOM-compact-cv5-numpy", "PLS-default-cv5", "regression"),
    ("AOMRidge-global-compact-none vs Ridge-TabPFN-HPO", "AOMRidge-global-compact-none", "Ridge-tabpfn-hpo-60trials", "regression"),
    ("AOMRidge-Local-compact-knn50 vs Ridge-TabPFN-HPO", "AOMRidge-Local-compact-knn50", "Ridge-tabpfn-hpo-60trials", "regression"),
    ("AOMRidge-Blender vs Ridge-TabPFN-HPO", "AOMRidge-Blender-headline-spxy3", "Ridge-tabpfn-hpo-60trials", "regression"),
    ("AOMRidge-AutoSelect vs Ridge-TabPFN-HPO", "AOMRidge-AutoSelect-headline-spxy3", "Ridge-tabpfn-hpo-60trials", "regression"),
]

CANONICAL_VARIANTS: list[str] = sorted(
    {c for _, c, _, _ in PRIMARY_COMPARISONS} | {r for _, _, r, _ in PRIMARY_COMPARISONS}
)

EXCLUDED_DATASETS: set[str] = {"Quartz_spxy70"}  # QUARTZ family is excluded from primary analyses.

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


@dataclass
class LoadReport:
    """Bookkeeping for which result files were found and what was loaded."""

    found: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------

_AOM_V0_WIDE_RENAME = {
    "RMSEP": "rmsep",
    "MAE_test": "mae",
    "r2_test": "r2",
    "aom_variant": "variant",
    "fit_time_s": "fit_time_s",
    "predict_time_s": "predict_time_s",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "macro_f1",
    "log_loss": "log_loss",
    "ece": "ece",
    "seed": "seed",
    "dataset": "dataset",
    "task": "task",
    "status": "status",
}

_HARNESS_RENAME = {
    "canonical_name": "variant",
    "rmsep": "rmsep",
    "mae": "mae",
    "r2": "r2",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "macro_f1",
    "fit_time_s": "fit_time_s",
    "predict_time_s": "predict_time_s",
    "n_train": "n_train",
    "n_test": "n_test",
    "n_features": "n_features",
    "seed": "seed",
    "dataset": "dataset",
    "task": "task",
    "status": "status",
}

UNIFIED_COLUMNS = [
    "source_run",
    "dataset",
    "task",
    "variant",
    "seed",
    "status",
    "rmsep",
    "mae",
    "r2",
    "balanced_accuracy",
    "macro_f1",
    "log_loss",
    "ece",
    "fit_time_s",
    "predict_time_s",
    "search_time_s",
    "total_time_s",
    "n_components_selected",
    "n_train",
    "n_test",
    "n_features",
]


def _map_aom_v0_wide(df: pd.DataFrame, source_run: str) -> pd.DataFrame:
    """Translate the AOM_v0 wide schema into the unified columns."""
    out = pd.DataFrame()
    for src, dst in _AOM_V0_WIDE_RENAME.items():
        out[dst] = df[src] if src in df.columns else np.nan
    out["source_run"] = source_run
    out["n_components_selected"] = df.get("n_components_selected", np.nan)
    out["n_train"] = np.nan
    out["n_test"] = np.nan
    out["n_features"] = np.nan
    out["search_time_s"] = np.nan
    return out


def _map_harness(df: pd.DataFrame, source_run: str) -> pd.DataFrame:
    """Translate the benchmark-harness schema into the unified columns."""
    out = pd.DataFrame()
    for src, dst in _HARNESS_RENAME.items():
        out[dst] = df[src] if src in df.columns else np.nan
    out["source_run"] = source_run
    out["n_components_selected"] = np.nan
    out["log_loss"] = df.get("log_loss", np.nan)
    out["ece"] = df.get("ece", np.nan)
    out["search_time_s"] = df.get("search_time_s", np.nan)
    return out


def _map_linear_hpo(df: pd.DataFrame, source_run: str) -> pd.DataFrame:
    """Translate the linear-HPO harness schema (``variant``, ``refit_time_s``).

    Dataset names in this schema are prefixed with the family (``ALPINE/foo``);
    we drop the prefix so they line up with the rest of the cohort.
    """
    out = pd.DataFrame()
    dataset = df.get("dataset", pd.Series([], dtype="string"))
    if isinstance(dataset, pd.Series):
        dataset = dataset.astype("string").str.split("/").str[-1]
    out["dataset"] = dataset
    out["task"] = df.get("task", "regression")
    out["variant"] = df.get("variant", np.nan)
    out["seed"] = df.get("seed", np.nan)
    out["status"] = df.get("status", np.nan)
    out["rmsep"] = df.get("rmsep", np.nan)
    out["mae"] = df.get("mae", np.nan)
    out["r2"] = df.get("r2", np.nan)
    out["balanced_accuracy"] = df.get("balanced_accuracy", np.nan)
    out["macro_f1"] = df.get("macro_f1", np.nan)
    out["log_loss"] = df.get("log_loss", np.nan)
    out["ece"] = df.get("ece", np.nan)
    # The linear-HPO harness logs only the refit time as fit_time.
    out["fit_time_s"] = df.get("refit_time_s", np.nan)
    out["predict_time_s"] = df.get("predict_time_s", np.nan)
    out["search_time_s"] = df.get("search_time_s", np.nan)
    out["n_components_selected"] = np.nan
    out["n_train"] = df.get("n_train", np.nan)
    out["n_test"] = df.get("n_test", np.nan)
    out["n_features"] = df.get("n_features", np.nan)
    out["source_run"] = source_run
    return out


_SCHEMA_MAPPERS = {
    "aom_v0_wide": _map_aom_v0_wide,
    "harness": _map_harness,
    "linear_hpo": _map_linear_hpo,
}


def load_all_results(
    workspaces: Sequence[dict[str, str]] | None = None,
    repo_root: Path | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, LoadReport]:
    """Read every available workspace CSV and concatenate into long format.

    Parameters
    ----------
    workspaces:
        Sequence of workspace dicts (``name``, ``path``, ``schema``,
        ``required``). Defaults to the module-level :data:`WORKSPACES`.
    repo_root:
        Repository root. Defaults to the package root.
    strict:
        If True, raise ``FileNotFoundError`` when a required workspace
        is missing on disk. Otherwise emit a warning.

    Returns
    -------
    (df, report)
        ``df`` is a long-format DataFrame with the columns listed in
        :data:`UNIFIED_COLUMNS`. ``report`` summarises what was found.
    """

    workspaces = list(workspaces) if workspaces is not None else WORKSPACES
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT

    frames: list[pd.DataFrame] = []
    report = LoadReport()

    for entry in workspaces:
        csv_path = repo_root / entry["path"]
        if not csv_path.exists():
            if entry.get("required", False):
                msg = f"Required workspace missing: {entry['name']} ({csv_path})"
                report.missing_required.append(entry["name"])
                if strict:
                    raise FileNotFoundError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                report.missing_optional.append(entry["name"])
                warnings.warn(
                    f"Optional workspace missing: {entry['name']} ({csv_path})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        raw = pd.read_csv(csv_path, low_memory=False)
        mapper = _SCHEMA_MAPPERS[entry["schema"]]
        mapped = mapper(raw, source_run=entry["name"])
        # Ensure all unified columns exist.
        for col in UNIFIED_COLUMNS:
            if col not in mapped.columns:
                mapped[col] = np.nan
        mapped = mapped[UNIFIED_COLUMNS]
        # Coerce numeric columns.
        numeric_cols = [
            "seed",
            "rmsep",
            "mae",
            "r2",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
            "ece",
            "fit_time_s",
            "predict_time_s",
            "search_time_s",
            "n_components_selected",
            "n_train",
            "n_test",
            "n_features",
        ]
        for col in numeric_cols:
            mapped[col] = pd.to_numeric(mapped[col], errors="coerce")
        # Derive total_time_s = fit + predict (+ search where present).
        mapped["total_time_s"] = mapped[["fit_time_s", "predict_time_s", "search_time_s"]].sum(
            axis=1, min_count=1
        )
        frames.append(mapped)
        report.found.append(entry["name"])
        report.row_counts[entry["name"]] = len(mapped)

    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS), report

    df = pd.concat(frames, ignore_index=True)
    # Strip whitespace from string keys.
    for col in ("dataset", "variant", "task", "status", "source_run"):
        df[col] = df[col].astype("string").str.strip()
    return df, report


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

REGRESSION_LOWER_IS_BETTER = {"rmsep", "mae", "log_loss", "ece"}
DEFAULT_METRIC = {
    "regression": "rmsep",
    "classification": "balanced_accuracy",
}


def _per_dataset_seed_mean(
    df: pd.DataFrame, variant: str, metric: str, task: str
) -> pd.Series:
    """Average ``metric`` across seeds for each dataset for one variant.

    Variant matching is case-insensitive because different result writers use
    different casing conventions (e.g. ``PLS-default-cv5`` vs ``pls-default-cv5``).
    """
    variant_lower = variant.lower()
    variants_norm = df["variant"].astype("string").str.lower()
    sub = df[(variants_norm == variant_lower) & (df["task"] == task)]
    sub = sub[sub["status"].isna() | (sub["status"].str.lower().isin({"ok", "success", "completed", ""}))]
    if sub.empty:
        return pd.Series(dtype=float, name=variant)
    grouped = sub.groupby("dataset")[metric].mean()
    return grouped.rename(variant)


def paired_table(
    df: pd.DataFrame,
    candidates: Sequence[str],
    reference: str,
    metric: str,
    task: str = "regression",
) -> pd.DataFrame:
    """Build the per-dataset paired comparison for one (candidate, reference).

    For lower-is-better metrics the column ``effect`` is the ratio
    candidate/reference. For higher-is-better metrics it is the
    difference candidate - reference.

    The output always includes one row per dataset where both sides have a
    finite value and the dataset is not in :data:`EXCLUDED_DATASETS`.
    """

    lower = metric in REGRESSION_LOWER_IS_BETTER
    ref_series = _per_dataset_seed_mean(df, reference, metric, task)
    rows: list[dict[str, float | str]] = []
    for cand in candidates:
        cand_series = _per_dataset_seed_mean(df, cand, metric, task)
        common = cand_series.index.intersection(ref_series.index)
        for dataset in sorted(common):
            if dataset in EXCLUDED_DATASETS:
                continue
            cv = cand_series.loc[dataset]
            rv = ref_series.loc[dataset]
            if not (np.isfinite(cv) and np.isfinite(rv)):
                continue
            if lower:
                if rv == 0 or not np.isfinite(rv):
                    continue
                effect = float(cv / rv)
            else:
                effect = float(cv - rv)
            rows.append(
                {
                    "candidate": cand,
                    "reference": reference,
                    "dataset": dataset,
                    "candidate_value": float(cv),
                    "reference_value": float(rv),
                    "effect": effect,
                    "metric": metric,
                    "lower_is_better": lower,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _bootstrap_median_ci(
    values: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta between paired vectors using the standard estimator."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    # Vectorised pairwise comparison via broadcasting.
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.sum() / (nx * ny))


def compute_paired_stats(paired: pd.DataFrame, lower_is_better: bool) -> dict:
    """Compute the headline statistics for a paired comparison table."""
    n = len(paired)
    if n == 0:
        return {
            "n": 0,
            "median_effect": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "wilcoxon_p": float("nan"),
            "sign_p": float("nan"),
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "cliffs_delta": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "worst": float("nan"),
            "worst_dataset": None,
        }

    cand = paired["candidate_value"].to_numpy(dtype=float)
    ref = paired["reference_value"].to_numpy(dtype=float)
    effects = paired["effect"].to_numpy(dtype=float)
    datasets = paired["dataset"].to_numpy()

    median_effect = float(np.median(effects))
    ci_lower, ci_upper = _bootstrap_median_ci(effects)

    # Wilcoxon on log-ratios (lower_is_better) or on raw deltas (higher_is_better).
    if lower_is_better:
        signal = np.log(effects[effects > 0])
    else:
        signal = effects
    if signal.size >= 1 and np.any(signal != 0):
        try:
            wilcoxon_p = float(
                stats.wilcoxon(signal, alternative="two-sided", zero_method="wilcox").pvalue
            )
        except ValueError:
            wilcoxon_p = float("nan")
    else:
        wilcoxon_p = float("nan")

    if lower_is_better:
        # candidate wins when cand < ref.
        wins = int(np.sum(cand < ref))
        losses = int(np.sum(cand > ref))
    else:
        wins = int(np.sum(cand > ref))
        losses = int(np.sum(cand < ref))
    ties = int(n - wins - losses)

    # Sign test (binomial on wins vs losses, ignoring ties).
    sig_count = wins + losses
    if sig_count > 0:
        try:
            sign_p = float(
                stats.binomtest(wins, sig_count, p=0.5, alternative="two-sided").pvalue
            )
        except AttributeError:  # SciPy < 1.7 fallback.
            sign_p = float(stats.binom_test(wins, sig_count, p=0.5, alternative="two-sided"))
    else:
        sign_p = float("nan")

    cliffs = _cliffs_delta(cand, ref)
    if not lower_is_better:
        cliffs = -cliffs  # so that "candidate wins" -> positive delta.

    # No-harm tails on the effect.
    if lower_is_better:
        q75 = float(np.percentile(effects, 75))
        q90 = float(np.percentile(effects, 90))
        worst_idx = int(np.argmax(effects))
    else:
        q75 = float(np.percentile(effects, 25))
        q90 = float(np.percentile(effects, 10))
        worst_idx = int(np.argmin(effects))

    return {
        "n": int(n),
        "median_effect": median_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "wilcoxon_p": wilcoxon_p,
        "sign_p": sign_p,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cliffs_delta": cliffs,
        "q75": q75,
        "q90": q90,
        "worst": float(effects[worst_idx]),
        "worst_dataset": str(datasets[worst_idx]),
    }


def holm_correct(pvals: dict[tuple, float]) -> dict[tuple, float]:
    """Apply Holm-Bonferroni correction over a family of comparisons."""
    items = [(k, p) for k, p in pvals.items() if p is not None and not math.isnan(p)]
    if not items:
        return {k: float("nan") for k in pvals}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    corrected: dict[tuple, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - i) * p)
        running = max(running, adjusted)
        corrected[key] = running
    for k in pvals:
        if k not in corrected:
            corrected[k] = float("nan")
    return corrected


def seed_stability(
    df: pd.DataFrame, candidates: Sequence[str], metric: str
) -> pd.DataFrame:
    """Per-method median/IQR/std across seeds and winner-change dataset count."""
    rows: list[dict[str, float | str]] = []
    candidate_lower = {c.lower() for c in candidates}
    variants_norm = df["variant"].astype("string").str.lower()
    sub = df[variants_norm.isin(candidate_lower)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "n_datasets",
                "n_seeds",
                "median",
                "iqr",
                "std",
                "winner_changes",
            ]
        )
    for variant in candidates:
        v = sub[sub["variant"].astype("string").str.lower() == variant.lower()]
        if v.empty:
            continue
        per_seed = v.groupby(["dataset", "seed"])[metric].mean().unstack("seed")
        seed_counts = per_seed.notna().sum(axis=1)
        full_datasets = per_seed[seed_counts >= 2]
        median = float(np.nanmedian(per_seed.values)) if per_seed.size else float("nan")
        iqr = float(
            np.nanpercentile(per_seed.values, 75) - np.nanpercentile(per_seed.values, 25)
        ) if per_seed.size else float("nan")
        std = float(np.nanstd(per_seed.values)) if per_seed.size else float("nan")
        rows.append(
            {
                "variant": variant,
                "n_datasets": int(per_seed.shape[0]),
                "n_seeds": int(per_seed.shape[1]),
                "median": median,
                "iqr": iqr,
                "std": std,
                "n_full_seed_datasets": int(full_datasets.shape[0]),
            }
        )
    out = pd.DataFrame(rows)

    # Winner change count: per dataset, how often the best variant flips across seeds.
    if not candidates:
        return out
    pivot = (
        sub.groupby(["dataset", "seed", "variant"])[metric]
        .mean()
        .unstack("variant")
    )
    pivot = pivot.dropna(axis=0, how="all")
    if pivot.empty:
        out["winner_changes"] = 0
        return out
    lower = metric in REGRESSION_LOWER_IS_BETTER
    if lower:
        winners = pivot.idxmin(axis=1, skipna=True)
    else:
        winners = pivot.idxmax(axis=1, skipna=True)
    winners_df = winners.unstack("seed")
    distinct = winners_df.nunique(axis=1, dropna=True)
    flips = int((distinct > 1).sum())
    out["winner_changes"] = flips
    return out


def runtime_summary(
    df: pd.DataFrame, candidates: Sequence[str]
) -> pd.DataFrame:
    """Median/q75/q90 fit and total time per candidate plus failure counts."""
    rows = []
    variants_norm = df["variant"].astype("string").str.lower()
    for variant in candidates:
        sub = df[variants_norm == variant.lower()]
        if sub.empty:
            rows.append(
                {
                    "variant": variant,
                    "n": 0,
                    "median_fit_s": float("nan"),
                    "q75_fit_s": float("nan"),
                    "q90_fit_s": float("nan"),
                    "median_total_s": float("nan"),
                    "q75_total_s": float("nan"),
                    "q90_total_s": float("nan"),
                    "failures": 0,
                    "timeouts": 0,
                }
            )
            continue
        fit = sub["fit_time_s"].dropna().to_numpy()
        total = sub["total_time_s"].dropna().to_numpy()
        status = sub["status"].fillna("").str.lower()
        failures = int((status.isin({"failed", "error"})).sum())
        timeouts = int((status == "timeout").sum())
        rows.append(
            {
                "variant": variant,
                "n": int(len(sub)),
                "median_fit_s": float(np.median(fit)) if fit.size else float("nan"),
                "q75_fit_s": float(np.percentile(fit, 75)) if fit.size else float("nan"),
                "q90_fit_s": float(np.percentile(fit, 90)) if fit.size else float("nan"),
                "median_total_s": float(np.median(total)) if total.size else float("nan"),
                "q75_total_s": float(np.percentile(total, 75)) if total.size else float("nan"),
                "q90_total_s": float(np.percentile(total, 90)) if total.size else float("nan"),
                "failures": failures,
                "timeouts": timeouts,
            }
        )
    return pd.DataFrame(rows)


def friedman_test(
    df: pd.DataFrame, candidates: Sequence[str], metric: str
) -> dict:
    """Friedman test plus Nemenyi critical difference on a common dataset set."""
    lower = metric in REGRESSION_LOWER_IS_BETTER
    candidate_lower = {c.lower() for c in candidates}
    variants_norm = df["variant"].astype("string").str.lower()
    pivot = (
        df[variants_norm.isin(candidate_lower)]
        .groupby(["dataset", "variant"])[metric]
        .mean()
        .unstack("variant")
    )
    pivot = pivot.dropna(axis=0)
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_datasets": int(pivot.shape[0]),
            "n_methods": int(pivot.shape[1]),
            "critical_difference": float("nan"),
            "mean_ranks": {},
        }
    ranks = pivot.rank(axis=1, ascending=lower)
    mean_ranks = ranks.mean(axis=0).to_dict()
    stat, p = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    # Nemenyi critical difference (alpha=0.05) via q_alpha table:
    k = pivot.shape[1]
    n = pivot.shape[0]
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_alpha.get(k, 3.164)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n))
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_datasets": int(n),
        "n_methods": int(k),
        "critical_difference": float(cd),
        "mean_ranks": {str(k_): float(v) for k_, v in mean_ranks.items()},
    }


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------


def _fmt_p(p: float) -> str:
    if p is None or math.isnan(p):
        return "n/a"
    if p < 1e-3:
        return f"${p:.2e}$".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
    return f"{p:.3f}"


def _fmt_effect(stats_row: dict, lower_is_better: bool) -> str:
    if math.isnan(stats_row["median_effect"]):
        return "n/a"
    if lower_is_better:
        return f"RMSEP ratio {stats_row['median_effect']:.3f}"
    return rf"$\Delta$={stats_row['median_effect'] * 100:.2f}\%"


def _fmt_ci(stats_row: dict, lower_is_better: bool) -> str:
    if math.isnan(stats_row["ci_lower"]):
        return "n/a"
    if lower_is_better:
        return f"{stats_row['ci_lower']:.3f}--{stats_row['ci_upper']:.3f}"
    return f"{stats_row['ci_lower'] * 100:.2f} to {stats_row['ci_upper'] * 100:.2f}\\%"


def _latex_text(value: object) -> str:
    """Escape plain-text table cells for LaTeX."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_latex_table_main(
    rows: list[dict], output_path: Path | str
) -> Path:
    """Write ``table_main_results.tex`` with the headline paired numbers."""
    output_path = Path(output_path)
    lines = [
        r"\begin{tabularx}{\linewidth}{p{0.30\linewidth}Xrrr}",
        r"\toprule",
        r"Comparison & Source run & Datasets & Median effect & Wins \\",
        r"\midrule",
    ]
    for row in rows:
        stats_ = row["stats"]
        if stats_["n"] == 0:
            effect_str = "n/a"
        else:
            effect_str = _fmt_effect(stats_, row["lower_is_better"])
        wins_str = f"{stats_['wins']}/{stats_['n']}" if stats_["n"] else "0/0"
        lines.append(
            f"{_latex_text(row['label'])} & {_latex_text(row['source'])} & "
            f"{stats_['n']} & {effect_str} & {wins_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", ""]
    output_path.write_text("\n".join(lines))
    return output_path


def format_latex_table_paired(
    rows: list[dict], output_path: Path | str
) -> Path:
    """Write ``table_paired_stats.tex`` with N / CI / wins / Holm p."""
    output_path = Path(output_path)
    lines = [
        r"\begin{tabularx}{\linewidth}{Xrrrrr}",
        r"\toprule",
        r"Comparison & $N$ & Median effect & 95\% bootstrap CI & Wins & Wilcoxon $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        stats_ = row["stats"]
        effect_str = _fmt_effect(stats_, row["lower_is_better"]) if stats_["n"] else "n/a"
        ci_str = _fmt_ci(stats_, row["lower_is_better"]) if stats_["n"] else "n/a"
        wins_str = f"{stats_['wins']}/{stats_['n']}" if stats_["n"] else "0/0"
        p_str = _fmt_p(row.get("p_holm", float("nan")))
        lines.append(
            f"{_latex_text(row['label'])} & {stats_['n']} & {effect_str} & "
            f"{ci_str} & {wins_str} & {p_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", ""]
    output_path.write_text("\n".join(lines))
    return output_path


def format_latex_table_time(
    rt: pd.DataFrame, output_path: Path | str
) -> Path:
    """Write ``table_time_budget.tex`` with per-candidate runtime quantiles."""
    output_path = Path(output_path)
    lines = [
        r"\begin{tabularx}{\linewidth}{Xrrrrrr}",
        r"\toprule",
        r"Candidate & $N$ & Median fit (s) & q75 fit & q90 fit & Median total (s) & Failures \\",
        r"\midrule",
    ]
    for _, r in rt.iterrows():
        def fmt(v: float) -> str:
            return f"{v:.2f}" if pd.notna(v) else "n/a"
        lines.append(
            f"{_latex_text(r['variant'])} & {int(r['n'])} & {fmt(r['median_fit_s'])} & "
            f"{fmt(r['q75_fit_s'])} & {fmt(r['q90_fit_s'])} & "
            f"{fmt(r['median_total_s'])} & {int(r['failures'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", ""]
    output_path.write_text("\n".join(lines))
    return output_path


def format_latex_table_classification(
    rows: list[dict], output_path: Path | str
) -> Path:
    """Write ``table_classification_main.tex``."""
    output_path = Path(output_path)
    lines = [
        r"\begin{tabularx}{\linewidth}{Xrrrrr}",
        r"\toprule",
        r"Comparison & $N$ & $\Delta$ balanced acc. & 95\% CI & Wins & Wilcoxon $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        stats_ = row["stats"]
        if stats_["n"] == 0:
            effect_str = "n/a"
            ci_str = "n/a"
            wins_str = "0/0"
        else:
            effect_str = f"{stats_['median_effect']:.3f}"
            ci_str = f"{stats_['ci_lower']:.3f}--{stats_['ci_upper']:.3f}"
            wins_str = f"{stats_['wins']}/{stats_['n']}"
        p_str = _fmt_p(row.get("p_holm", float("nan")))
        lines.append(
            f"{_latex_text(row['label'])} & {stats_['n']} & {effect_str} & "
            f"{ci_str} & {wins_str} & {p_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", ""]
    output_path.write_text("\n".join(lines))
    return output_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _matplotlib_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


_FAMILY_COLORS = {
    "PLS": "#1f6fb4",       # blue
    "Ridge": "#3a9a3a",     # green
    "AOM-PLS": "#1fb4a6",   # teal
    "AOM-Ridge": "#0f7a6f",  # darker teal
    "TabPFN": "#999999",
    "Other": "#7f7f7f",
}


def _variant_family(variant: str) -> str:
    v = variant.lower()
    if "aomridge" in v or "aom-ridge" in v:
        return "AOM-Ridge"
    if "aom" in v:
        return "AOM-PLS"
    if "ridge" in v:
        return "Ridge"
    if "pls" in v:
        return "PLS"
    if "tabpfn" in v:
        return "TabPFN"
    return "Other"


def plot_accuracy_time_pareto(
    df: pd.DataFrame,
    candidates: Sequence[str],
    output_path: Path | str,
    metric: str = "rmsep",
) -> Path:
    """Median fit time vs median metric across datasets, per candidate."""
    plt = _matplotlib_setup()
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    variants_norm = df["variant"].astype("string").str.lower()
    for variant in candidates:
        sub = df[variants_norm == variant.lower()]
        if sub.empty:
            continue
        per_dataset_metric = sub.groupby("dataset")[metric].mean()
        per_dataset_time = sub.groupby("dataset")["fit_time_s"].mean()
        x = per_dataset_time.median()
        y = per_dataset_metric.median()
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        family = _variant_family(variant)
        ax.scatter(
            x, y,
            s=42,
            color=_FAMILY_COLORS.get(family, "#444"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )
        ax.annotate(
            variant,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median fit time (s, log scale)")
    ax.set_ylabel(f"Median {metric.upper()} across datasets")
    ax.set_title("Accuracy vs. fit time")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return output_path


def plot_runtime_distribution(
    df: pd.DataFrame,
    candidates: Sequence[str],
    output_path: Path | str,
) -> Path:
    """Per-candidate boxplot of per-run fit_time_s."""
    plt = _matplotlib_setup()
    output_path = Path(output_path)
    variants_norm = df["variant"].astype("string").str.lower()
    data = []
    labels = []
    for variant in candidates:
        sub = df[variants_norm == variant.lower()]["fit_time_s"].dropna()
        if sub.empty:
            continue
        data.append(sub.to_numpy())
        labels.append(variant)
    if not data:
        return output_path
    fig, ax = plt.subplots(figsize=(6.0, max(2.2, 0.32 * len(labels) + 1.0)))
    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.6,
        flierprops={"marker": ".", "markersize": 3, "markerfacecolor": "#444"},
        boxprops={"linewidth": 0.6},
        medianprops={"color": "#222", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
    )
    for patch, variant in zip(bp["boxes"], labels, strict=True):
        patch.set_facecolor(_FAMILY_COLORS.get(_variant_family(variant), "#888"))
        patch.set_alpha(0.85)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Fit time (s, log scale)")
    ax.set_title("Per-run fit-time distribution")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Headline report
# ---------------------------------------------------------------------------


def _build_comparison_rows(
    df: pd.DataFrame,
    comparisons: Sequence[tuple[str, str, str, str]],
    metric_per_task: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    raw_pvals: dict[tuple, float] = {}
    for label, cand, ref, task in comparisons:
        metric = metric_per_task[task]
        lower = metric in REGRESSION_LOWER_IS_BETTER
        paired = paired_table(df, [cand], ref, metric, task=task)
        stats_ = compute_paired_stats(paired, lower_is_better=lower)
        rows.append(
            {
                "label": label,
                "candidate": cand,
                "reference": ref,
                "task": task,
                "metric": metric,
                "lower_is_better": lower,
                "source": "paper_aom multi-seed" if stats_["n"] else "missing",
                "stats": stats_,
                "paired": paired,
            }
        )
        raw_pvals[(label,)] = stats_["wilcoxon_p"]
    holm = holm_correct(raw_pvals)
    for row in rows:
        row["p_holm"] = holm.get((row["label"],), float("nan"))
    return rows


def write_final_stats_markdown(
    rows: list[dict],
    rt: pd.DataFrame,
    seed_stab: pd.DataFrame,
    friedman: dict,
    output_path: Path | str,
    report: LoadReport,
) -> Path:
    output_path = Path(output_path)
    lines: list[str] = ["# AOM final statistics summary", ""]
    lines.append("## Workspaces ingested")
    if report.found:
        for name in report.found:
            lines.append(f"- {name} ({report.row_counts.get(name, 0)} rows)")
    else:
        lines.append("- (none)")
    if report.missing_required:
        lines.append("")
        lines.append("## Missing required workspaces (run still in progress)")
        for name in report.missing_required:
            lines.append(f"- {name}")
    if report.missing_optional:
        lines.append("")
        lines.append("## Missing optional workspaces")
        for name in report.missing_optional:
            lines.append(f"- {name}")
    lines += ["", "## Pre-registered paired comparisons", ""]
    lines.append("| Comparison | N | Median effect | 95% CI | Wins | Wilcoxon p_Holm | Cliff's delta |")
    lines.append("| --- | ---: | --- | --- | --- | --- | ---: |")
    for row in rows:
        s = row["stats"]
        if s["n"] == 0:
            lines.append(f"| {row['label']} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        lower = row["lower_is_better"]
        if lower:
            effect = f"ratio={s['median_effect']:.3f}"
            ci = f"{s['ci_lower']:.3f}-{s['ci_upper']:.3f}"
        else:
            effect = f"delta={s['median_effect']:.4f}"
            ci = f"{s['ci_lower']:.4f}-{s['ci_upper']:.4f}"
        wins = f"{s['wins']}/{s['n']} (ties {s['ties']})"
        p_holm = row.get("p_holm", float("nan"))
        p_str = "n/a" if math.isnan(p_holm) else f"{p_holm:.3g}"
        lines.append(
            f"| {row['label']} | {s['n']} | {effect} | {ci} | {wins} | {p_str} | {s['cliffs_delta']:.3f} |"
        )

    lines += ["", "## Friedman / Nemenyi"]
    if math.isnan(friedman["p_value"]):
        lines.append("- insufficient data on the common dataset set.")
    else:
        lines.append(
            f"- {friedman['n_methods']} candidates on {friedman['n_datasets']} datasets, "
            f"chi^2={friedman['statistic']:.3f}, p={friedman['p_value']:.3g}, "
            f"CD@0.05={friedman['critical_difference']:.3f}."
        )
        lines.append("- mean ranks (smaller is better):")
        for v, r in sorted(friedman["mean_ranks"].items(), key=lambda kv: kv[1]):
            lines.append(f"  - {v}: {r:.2f}")

    lines += ["", "## Runtime summary"]
    if rt.empty:
        lines.append("- (no runtime data)")
    else:
        lines.append("| Variant | N | median fit | q75 fit | q90 fit | median total | failures |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in rt.iterrows():
            lines.append(
                f"| {r['variant']} | {int(r['n'])} | {r['median_fit_s']:.2f} | "
                f"{r['q75_fit_s']:.2f} | {r['q90_fit_s']:.2f} | {r['median_total_s']:.2f} | {int(r['failures'])} |"
            )

    lines += ["", "## Seed stability"]
    if seed_stab.empty:
        lines.append("- (no seed-stability data)")
    else:
        lines.append("| Variant | datasets | seeds | median | IQR | std | full-seed datasets | winner changes |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in seed_stab.iterrows():
            lines.append(
                f"| {r['variant']} | {int(r['n_datasets'])} | {int(r['n_seeds'])} | {r['median']:.4f} | "
                f"{r['iqr']:.4f} | {r['std']:.4f} | {int(r['n_full_seed_datasets'])} | "
                f"{int(r.get('winner_changes', 0))} |"
            )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_classification_markdown(
    rows: list[dict],
    output_path: Path | str,
) -> Path:
    output_path = Path(output_path)
    lines = ["# AOM classification statistics", ""]
    if not rows:
        lines.append("- (no classification comparisons available)")
        output_path.write_text("\n".join(lines) + "\n")
        return output_path
    lines.append("| Comparison | N | Delta balanced acc. | CI | Wins | Wilcoxon p_Holm |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for row in rows:
        s = row["stats"]
        if s["n"] == 0:
            lines.append(f"| {row['label']} | 0 | n/a | n/a | n/a | n/a |")
            continue
        ci = f"{s['ci_lower']:.4f}-{s['ci_upper']:.4f}"
        p_holm = row.get("p_holm", float("nan"))
        p_str = "n/a" if math.isnan(p_holm) else f"{p_holm:.3g}"
        lines.append(
            f"| {row['label']} | {s['n']} | {s['median_effect']:.4f} | {ci} | {s['wins']}/{s['n']} | {p_str} |"
        )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required workspace is missing.",
    )
    mode.add_argument(
        "--partial",
        action="store_true",
        help="Aggregate whatever exists; warn on missing (default).",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Override the repository root.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(args.repo_root)
    df, report = load_all_results(WORKSPACES, repo_root=repo_root, strict=args.strict)

    print(f"Loaded {len(df)} rows from {len(report.found)} workspaces.", file=sys.stderr)
    if report.missing_required:
        print(
            "Missing required workspaces: " + ", ".join(report.missing_required),
            file=sys.stderr,
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-registered comparisons.
    rows = _build_comparison_rows(df, PRIMARY_COMPARISONS, DEFAULT_METRIC)

    # Classification comparisons. The classification family mirrors the
    # AOM-Ridge / AOM-PLS-DA primary lineup but uses balanced_accuracy.
    classification_pairs: list[tuple[str, str, str, str]] = []
    classification_variants = (
        df[df["task"] == "classification"]["variant"].dropna().unique().tolist()
    )
    classification_rows: list[dict] = []
    if classification_variants:
        for cand in classification_variants:
            if "PLS-DA-standard" in classification_variants and cand != "PLS-DA-standard":
                classification_pairs.append((f"{cand} vs PLS-DA", cand, "PLS-DA-standard", "classification"))
        classification_rows = _build_comparison_rows(
            df, classification_pairs, DEFAULT_METRIC
        )

    # Runtime / seed stability over the candidate family.
    runtime_candidates = list({c for _, c, _, _ in PRIMARY_COMPARISONS}) + list(
        {r for _, _, r, _ in PRIMARY_COMPARISONS}
    )
    rt = runtime_summary(df, runtime_candidates)
    seed_stab = seed_stability(df, runtime_candidates, "rmsep")

    variants_present = {
        str(v).lower() for v in df["variant"].dropna().unique()
    }
    common_variants = [v for v in runtime_candidates if v.lower() in variants_present]
    fr = friedman_test(df, common_variants, "rmsep")

    # Emit LaTeX tables.
    format_latex_table_main(rows, TABLES_DIR / "table_main_results.tex")
    format_latex_table_paired(rows, TABLES_DIR / "table_paired_stats.tex")
    format_latex_table_time(rt, TABLES_DIR / "table_time_budget.tex")
    format_latex_table_classification(
        classification_rows, TABLES_DIR / "table_classification_main.tex"
    )

    # Figures.
    plot_accuracy_time_pareto(
        df, common_variants, FIGURES_DIR / "fig_accuracy_time_pareto"
    )
    plot_runtime_distribution(df, common_variants, FIGURES_DIR / "fig_runtime_distribution")

    # Markdown summaries.
    write_final_stats_markdown(
        rows, rt, seed_stab, fr, REVIEW_DIR / "final_stats.md", report
    )
    write_classification_markdown(
        classification_rows, REVIEW_DIR / "classification_stats.md"
    )

    print(
        json.dumps(
            {
                "rows_total": len(df),
                "workspaces_found": report.found,
                "missing_required": report.missing_required,
                "missing_optional": report.missing_optional,
                "primary_comparisons_with_data": [
                    r["label"] for r in rows if r["stats"]["n"] > 0
                ],
                "friedman_p": fr["p_value"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
