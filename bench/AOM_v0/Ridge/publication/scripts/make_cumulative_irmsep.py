"""Generate cumulative iRMSEP and Pareto figures for the AOM-Ridge paper.

Two figures are produced (each saved as PDF + PNG into ``../figures/``):

- ``fig_cumulative_irmsep``  Cumulative iRMSEP versus the TabPFN-Raw
  baseline. Two side-by-side panels: datasets sorted by ``n_features``
  (left) and by ``n_train`` (right). Shows the running sum of

      iRMSEP(M, i) = (RMSEP_TabPFN-Raw(i) - RMSEP_M(i)) / RMSEP_TabPFN-Raw(i)

  for the eight non-baseline methods (TabPFN-opt, Ridge, PLS, CatBoost,
  Top-1/Top-2 AOM-PLS, Top-1/Top-2 AOM-Ridge). The TabPFN-Raw baseline
  is the zero line.

- ``fig_irmsep_vs_time``     Pareto / efficiency view: mean iRMSEP across
  datasets versus log10(fit_time_s). Markers sized by the number of
  datasets where the method beats TabPFN-Raw. The non-dominated frontier
  (lower-left dominance) is annotated.

Inputs (read-only):
  - ``bench/AOM_v0/publication/tables/master_pivot.csv``   TabPFN paper
    baselines
  - ``bench/AOM_v0/benchmarks/cohort_regression.csv``      Cohort
    metadata (n_train, n_test, p)
  - ``bench/AOM_v0/benchmark_runs/full/results.csv``        AOM-PLS
    benchmark
  - ``bench/AOM_v0/Ridge/benchmark_runs/curated_v2/results.csv``  if
    present, otherwise ``.../curated/results.csv``        AOM-Ridge
    benchmark

Conventions match the existing AOM-PLS / AOM-Ridge publication scripts:
the ``Quartz_spxy70`` dataset is dropped (degenerate reference RMSE),
only datasets present in **all four** input files are kept (inner join),
and the top-1 / top-2 AOM variants are picked by lowest median RMSEP
across the inner-join datasets so a single uniform variant label is
applied per method.

Usage::

    PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge \
        .venv/bin/python bench/AOM_v0/Ridge/publication/scripts/make_cumulative_irmsep.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PUB_ROOT = HERE.parent
RIDGE_ROOT = PUB_ROOT.parent  # bench/AOM_v0/Ridge
AOM_ROOT = RIDGE_ROOT.parent  # bench/AOM_v0

DEFAULT_OUT = PUB_ROOT / "figures"
DEFAULT_MASTER = AOM_ROOT / "publication" / "tables" / "master_pivot.csv"
DEFAULT_COHORT = AOM_ROOT / "benchmarks" / "cohort_regression.csv"
DEFAULT_AOMPLS = AOM_ROOT / "benchmark_runs" / "full" / "results.csv"
DEFAULT_RIDGE_V2 = RIDGE_ROOT / "benchmark_runs" / "curated_v2" / "results.csv"
DEFAULT_RIDGE_V1 = RIDGE_ROOT / "benchmark_runs" / "curated" / "results.csv"

# Datasets to drop (degenerate reference RMSE).
EXCLUDED_DATASETS = {"Quartz_spxy70"}

# ---------------------------------------------------------------------------
# Plot style — color-blind safe (Okabe-Ito + a few accents). The AOM family
# is bold and saturated; baselines use muted greys / desaturated tones.
# ---------------------------------------------------------------------------

METHOD_STYLE: dict[str, dict] = {
    # Baselines (muted)
    "TabPFN-Raw":   {"color": "#999999", "linestyle": ":",  "linewidth": 1.3, "marker": "x"},
    "TabPFN-opt":   {"color": "#D55E00", "linestyle": "--", "linewidth": 1.4, "marker": "s"},
    "Ridge":        {"color": "#56B4E9", "linestyle": "--", "linewidth": 1.4, "marker": "v"},
    "PLS":          {"color": "#009E73", "linestyle": "--", "linewidth": 1.4, "marker": "^"},
    "Catboost":     {"color": "#CC79A7", "linestyle": "--", "linewidth": 1.4, "marker": "P"},
    # AOM family (bold, prominent)
    "AOM-PLS-1":    {"color": "#E69F00", "linestyle": "-",  "linewidth": 2.2, "marker": "o"},
    "AOM-PLS-2":    {"color": "#F0A030", "linestyle": "-",  "linewidth": 1.9, "marker": "o"},
    "AOM-Ridge-1":  {"color": "#0072B2", "linestyle": "-",  "linewidth": 2.4, "marker": "D"},
    "AOM-Ridge-2":  {"color": "#3590D0", "linestyle": "-",  "linewidth": 2.0, "marker": "D"},
}

PLT_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _resolve_ridge_path(arg: Path | None) -> Path:
    if arg is not None:
        return Path(arg)
    if DEFAULT_RIDGE_V2.exists():
        return DEFAULT_RIDGE_V2
    return DEFAULT_RIDGE_V1


def _load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)].copy()
    return df


def _load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)].copy()
    df["n_train"] = pd.to_numeric(df["n_train"], errors="coerce")
    df["n_test"] = pd.to_numeric(df["n_test"], errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.rename(columns={"p": "n_features"})
    df = df.dropna(subset=["n_train", "n_features"])
    return df[["database_name", "dataset", "n_train", "n_test", "n_features"]]


def _load_aompls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)].copy()
    df["RMSEP"] = pd.to_numeric(df["RMSEP"], errors="coerce")
    df["fit_time_s"] = pd.to_numeric(df["fit_time_s"], errors="coerce")
    df = df.dropna(subset=["RMSEP"])
    return df


def _load_aomridge(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)].copy()
    df["rmsep"] = pd.to_numeric(df["rmsep"], errors="coerce")
    df["fit_time_s"] = pd.to_numeric(df["fit_time_s"], errors="coerce")
    df = df.dropna(subset=["rmsep"])
    return df


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

_AOMPLS_PREFIXES = ("AOM", "POP", "HPO-AOM", "SPXY-AOM")


def _top_variants_aompls(df: pd.DataFrame, datasets: set[str], k: int = 2) -> list[str]:
    """Return the k AOM-PLS variant labels with the lowest median RMSEP
    across the supplied dataset set. Only true AOM/POP family variants
    are considered (PLS-standard-numpy is excluded).
    """
    sub = df[df["aom_variant"].str.startswith(_AOMPLS_PREFIXES, na=False)]
    sub = sub[sub["dataset"].isin(datasets)]
    pivot = sub.pivot_table(index="dataset", columns="aom_variant",
                             values="RMSEP", aggfunc="first")
    # Require a variant to be present on >= 80% of the inner-join datasets
    coverage = pivot.notna().sum(axis=0) / len(datasets)
    pivot = pivot.loc[:, coverage >= 0.8]
    medians = pivot.median(axis=0).sort_values()
    return list(medians.head(k).index)


def _top_variants_aomridge(df: pd.DataFrame, master: pd.DataFrame,
                            datasets: set[str], k: int = 2) -> list[str]:
    """Return the k AOM-Ridge variant labels with the highest median delta
    versus the paper Ridge HPO baseline across the supplied dataset set.
    Only ``AOMRidge-*`` variants are considered (Ridge-raw baselines are
    excluded so the picks reflect a true AOM-Ridge configuration).
    """
    sub = df[df["variant"].str.startswith("AOMRidge", na=False)]
    sub = sub[sub["dataset"].isin(datasets)]
    pivot = sub.pivot_table(index="dataset", columns="variant",
                             values="rmsep", aggfunc="first")
    coverage = pivot.notna().sum(axis=0) / len(datasets)
    pivot = pivot.loc[:, coverage >= 0.8]
    ridge_ref = master.set_index("dataset")["Ridge"]
    delta = pivot.apply(lambda col: (ridge_ref.reindex(col.index) - col)
                        / ridge_ref.reindex(col.index), axis=0)
    medians = delta.median(axis=0).sort_values(ascending=False)
    return list(medians.head(k).index)


# ---------------------------------------------------------------------------
# iRMSEP construction
# ---------------------------------------------------------------------------

def build_irmsep_table(
    master: pd.DataFrame,
    cohort: pd.DataFrame,
    aompls: pd.DataFrame,
    aomridge: pd.DataFrame,
    aompls_top: list[str],
    aomridge_top: list[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, float | None]]:
    """Assemble a wide-format iRMSEP table indexed by dataset.

    Returns
    -------
    irmsep : DataFrame indexed by dataset, columns = {n_train, n_features,
              database_name, plus one column per displayed method holding
              the iRMSEP value}.
    method_to_label : mapping from the internal method key (e.g.
        ``AOM-PLS-1``) to its human-readable variant label.
    method_fit_time : mapping from method key to the median fit_time_s
        across the inner-join datasets, when available.
    """
    # 1) Datasets in the inner join of all four sources.
    aompls_dsets = set(aompls["dataset"].unique())
    aomridge_dsets = set(aomridge["dataset"].unique())
    master_dsets = set(master["dataset"].unique())
    cohort_dsets = set(cohort["dataset"].unique())
    inner = aompls_dsets & aomridge_dsets & master_dsets & cohort_dsets
    inner -= EXCLUDED_DATASETS
    if not inner:
        raise RuntimeError("Empty inner join across the four input sources.")

    # 2) Restrict everything to the inner join.
    master_ij = master[master["dataset"].isin(inner)].copy()
    cohort_ij = cohort[cohort["dataset"].isin(inner)].copy()
    aompls_ij = aompls[aompls["dataset"].isin(inner)].copy()
    aomridge_ij = aomridge[aomridge["dataset"].isin(inner)].copy()

    # 3) Reference TabPFN-Raw RMSEP per dataset.
    tab_raw = master_ij.set_index("dataset")["TabPFN-Raw"].astype(float)

    # 4) Build a per-method RMSEP map.
    method_rmsep: dict[str, pd.Series] = {}
    method_label: dict[str, str] = {}

    # Paper baselines (from master_pivot).
    for col in ("TabPFN-opt", "Ridge", "PLS", "Catboost"):
        method_rmsep[col] = master_ij.set_index("dataset")[col].astype(float)
        method_label[col] = col
    method_rmsep["TabPFN-Raw"] = tab_raw.copy()
    method_label["TabPFN-Raw"] = "TabPFN-Raw"

    # AOM-PLS top variants.
    aompls_long = aompls_ij[aompls_ij["aom_variant"].isin(aompls_top)]
    aompls_pivot = aompls_long.pivot_table(index="dataset", columns="aom_variant",
                                           values="RMSEP", aggfunc="first")
    for rank, variant in enumerate(aompls_top, start=1):
        key = f"AOM-PLS-{rank}"
        method_rmsep[key] = aompls_pivot.get(variant, pd.Series(dtype=float))
        method_label[key] = variant

    # AOM-Ridge top variants.
    aomridge_long = aomridge_ij[aomridge_ij["variant"].isin(aomridge_top)]
    aomridge_pivot = aomridge_long.pivot_table(index="dataset", columns="variant",
                                               values="rmsep", aggfunc="first")
    for rank, variant in enumerate(aomridge_top, start=1):
        key = f"AOM-Ridge-{rank}"
        method_rmsep[key] = aomridge_pivot.get(variant, pd.Series(dtype=float))
        method_label[key] = variant

    # 5) Compute iRMSEP per method.
    irmsep_data: dict[str, pd.Series] = {}
    for key, rmsep in method_rmsep.items():
        rmsep = rmsep.reindex(sorted(inner))
        irmsep_data[key] = (tab_raw.reindex(sorted(inner)) - rmsep) / tab_raw.reindex(sorted(inner))

    irmsep = pd.DataFrame(irmsep_data)
    cohort_ij = cohort_ij.set_index("dataset").loc[sorted(inner)]
    irmsep["database_name"] = cohort_ij["database_name"]
    irmsep["n_train"] = cohort_ij["n_train"]
    irmsep["n_features"] = cohort_ij["n_features"]
    irmsep.index.name = "dataset"

    # 6) Per-method median fit_time_s where available.
    method_fit_time: dict[str, float | None] = {
        "TabPFN-Raw": None,
        "TabPFN-opt": None,
        "Ridge": None,
        "PLS": None,
        "Catboost": None,
    }
    aompls_fit = aompls_ij.set_index(["dataset", "aom_variant"])["fit_time_s"]
    for rank, variant in enumerate(aompls_top, start=1):
        key = f"AOM-PLS-{rank}"
        try:
            ts = aompls_fit.xs(variant, level="aom_variant")
            method_fit_time[key] = float(ts.median()) if len(ts) else None
        except KeyError:
            method_fit_time[key] = None
    aomridge_fit = aomridge_ij.set_index(["dataset", "variant"])["fit_time_s"]
    for rank, variant in enumerate(aomridge_top, start=1):
        key = f"AOM-Ridge-{rank}"
        try:
            ts = aomridge_fit.xs(variant, level="variant")
            method_fit_time[key] = float(ts.median()) if len(ts) else None
        except KeyError:
            method_fit_time[key] = None

    return irmsep, method_label, method_fit_time


# ---------------------------------------------------------------------------
# Figure 1 - cumulative iRMSEP
# ---------------------------------------------------------------------------

DISPLAY_ORDER = [
    # Baseline first (zero reference line)
    "TabPFN-Raw",
    "TabPFN-opt",
    "Ridge",
    "PLS",
    "Catboost",
    "AOM-PLS-1",
    "AOM-PLS-2",
    "AOM-Ridge-1",
    "AOM-Ridge-2",
]


def _cumulative_for_axis(ax, irmsep: pd.DataFrame, sort_col: str,
                         method_label: Mapping[str, str], title: str) -> None:
    df = irmsep.sort_values(sort_col, kind="mergesort").copy()
    n = len(df)
    x = np.arange(1, n + 1)
    twin_x = df[sort_col].to_numpy()

    # Baseline (zero) line for TabPFN-Raw.
    style = METHOD_STYLE["TabPFN-Raw"]
    ax.axhline(0.0, color=style["color"], linestyle=style["linestyle"],
               linewidth=style["linewidth"], alpha=0.8, zorder=1,
               label=f"TabPFN-Raw (n={n})")

    # Plot each non-baseline method.
    endpoints: list[tuple[str, float]] = []
    for key in DISPLAY_ORDER:
        if key == "TabPFN-Raw":
            continue
        if key not in df.columns:
            continue
        series = df[key]
        # NaNs contribute zero to the cumulative sum but the marker is omitted.
        cum = series.fillna(0.0).cumsum().to_numpy()
        valid = series.notna().to_numpy()
        style = METHOD_STYLE[key]
        legend_label = method_label.get(key, key)
        if key.startswith("AOM-"):
            legend_label = f"{key}: {legend_label}"
        ax.plot(x, cum, color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], marker=style["marker"],
                markersize=4 if key.startswith("AOM-") else 3,
                markevery=max(1, n // 12), alpha=0.95,
                label=legend_label, zorder=3 if key.startswith("AOM-") else 2)
        ax.plot(x[valid], cum[valid], color=style["color"], linestyle="None",
                marker=style["marker"], markersize=2.5, alpha=0.4, zorder=2)
        endpoints.append((key, float(cum[-1])))

    # Endpoint annotations to the right of the plot.
    if endpoints:
        endpoints.sort(key=lambda kv: kv[1], reverse=True)
        for key, val in endpoints:
            style = METHOD_STYLE[key]
            ax.annotate(f"{val:+.2f}", xy=(n, val), xytext=(6, 0),
                        textcoords="offset points",
                        color=style["color"],
                        fontsize=7,
                        fontweight="bold" if key.startswith("AOM-") else "normal",
                        va="center", ha="left")

    # Twin x-axis to show actual sort_col values for context.
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel(f"Dataset rank (sorted by {sort_col} ascending)")
    ax.set_ylabel("Cumulative iRMSEP vs TabPFN-Raw")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", framealpha=0.9, ncol=1)

    # Add a thin secondary axis showing min/median/max of sort_col for context.
    if n >= 3:
        sec = ax.secondary_xaxis("top")
        ticks = [1, n // 2, n]
        sec.set_xticks(ticks)
        sec.set_xticklabels([f"{int(twin_x[t-1])}" for t in ticks])
        sec.set_xlabel(f"{sort_col} (min / median / max)")


def fig_cumulative_irmsep(irmsep: pd.DataFrame,
                           method_label: Mapping[str, str],
                           out_stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    _cumulative_for_axis(axes[0], irmsep, "n_features", method_label,
                         "Cumulative iRMSEP vs TabPFN-Raw, sorted by n_features")
    _cumulative_for_axis(axes[1], irmsep, "n_train", method_label,
                         "Cumulative iRMSEP vs TabPFN-Raw, sorted by n_train")
    fig.tight_layout()
    return _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Figure 2 - mean iRMSEP vs training time (Pareto)
# ---------------------------------------------------------------------------

# Approximate / annotated fit times for paper baselines (seconds).
# These are not rigorously measured here — they are placeholders shown
# as an annotated band to keep the methods on a comparable axis.
APPROX_FIT_TIMES: dict[str, float] = {
    "TabPFN-Raw": 10.0,    # amortized inference cost
    "TabPFN-opt": 60.0,    # tuned variant
    "Ridge": 1.0,          # paper Ridge HPO is fast on these sizes
    "PLS": 0.5,
    "Catboost": 30.0,
}


def _pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    """Return the set of method keys on the Pareto front (lower fit time
    AND higher iRMSEP wins). points = [(key, x_time, y_irmsep), ...].
    """
    front: set[str] = set()
    for key_i, x_i, y_i in points:
        dominated = False
        for key_j, x_j, y_j in points:
            if key_i == key_j:
                continue
            # j dominates i iff j is no worse on both and strictly better on one.
            if x_j <= x_i and y_j >= y_i and (x_j < x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            front.add(key_i)
    return front


def fig_irmsep_vs_time(irmsep: pd.DataFrame,
                        method_label: Mapping[str, str],
                        method_fit_time: Mapping[str, float | None],
                        out_stem: Path) -> list[Path]:
    method_keys = [k for k in DISPLAY_ORDER if k in irmsep.columns]
    points: list[tuple[str, float, float, int]] = []  # (key, t, mean_irmsep, win_count)
    for key in method_keys:
        s = irmsep[key].dropna()
        if s.empty:
            continue
        mean_ir = float(s.mean())
        win = int((s > 0).sum())
        t = method_fit_time.get(key) or APPROX_FIT_TIMES.get(key)
        if t is None or t <= 0:
            t = 1.0
        points.append((key, t, mean_ir, win))

    if not points:
        raise RuntimeError("No methods available for the Pareto figure.")

    # Compute the Pareto front (minimize log-time, maximize mean iRMSEP).
    front = _pareto_front([(k, np.log10(t), y) for k, t, y, _ in points])

    fig, ax = plt.subplots(figsize=(10, 6))
    # Marker size scaled by win count (range [40, 320]).
    win_arr = np.array([p[3] for p in points], dtype=float)
    if win_arr.max() > 0:
        sizes = 40 + (win_arr / win_arr.max()) * 280
    else:
        sizes = np.full_like(win_arr, 80.0)

    log_times = np.array([np.log10(p[1]) for p in points])
    y_vals = np.array([p[2] for p in points])

    # Markers that cannot have a separate edge color (e.g. 'x', '+', '|', '_').
    UNFILLED_MARKERS = {"x", "+", "1", "2", "3", "4", "|", "_"}
    for (key, t, y, win), size in zip(points, sizes, strict=True):
        style = METHOD_STYLE[key]
        is_approx = method_fit_time.get(key) is None
        marker = style["marker"]
        if marker in UNFILLED_MARKERS:
            ax.scatter(np.log10(t), y, s=size, marker=marker,
                       color=style["color"], linewidths=1.6,
                       zorder=4 if key in front else 3, alpha=0.9)
        else:
            face = style["color"] if not is_approx else "none"
            ax.scatter(np.log10(t), y, s=size, marker=marker,
                       facecolors=face, edgecolors=style["color"], linewidths=1.6,
                       zorder=4 if key in front else 3, alpha=0.9)
        label = method_label.get(key, key)
        if key.startswith("AOM-"):
            label = f"{key}: {label}"
        suffix = " (approx. time)" if is_approx else ""
        ax.annotate(f"{label}{suffix}\n(wins={win})",
                    xy=(np.log10(t), y), xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=style["color"],
                    fontweight="bold" if key.startswith("AOM-") else "normal")

    # Pareto front polyline (sorted by x).
    front_pts = sorted(((np.log10(t), y) for k, t, y, _ in points if k in front),
                       key=lambda xy: xy[0])
    if len(front_pts) >= 2:
        fx, fy = zip(*front_pts, strict=True)
        ax.plot(fx, fy, linestyle="--", color="#444444", linewidth=1.0,
                alpha=0.7, zorder=2, label="Pareto front")

    ax.axhline(0.0, color="#999999", linestyle=":", linewidth=1.0, alpha=0.7,
               label="TabPFN-Raw baseline (iRMSEP = 0)")
    ax.set_xlabel("log10(fit time, seconds)")
    ax.set_ylabel("Mean iRMSEP across datasets (vs TabPFN-Raw)")
    ax.set_title("iRMSEP vs training time (mean across datasets)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)

    # Footnote about approximated baselines.
    approx_keys = [k for k, _, _, _ in points if method_fit_time.get(k) is None]
    if approx_keys:
        text = ("Open markers: fit time approximated (not measured here). "
                f"Affected: {', '.join(approx_keys)}.")
        fig.text(0.01, -0.02, text, fontsize=7, color="#444444")

    fig.tight_layout()
    return _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_stem: Path) -> list[Path]:
    out_stem = Path(out_stem)
    out_pdf = out_stem.with_suffix(".pdf")
    out_png = out_stem.with_suffix(".png")
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)
    return [out_pdf, out_png]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER,
                        help="Path to master_pivot.csv")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT,
                        help="Path to cohort_regression.csv")
    parser.add_argument("--aompls", type=Path, default=DEFAULT_AOMPLS,
                        help="Path to AOM-PLS results.csv")
    parser.add_argument("--aomridge", type=Path, default=None,
                        help="Path to AOM-Ridge results.csv "
                             "(default: curated_v2 if present, else curated)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory for figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLT_RC)

    # Load all four sources.
    master = _load_master(args.master)
    cohort = _load_cohort(args.cohort)
    aompls = _load_aompls(args.aompls)
    ridge_path = _resolve_ridge_path(args.aomridge)
    aomridge = _load_aomridge(ridge_path)

    # Inner join.
    inner = (set(master["dataset"]) & set(cohort["dataset"])
             & set(aompls["dataset"]) & set(aomridge["dataset"])) - EXCLUDED_DATASETS
    print(f"[inner join] master ∩ cohort ∩ AOM-PLS ∩ AOM-Ridge = {len(inner)} datasets")
    print(f"             AOM-Ridge source: {ridge_path}")

    # Variant selection on the inner-join dataset set.
    aompls_top = _top_variants_aompls(aompls, inner, k=2)
    aomridge_top = _top_variants_aomridge(aomridge, master, inner, k=2)
    print("[selected variants]")
    for rank, label in enumerate(aompls_top, start=1):
        print(f"  AOM-PLS Top-{rank}:   {label}")
    for rank, label in enumerate(aomridge_top, start=1):
        print(f"  AOM-Ridge Top-{rank}: {label}")

    irmsep, method_label, method_fit_time = build_irmsep_table(
        master, cohort, aompls, aomridge, aompls_top, aomridge_top
    )

    # Figure 1
    fig1_paths = fig_cumulative_irmsep(irmsep, method_label,
                                       out_dir / "fig_cumulative_irmsep")
    print("[wrote]", *[p.name for p in fig1_paths])

    # Figure 2
    fig2_paths = fig_irmsep_vs_time(irmsep, method_label, method_fit_time,
                                    out_dir / "fig_irmsep_vs_time")
    print("[wrote]", *[p.name for p in fig2_paths])

    return 0


if __name__ == "__main__":
    sys.exit(main())
