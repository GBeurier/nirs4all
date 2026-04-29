"""Generate publication figures for the AOM_v0 paper.

Figures produced (all PDF, saved into ``../figures/``):

- ``fig_framework.pdf``          : schematic of the operator-bank ->
                                    selection -> PLS core -> tasks flow.
- ``fig_operator_paths.pdf``     : POP vs AOM operator selection trace
                                    on a synthetic regression dataset.
- ``fig_regression_cd.pdf``      : critical-difference diagram for
                                    regression. Computed from
                                    ``benchmark_runs/.../results.csv``
                                    when available, otherwise rendered
                                    as a placeholder.
- ``fig_classification_cd.pdf``  : same for classification.
- ``fig_probability_calibration.pdf``: reliability diagram for the
                                    PLS-DA classifiers.

The script is designed to be runnable today on the smoke results;
where data is missing it produces a labelled placeholder figure rather
than failing, so that the LaTeX build always finds every referenced
PDF.

Usage::

    python bench/AOM_v0/publication/scripts/make_figures.py

Optional flags::

    --results        path to a regression results CSV (default: smoke).
    --results-class  path to a classification results CSV.
    --master         path to the TabPFN master CSV.
    --out            output directory (default: ../figures).

The script imports from the local ``aompls`` package shipped in
``bench/AOM_v0/aompls/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # bench/AOM_v0
sys.path.insert(0, str(ROOT))

from aompls.banks import compact_bank  # noqa: E402
from aompls.estimators import AOMPLSRegressor, POPPLSRegressor  # noqa: E402
from aompls.synthetic import make_classification, make_regression  # noqa: E402


DEFAULT_OUT = HERE.parent / "figures"
DEFAULT_RESULTS = ROOT / "benchmark_runs" / "smoke" / "results.csv"
DEFAULT_RESULTS_CLASS = ROOT / "benchmark_runs" / "smoke" / "results_classification.csv"
DEFAULT_MASTER = ROOT.parent / "tabpfn_paper" / "master_results.csv"


# ---------------------------------------------------------------------------
# Figure 1 - Framework schematic
# ---------------------------------------------------------------------------

def draw_box(ax, xy, w, h, text, fc="white", ec="black", fontsize=10):
    rect = plt.Rectangle(xy, w, h, fill=True, fc=fc, ec=ec, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", linewidth=1.4))


def fig_framework(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Operator bank (left column)
    draw_box(ax, (0.3, 4.3), 2.6, 1.0, "Operator bank\n$\\{A_1, A_2, \\ldots, A_B\\}$",
             fc="#e8f3ff")
    draw_box(ax, (0.3, 3.0), 2.6, 1.0, "Identity, SG, FD,\nDetrend, Whittaker, NW",
             fc="#f0f0f0", fontsize=9)
    draw_box(ax, (0.3, 1.7), 2.6, 1.0, "Strict linear,\n$X_b = X A_b^\\top$",
             fc="#f7f7f7", fontsize=9)

    # Selection policy (middle column)
    draw_box(ax, (3.6, 4.3), 3.2, 1.0, "Selection policy\nnone / global / per-comp / soft / superblock",
             fc="#fff5e6", fontsize=9)
    draw_box(ax, (3.6, 3.0), 3.2, 1.0, "Criterion\ncov / CV / aPRESS / hybrid / holdout",
             fc="#fff5e6", fontsize=9)
    draw_box(ax, (3.6, 1.7), 3.2, 1.0, "Identity\n$(X A^\\top)^\\top Y = A X^\\top Y$",
             fc="#fff0db", fontsize=9)

    # PLS core (right-middle column)
    draw_box(ax, (7.6, 4.3), 2.4, 1.0, "PLS core\nNIPALS / SIMPLS",
             fc="#e6f5e6")
    draw_box(ax, (7.6, 3.0), 2.4, 1.0, "Materialized\nor covariance",
             fc="#e6f5e6", fontsize=9)
    draw_box(ax, (7.6, 1.7), 2.4, 1.0, "Effective weights\n$Z = \\sum_a A^{(a)\\top} r_a$",
             fc="#e6f5e6", fontsize=9)

    # Tasks (right)
    draw_box(ax, (10.4, 4.3), 1.4, 1.0, "Regression\n$\\hat Y$",
             fc="#f5e6f5", fontsize=10)
    draw_box(ax, (10.4, 3.0), 1.4, 1.0, "PLS-DA\n$\\hat p(c)$",
             fc="#f5e6f5", fontsize=10)
    draw_box(ax, (10.4, 1.7), 1.4, 1.0, "Diagnostics\nop. paths",
             fc="#f5e6f5", fontsize=9)

    # Arrows between columns
    for y in (4.8, 3.5, 2.2):
        draw_arrow(ax, 2.95, y, 3.55, y)
        draw_arrow(ax, 6.85, y, 7.55, y)
        draw_arrow(ax, 10.05, y, 10.35, y)

    ax.set_title("Operator-Adaptive PLS framework", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 - POP vs AOM operator paths on a synthetic dataset
# ---------------------------------------------------------------------------

def fig_operator_paths(out_path: Path) -> None:
    ds = make_regression(n_train=120, n_test=60, p=180, noise=0.03, random_state=0)
    bank = compact_bank(p=ds.X_train.shape[1])
    bank_names = [op.name for op in bank]

    aom = AOMPLSRegressor(
        n_components="auto",
        max_components=10,
        engine="simpls_covariance",
        selection="global",
        criterion="covariance",
        operator_bank="compact",
        cv=5,
        random_state=0,
    ).fit(ds.X_train, ds.y_train)
    pop = POPPLSRegressor(
        n_components="auto",
        max_components=10,
        engine="simpls_covariance",
        selection="per_component",
        criterion="covariance",
        operator_bank="compact",
        cv=5,
        random_state=0,
    ).fit(ds.X_train, ds.y_train)

    aom_diag = aom.get_diagnostics()
    pop_diag = pop.get_diagnostics()
    aom_path = list(aom_diag.get("selected_operator_indices", []))
    pop_path = list(pop_diag.get("selected_operator_indices", []))

    K = max(len(aom_path), len(pop_path), 1)
    if not aom_path:
        aom_path = [0] * K
    if not pop_path:
        pop_path = [0] * K
    components = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(components, aom_path[:K], marker="o", linewidth=2,
            label="AOM (global)", color="#1f77b4")
    ax.plot(components, pop_path[:K], marker="s", linewidth=2,
            label="POP (per-component)", color="#d62728")
    ax.set_xticks(components)
    ax.set_yticks(np.arange(len(bank_names)))
    ax.set_yticklabels(bank_names, fontsize=9)
    ax.set_xlabel("Latent component")
    ax.set_ylabel("Selected operator")
    ax.set_title("AOM vs POP operator selection on synthetic data")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 / 4 - Critical difference diagrams
# ---------------------------------------------------------------------------

def _per_dataset_ranks(df: pd.DataFrame, key_col: str, score_col: str,
                       higher_is_better: bool) -> pd.DataFrame:
    """Return a wide table dataset x variant of average ranks."""
    pivot = df.pivot_table(
        index=["database_name", "dataset"],
        columns=key_col,
        values=score_col,
        aggfunc="mean",
    )
    if pivot.empty:
        return pivot
    if higher_is_better:
        ranked = pivot.rank(axis=1, method="average", ascending=False)
    else:
        ranked = pivot.rank(axis=1, method="average", ascending=True)
    return ranked


def _critical_difference_plot(ax, ranks_df: pd.DataFrame, title: str) -> None:
    if ranks_df.empty:
        ax.text(0.5, 0.5, "No data available\n(placeholder).",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        ax.axis("off")
        ax.set_title(title)
        return
    avg = ranks_df.mean(axis=0).sort_values()
    methods = list(avg.index)
    ranks = avg.values
    n_methods = len(methods)
    n_data = ranks_df.shape[0]

    # Approximate Nemenyi critical difference at alpha=0.05
    # CD = q_{alpha,k} * sqrt(k(k+1)/(6N))
    # For up to 12 methods we use a small lookup; otherwise default to a
    # broad value to avoid pretending precision.
    qalpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
              7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
              11: 3.219, 12: 3.268}
    q = qalpha.get(n_methods, 3.4)
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * max(n_data, 1)))

    rmin = max(1, np.floor(ranks.min()) - 0.5)
    rmax = min(n_methods, np.ceil(ranks.max()) + 0.5)
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for r, m in zip(ranks, methods):
        ax.plot([r, r], [0.55, 0.7], color="black", linewidth=1.2)
        ax.text(r, 0.78, f"{m}\n({r:.2f})",
                ha="center", va="bottom", rotation=0, fontsize=8)
    ax.plot([rmin, rmax], [0.55, 0.55], color="black", linewidth=1.5)
    # CD bar
    cd_left = rmin + 0.1
    ax.plot([cd_left, cd_left + cd], [0.32, 0.32], color="black", linewidth=2)
    ax.text(cd_left + cd / 2, 0.18, f"CD = {cd:.3f}",
            ha="center", va="center", fontsize=10)
    ax.set_title(title)


def fig_regression_cd(out_path: Path, results_csv: Path,
                       master_csv: Optional[Path]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.0))
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df = df[df["status"] == "ok"]
        df["RMSEP"] = pd.to_numeric(df["RMSEP"], errors="coerce")
        df = df.dropna(subset=["RMSEP"])
        if master_csv is not None and master_csv.exists():
            master = pd.read_csv(master_csv)
            ref_pls = master[master["model"] == "PLS"][
                ["database_name", "dataset", "RMSEP"]
            ].copy()
            ref_pls["aom_variant"] = "PLS-master"
            ref_tab_raw = master[master["model"] == "TabPFN-Raw"][
                ["database_name", "dataset", "RMSEP"]
            ].copy()
            ref_tab_raw["aom_variant"] = "TabPFN-Raw"
            ref_tab_opt = master[master["model"] == "TabPFN-opt"][
                ["database_name", "dataset", "RMSEP"]
            ].copy()
            ref_tab_opt["aom_variant"] = "TabPFN-opt"
            ref = pd.concat([ref_pls, ref_tab_raw, ref_tab_opt], ignore_index=True)
            df_aom = df[["database_name", "dataset", "aom_variant", "RMSEP"]]
            df = pd.concat([df_aom, ref], ignore_index=True)
        ranks = _per_dataset_ranks(df, "aom_variant", "RMSEP", higher_is_better=False)
    else:
        ranks = pd.DataFrame()
    _critical_difference_plot(ax, ranks, "Regression: critical-difference diagram (RMSEP)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def fig_classification_cd(out_path: Path, results_csv: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.0))
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df = df[df["status"] == "ok"]
        df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
        df = df.dropna(subset=["balanced_accuracy"])
        ranks = _per_dataset_ranks(df, "aom_variant", "balanced_accuracy",
                                    higher_is_better=True)
    else:
        ranks = pd.DataFrame()
    _critical_difference_plot(ax, ranks, "Classification: critical-difference diagram (balanced accuracy)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 - Reliability diagram
# ---------------------------------------------------------------------------

def fig_probability_calibration(out_path: Path) -> None:
    """Reliability diagram from a synthetic classification run."""
    from aompls.classification import AOMPLSDAClassifier, POPPLSDAClassifier

    ds = make_classification(n_train=180, n_test=120, p=160, n_classes=3,
                              noise=0.03, random_state=0)
    aom = AOMPLSDAClassifier(
        n_components="auto", max_components=8,
        engine="simpls_covariance", selection="global",
        criterion="covariance", operator_bank="compact",
        cv=3, random_state=0,
    ).fit(ds.X_train, ds.y_train)
    pop = POPPLSDAClassifier(
        n_components="auto", max_components=8,
        engine="simpls_covariance", selection="per_component",
        criterion="covariance", operator_bank="compact",
        cv=3, random_state=0,
    ).fit(ds.X_train, ds.y_train)

    proba_aom = aom.predict_proba(ds.X_test)
    proba_pop = pop.predict_proba(ds.X_test)
    classes = aom.classes_
    y_true = ds.y_test

    def reliability_curve(y_true, proba, classes, n_bins=10):
        # Use top-class confidence as the calibration variable
        conf = proba.max(axis=1)
        pred = classes[proba.argmax(axis=1)]
        correct = (pred == y_true).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(conf, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        x, y = [], []
        for k in range(n_bins):
            mask = idx == k
            if mask.sum() == 0:
                continue
            x.append(conf[mask].mean())
            y.append(correct[mask].mean())
        return np.array(x), np.array(y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="ideal")
    x_a, y_a = reliability_curve(y_true, proba_aom, classes)
    x_p, y_p = reliability_curve(y_true, proba_pop, classes)
    if len(x_a) > 0:
        ax.plot(x_a, y_a, marker="o", linewidth=2, color="#1f77b4",
                label="AOM-PLSDA (global)")
    if len(x_p) > 0:
        ax.plot(x_p, y_p, marker="s", linewidth=2, color="#d62728",
                label="POP-PLSDA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (top-class probability)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram on synthetic classification data")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--results-class", type=Path, default=DEFAULT_RESULTS_CLASS)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"[make_figures] output dir: {out}")

    fig_framework(out / "fig_framework.pdf")
    print("  - fig_framework.pdf")

    fig_operator_paths(out / "fig_operator_paths.pdf")
    print("  - fig_operator_paths.pdf")

    fig_regression_cd(out / "fig_regression_cd.pdf",
                       args.results, args.master)
    print("  - fig_regression_cd.pdf")

    fig_classification_cd(out / "fig_classification_cd.pdf",
                           args.results_class)
    print("  - fig_classification_cd.pdf")

    fig_probability_calibration(out / "fig_probability_calibration.pdf")
    print("  - fig_probability_calibration.pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
