from __future__ import annotations

from pathlib import Path

import pandas as pd


BENCH = Path(__file__).resolve().parent
CSV = BENCH / "benchmark_master_results.csv"
OUT = BENCH / "figures" / "benchmark_master"


def require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def source_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[
        df["record_type"].isin(["observed", "reference_paper"])
        & (df["task"] == "regression")
        & (df["score_metric"] == "rmsep")
        & (df["status"].fillna("").str.lower().isin(["ok", ""]))
    ].copy()
    for col in [
        "score_ratio_vs_source_run_pls",
        "score_ratio_vs_dataset_pls",
        "fit_time_s",
        "rmsep",
    ]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def best_per_model_dataset(rows: pd.DataFrame, ratio_col: str) -> pd.DataFrame:
    tmp = rows.dropna(subset=[ratio_col]).copy()
    tmp = tmp.sort_values(ratio_col, ascending=True)
    return tmp.drop_duplicates(["model_class", "model_name", "dataset"], keep="first")


def plot_oracle_by_class(df: pd.DataFrame) -> None:
    plt = require_matplotlib()
    oracle = df[
        (df["record_type"] == "oracle_by_model_class")
        & (df["score_metric"] == "rmsep")
    ].copy()
    oracle["rel"] = pd.to_numeric(oracle["score_ratio_vs_dataset_pls"], errors="coerce")
    summary = (
        oracle.dropna(subset=["rel"])
        .groupby("model_class", as_index=False)
        .agg(median_rel=("rel", "median"), n_datasets=("dataset", "nunique"))
        .query("n_datasets >= 5")
        .sort_values("median_rel")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(summary["model_class"], summary["median_rel"], color="#4c78a8")
    ax.axvline(1.0, color="black", linewidth=1)
    ax.set_xlabel("Median RMSEP / best observed PLS")
    ax.set_title("Oracle by model class")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "oracle_by_model_class.png", dpi=180)
    plt.close(fig)


def plot_protocol_leaderboard(rows: pd.DataFrame) -> None:
    plt = require_matplotlib()
    best = best_per_model_dataset(rows, "score_ratio_vs_source_run_pls")
    summary = (
        best.groupby(["model_class", "model_name"], as_index=False)
        .agg(
            median_rel=("score_ratio_vs_source_run_pls", "median"),
            q25=("score_ratio_vs_source_run_pls", lambda x: x.quantile(0.25)),
            q75=("score_ratio_vs_source_run_pls", lambda x: x.quantile(0.75)),
            n_datasets=("dataset", "nunique"),
        )
        .query("n_datasets >= 10")
        .sort_values("median_rel")
        .head(25)
    )
    labels = summary["model_name"].str.slice(0, 45)
    xerr = [
        summary["median_rel"] - summary["q25"],
        summary["q75"] - summary["median_rel"],
    ]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.errorbar(summary["median_rel"], labels, xerr=xerr, fmt="o", color="#2a6f97")
    ax.axvline(1.0, color="black", linewidth=1)
    ax.set_xlabel("Median RMSEP / source-run PLS, with IQR")
    ax.set_title("Protocol-local leaderboard")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "protocol_local_leaderboard.png", dpi=180)
    plt.close(fig)


def plot_heatmap(rows: pd.DataFrame) -> None:
    plt = require_matplotlib()
    best = best_per_model_dataset(rows, "score_ratio_vs_source_run_pls")
    top_models = (
        best.groupby("model_name")["score_ratio_vs_source_run_pls"]
        .median()
        .sort_values()
        .head(15)
        .index
    )
    pivot = (
        best[best["model_name"].isin(top_models)]
        .pivot_table(
            index="model_name",
            columns="dataset",
            values="score_ratio_vs_source_run_pls",
            aggfunc="min",
        )
        .loc[top_models]
    )
    pivot = pivot.clip(lower=0.5, upper=1.5)
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([name[:45] for name in pivot.index], fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=5)
    ax.set_title("Top models x datasets, RMSEP / source-run PLS")
    fig.colorbar(im, ax=ax, label="ratio, clipped to [0.5, 1.5]")
    fig.tight_layout()
    fig.savefig(OUT / "top_model_dataset_heatmap.png", dpi=180)
    plt.close(fig)


def plot_runtime_pareto(rows: pd.DataFrame) -> None:
    plt = require_matplotlib()
    best = best_per_model_dataset(rows, "score_ratio_vs_source_run_pls")
    summary = (
        best.groupby(["model_class", "model_name"], as_index=False)
        .agg(
            median_rel=("score_ratio_vs_source_run_pls", "median"),
            median_fit_time=("fit_time_s", "median"),
            n_datasets=("dataset", "nunique"),
        )
        .query("n_datasets >= 10")
        .dropna(subset=["median_rel", "median_fit_time"])
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_class, part in summary.groupby("model_class"):
        ax.scatter(
            part["median_fit_time"].clip(lower=1e-3),
            part["median_rel"],
            s=part["n_datasets"] * 3,
            alpha=0.7,
            label=model_class,
        )
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Median fit time (s, log scale)")
    ax.set_ylabel("Median RMSEP / source-run PLS")
    ax.set_title("Runtime vs accuracy Pareto view")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "runtime_accuracy_pareto.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV, low_memory=False)
    rows = source_rows(df)
    plot_oracle_by_class(df)
    plot_protocol_leaderboard(rows)
    plot_heatmap(rows)
    plot_runtime_pareto(rows)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
