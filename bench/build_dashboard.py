"""Generate a self-contained interactive dashboard for the benchmark master CSV.

Reads `benchmark_master_results.csv`, distils it into a compact JSON payload of
aggregations (oracle table, leaderboards, heatmap, pareto, wins) and emits a
single-file HTML dashboard at `figures/benchmark_master/dashboard.html`.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parent
CSV = BENCH / "benchmark_master_results.csv"
OUT_DIR = BENCH / "figures" / "benchmark_master"
TEMPLATE = BENCH / "dashboard_template.html"
OUT_HTML = OUT_DIR / "dashboard.html"
OUT_JSON = OUT_DIR / "dashboard_data.json"

MIN_DATASETS_FOR_LEADERBOARD = 5
HEATMAP_TOP_K = 15
PARETO_MIN_DATASETS = 8


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


def median_iqr(values: pd.Series) -> dict:
    arr = values.dropna().to_numpy()
    if arr.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_oracle_by_class(df: pd.DataFrame) -> list[dict]:
    oracle = df[
        (df["record_type"] == "oracle_by_model_class") & (df["score_metric"] == "rmsep")
    ].copy()
    oracle["rel"] = pd.to_numeric(oracle["score_ratio_vs_dataset_pls"], errors="coerce")
    grouped = oracle.dropna(subset=["rel"]).groupby("model_class")
    out = []
    for cls, part in grouped:
        rels = part["rel"].to_numpy()
        wins = int((rels < 1.0).sum())
        n = int(rels.size)
        out.append(
            {
                "model_class": cls,
                "n_datasets": n,
                "median_rel": float(np.median(rels)),
                "q25": float(np.quantile(rels, 0.25)),
                "q75": float(np.quantile(rels, 0.75)),
                "wins": wins,
                "win_rate": wins / n if n else 0.0,
            }
        )
    out.sort(key=lambda r: r["median_rel"])
    return out


def build_leaderboards(rows: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Best per (model, dataset). Aggregate to one row per model with median/IQR/wins."""

    def agg(rows_subset: pd.DataFrame, ratio_col: str) -> list[dict]:
        best = best_per_model_dataset(rows_subset, ratio_col)
        recs: list[dict] = []
        for (cls, name), part in best.groupby(["model_class", "model_name"]):
            stats = median_iqr(part[ratio_col])
            if stats["n"] < MIN_DATASETS_FOR_LEADERBOARD:
                continue
            wins = int((part[ratio_col] < 1.0).sum())
            ft = pd.to_numeric(part["fit_time_s"], errors="coerce").dropna()
            rec = {
                "model_class": cls,
                "model_name": name,
                "n_datasets": stats["n"],
                "median": stats["median"],
                "q25": stats["q25"],
                "q75": stats["q75"],
                "min": stats["min"],
                "max": stats["max"],
                "wins": wins,
                "win_rate": wins / stats["n"],
                "median_fit_time": float(ft.median()) if not ft.empty else None,
            }
            recs.append(rec)
        recs.sort(key=lambda r: r["median"])
        return recs

    return agg(rows, "score_ratio_vs_source_run_pls"), agg(rows, "score_ratio_vs_dataset_pls")


def build_heatmap(rows: pd.DataFrame, ratio_col: str = "score_ratio_vs_source_run_pls") -> dict:
    best = best_per_model_dataset(rows, ratio_col)
    qualifying = (
        best.groupby("model_name")
        .agg(median_rel=(ratio_col, "median"), n=(ratio_col, "count"))
        .query(f"n >= {MIN_DATASETS_FOR_LEADERBOARD}")
        .sort_values("median_rel")
        .head(HEATMAP_TOP_K)
        .index.tolist()
    )
    sub = best[best["model_name"].isin(qualifying)]
    pivot = sub.pivot_table(
        index="model_name", columns="dataset", values=ratio_col, aggfunc="min"
    ).reindex(qualifying)

    # Order datasets by overall median rel (left = where everyone struggles)
    dataset_order = (
        sub.groupby("dataset")[ratio_col]
        .median()
        .sort_values()
        .index.tolist()
    )
    pivot = pivot[dataset_order]

    model_class_lookup = (
        sub.drop_duplicates("model_name").set_index("model_name")["model_class"].to_dict()
    )

    log2 = np.log2(pivot.to_numpy().astype(float))
    log2_clipped = np.clip(log2, -1.0, 1.0)
    # Replace nan with None for JSON
    z = [[None if math.isnan(v) else float(v) for v in row] for row in log2_clipped]
    raw_ratios = [
        [None if pd.isna(v) else float(v) for v in row] for row in pivot.to_numpy()
    ]

    return {
        "models": list(pivot.index),
        "model_classes": [model_class_lookup.get(m, "") for m in pivot.index],
        "datasets": list(pivot.columns),
        "z_log2": z,
        "ratios": raw_ratios,
    }


def build_pareto(rows: pd.DataFrame) -> list[dict]:
    best = best_per_model_dataset(rows, "score_ratio_vs_source_run_pls")
    out: list[dict] = []
    for (cls, name), part in best.groupby(["model_class", "model_name"]):
        if part["dataset"].nunique() < PARETO_MIN_DATASETS:
            continue
        ft = pd.to_numeric(part["fit_time_s"], errors="coerce").dropna()
        if ft.empty:
            continue
        out.append(
            {
                "model_class": cls,
                "model_name": name,
                "n_datasets": int(part["dataset"].nunique()),
                "median_rel": float(part["score_ratio_vs_source_run_pls"].median()),
                "median_fit_time": float(ft.median()),
                "wins": int((part["score_ratio_vs_source_run_pls"] < 1.0).sum()),
            }
        )
    out.sort(key=lambda r: r["median_rel"])
    return out


def build_meta(df: pd.DataFrame, rows: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "n_rows": int(len(df)),
        "n_observed": int((df["record_type"] == "observed").sum()),
        "n_reference": int((df["record_type"] == "reference_paper").sum()),
        "n_oracle_class": int((df["record_type"] == "oracle_by_model_class").sum()),
        "n_oracle_global": int((df["record_type"] == "oracle_global_dataset").sum()),
        "n_datasets": int(rows["dataset"].nunique()),
        "n_models": int(rows["model_name"].nunique()),
        "n_model_classes": int(rows["model_class"].nunique()),
        "source_families": sorted(rows["source_family"].dropna().unique().tolist()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV, low_memory=False)
    rows = source_rows(df)
    payload = {
        "meta": build_meta(df, rows),
        "oracle_by_class": build_oracle_by_class(df),
        "leaderboard_protocol": build_leaderboards(rows)[0],
        "leaderboard_global": build_leaderboards(rows)[1],
        "heatmap": build_heatmap(rows),
        "pareto": build_pareto(rows),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size / 1024:.1f} KB)")

    template = TEMPLATE.read_text()
    embedded_json = json.dumps(payload, separators=(",", ":"))
    html = template.replace("__DASHBOARD_DATA__", embedded_json)
    OUT_HTML.write_text(html)
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
