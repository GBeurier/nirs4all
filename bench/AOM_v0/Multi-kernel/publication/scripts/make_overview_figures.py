"""Overview figures comparing our multi-kernel variants to ALL
TabPFN-paper baselines: PLS, Ridge, TabPFN-raw, TabPFN-opt, CNN-NICON,
CatBoost.

Inputs: iter8 + iter12 results CSVs.
Outputs: 4 figures in publication/figures/fig_overview_*.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("bench/AOM_v0/Multi-kernel")
ITER8 = ROOT / "benchmark_runs/iter8_full54_champions/results.csv"
ITER12 = ROOT / "benchmark_runs/iter12_sparse2_default/results.csv"
FIG_DIR = ROOT / "publication/figures"

# ---- Load and merge ----
df8 = pd.read_csv(ITER8)
df12 = pd.read_csv(ITER12)
ok = pd.concat([df8, df12], ignore_index=True)
ok = ok[(ok.status == "ok") & (ok.dataset != "Quartz_spxy70")]

OURS = {
    "mkR-default-sparse3": "mkR-softmax_cv-default-active15-sparse3",
    "MKM-reml-asls": "MKM-reml-asls-default-active15",
    "mkR-asls-sparse2": "mkR-softmax_cv-asls-default-active15-sparse2",
    "mkR-asls-sparse1": "mkR-softmax_cv-asls-default-active15-sparse1",
    "mkR-msc-sparse3": "mkR-softmax_cv-msc-default-active15-sparse3",
    "Ridge-raw": "Ridge-raw",
}
COLOR_OURS = {
    "mkR-default-sparse3": "#1f77b4",
    "MKM-reml-asls": "#9467bd",
    "mkR-asls-sparse2": "#d62728",
    "mkR-asls-sparse1": "#e377c2",
    "mkR-msc-sparse3": "#2ca02c",
    "Ridge-raw": "#7f7f7f",
    "Oracle (best of ours)": "#ff7f0e",
}

# Baseline columns from TabPFN paper
BASELINES = {
    "PLS": "ref_rmse_pls",
    "Ridge (paper)": "ref_rmse_ridge",
    "TabPFN-raw": "ref_rmse_tabpfn_raw",
    "TabPFN-opt": "ref_rmse_tabpfn_opt",
    "CNN-NICON": "ref_rmse_cnn",
    "CatBoost": "ref_rmse_catboost",
}
COLOR_BASE = {
    "PLS": "#aec7e8",
    "Ridge (paper)": "#c7c7c7",
    "TabPFN-raw": "#ffbb78",
    "TabPFN-opt": "#ff9896",
    "CNN-NICON": "#98df8a",
    "CatBoost": "#c5b0d5",
}

# ---- Build a long-form table: dataset, variant, rmse, group ----
rows = []
# Get one representative ok-row per dataset to read ref_rmse_* (they're constant per dataset)
for ds in sorted(ok.dataset.unique()):
    sub = ok[ok.dataset == ds]
    rep = sub.iloc[0]
    # Add baselines
    for label, col in BASELINES.items():
        if col in rep and pd.notna(rep[col]):
            rows.append({
                "dataset": ds, "variant": label, "rmse": float(rep[col]),
                "rel_pls": float(rep[col]) / float(rep["ref_rmse_pls"]),
                "group": "baseline",
            })
    # Add our variants
    for short, full in OURS.items():
        m = sub[sub.variant == full]
        if len(m):
            rmse = float(m.iloc[0].rmsep)
            rows.append({
                "dataset": ds, "variant": short, "rmse": rmse,
                "rel_pls": rmse / float(rep["ref_rmse_pls"]),
                "group": "ours",
            })
    # Add oracle (best of ours per dataset)
    ours_rows = sub[sub.variant.isin(OURS.values())]
    if len(ours_rows):
        best = ours_rows.sort_values("rmsep").iloc[0]
        rows.append({
            "dataset": ds, "variant": "Oracle (best of ours)", "rmse": float(best.rmsep),
            "rel_pls": float(best.rmsep) / float(rep["ref_rmse_pls"]),
            "group": "ours",
        })

long_df = pd.DataFrame(rows)
# Dedupe (dataset, variant) keeping the smaller rmse if duplicates
long_df = long_df.sort_values('rmse').drop_duplicates(subset=['dataset', 'variant'], keep='first')
print(f"Long-form rows: {len(long_df)}, datasets: {long_df.dataset.nunique()}")


# ---- Figure 1: Median rel-PLS bar chart ----
def fig1_median_relpls():
    medians = long_df.groupby("variant").rel_pls.agg(['median', 'count']).reset_index()
    # Order: ours first, then baselines
    order_ours = list(OURS.keys()) + ["Oracle (best of ours)"]
    order_base = list(BASELINES.keys())
    order = order_ours + order_base
    medians["order"] = medians.variant.map({v: i for i, v in enumerate(order)})
    medians = medians.dropna(subset=["order"]).sort_values("order")

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [
        COLOR_OURS[v] if v in COLOR_OURS else COLOR_BASE.get(v, "#888888")
        for v in medians.variant
    ]
    bars = ax.bar(range(len(medians)), medians['median'], color=colors, alpha=0.85)
    for i, (m, c) in enumerate(zip(medians['median'], medians['count'])):
        ax.text(i, m + 0.005, f'{m:.3f}\n(n={c})', ha='center', va='bottom', fontsize=8)
    ax.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5)
    ax.text(len(medians) - 0.5, 1.0, 'PLS parity', ha='right', va='bottom', fontsize=9, color='dimgray')
    ax.set_xticks(range(len(medians)))
    ax.set_xticklabels(medians.variant, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Median rel-RMSEP vs PLS  (lower is better)', fontsize=11)
    ax.set_title('Median performance across 50 Stage A datasets — ours vs TabPFN-paper baselines', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0.7, 1.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_overview_median_relpls.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    print(f"Saved {out}")


# ---- Figure 2: Win-rate matrix — our variants beat each baseline on N datasets ----
def fig2_win_matrix():
    pivot = long_df.pivot(index="dataset", columns="variant", values="rmse")
    rows_ours = list(OURS.keys()) + ["Oracle (best of ours)"]
    cols_base = list(BASELINES.keys())
    win_mat = np.zeros((len(rows_ours), len(cols_base)), dtype=int)
    for i, ours_v in enumerate(rows_ours):
        for j, base_v in enumerate(cols_base):
            both = pivot[[ours_v, base_v]].dropna()
            wins = (both[ours_v] < both[base_v]).sum()
            win_mat[i, j] = wins

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(win_mat, cmap="RdYlGn", vmin=0, vmax=50, aspect="auto")
    for i in range(len(rows_ours)):
        for j in range(len(cols_base)):
            n_total = int(pivot[[rows_ours[i], cols_base[j]]].dropna().shape[0])
            color = "white" if win_mat[i, j] < 15 or win_mat[i, j] > 35 else "black"
            ax.text(j, i, f"{win_mat[i, j]}/{n_total}", ha="center", va="center",
                    color=color, fontsize=10, weight='bold')
    ax.set_xticks(range(len(cols_base)))
    ax.set_xticklabels(cols_base, rotation=30, ha="right")
    ax.set_yticks(range(len(rows_ours)))
    ax.set_yticklabels(rows_ours)
    ax.set_title("Wins of our methods vs each TabPFN-paper baseline (count out of paired-dataset count)")
    cbar = plt.colorbar(im, ax=ax, label="datasets won (out of ~50)")
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig_overview_win_matrix.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    print(f"Saved {out}")


# ---- Figure 3: Cumulative distribution rel-PLS ----
def fig3_cumulative_relpls():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Plot ours first (thicker lines)
    ours_to_plot = list(OURS.keys()) + ["Oracle (best of ours)"]
    for v in ours_to_plot:
        sub = long_df[long_df.variant == v]
        if len(sub) == 0:
            continue
        sorted_rel = np.sort(sub.rel_pls.values)
        cum = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
        ax.plot(sorted_rel, cum, label=f"{v} (ours)" if v != "Ridge-raw" else "Ridge-raw (sklearn)",
                color=COLOR_OURS.get(v, "gray"), lw=2.2 if v == "Oracle (best of ours)" else 1.6, alpha=0.9)
    for v in BASELINES.keys():
        sub = long_df[long_df.variant == v]
        if len(sub) == 0:
            continue
        sorted_rel = np.sort(sub.rel_pls.values)
        cum = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
        ax.plot(sorted_rel, cum, label=v, color=COLOR_BASE[v], lw=1.4, ls="--", alpha=0.85)
    ax.axvline(1.0, color="black", ls=":", lw=1, alpha=0.6)
    ax.text(1.01, 0.05, "PLS parity", ha="left", va="bottom", fontsize=9, color="dimgray")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("RMSEP / PLS-RMSEP  (lower is better)", fontsize=11)
    ax.set_ylabel("Fraction of datasets ≤ x")
    ax.set_title("Cumulative distribution of rel-PLS — ours (solid) vs TabPFN-paper baselines (dashed)", fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_overview_cumulative_relpls.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    print(f"Saved {out}")


# ---- Figure 4: Per-dataset rel-PLS heatmap (top contenders only) ----
def fig4_per_dataset_heatmap():
    contenders = ["mkR-default-sparse3", "MKM-reml-asls", "mkR-asls-sparse2",
                  "mkR-msc-sparse3", "Oracle (best of ours)",
                  "PLS", "Ridge (paper)", "TabPFN-opt", "CNN-NICON", "CatBoost"]
    pivot = long_df.pivot(index="dataset", columns="variant", values="rel_pls")
    pivot = pivot[contenders]
    # Sort datasets by Oracle (best of ours)
    pivot = pivot.sort_values("Oracle (best of ours)")
    fig, ax = plt.subplots(figsize=(10, 12))
    data = pivot.values
    vmax = float(np.nanpercentile(data, 95))
    vmin = float(np.nanpercentile(data, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(contenders)))
    ax.set_xticklabels(contenders, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([d if len(d) <= 35 else d[:32]+'...' for d in pivot.index], fontsize=7)
    ax.set_title("Per-dataset rel-PLS  (greener = better)\nDatasets sorted by Oracle (best of ours)")
    plt.colorbar(im, ax=ax, label="rel-RMSEP vs PLS")
    fig.tight_layout()
    out = FIG_DIR / "fig_overview_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    print(f"Saved {out}")


# ---- Figure 5: Win-fraction summary (a single horizontal bar per variant: % datasets where it wins vs PLS / vs TabPFN-opt) ----
def fig5_win_fraction():
    pivot_pre = long_df.pivot(index="dataset", columns="variant", values="rmse")
    rows = []
    for v in list(OURS.keys()) + ["Oracle (best of ours)"] + list(BASELINES.keys()):
        sub = long_df[long_df.variant == v]
        if len(sub) == 0:
            continue
        n = int(len(sub))
        win_pls = int((sub.rel_pls < 1.0).sum())
        if v == "TabPFN-opt":
            n_pair = n; wt = 0  # self-comparison, by convention 0 wins
        elif "TabPFN-opt" in pivot_pre.columns and v in pivot_pre.columns:
            paired = pivot_pre[[v, "TabPFN-opt"]].dropna()
            n_pair = int(len(paired))
            wt = int((paired[v] < paired["TabPFN-opt"]).sum())
        else:
            n_pair = 0; wt = 0
        rows.append({"variant": v, "n_pls": n, "wins_pls": win_pls,
                     "n_tabpfn": n_pair, "wins_tabpfn": wt,
                     "rate_pls": float(100.0 * win_pls / n),
                     "rate_tabpfn": float(100.0 * wt / n_pair) if n_pair else 0.0,
                     "is_ours": bool(v in COLOR_OURS)})
    df = pd.DataFrame(rows)
    df_ours = df[df.is_ours].copy().sort_values("rate_tabpfn", ascending=False)
    df_base = df[~df.is_ours].copy().sort_values("rate_tabpfn", ascending=False)
    df = pd.concat([df_ours, df_base], ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.4
    colors_pls = [COLOR_OURS.get(v, COLOR_BASE.get(v, "gray")) for v in df.variant]
    ax.bar(x - w/2, df.rate_pls, w, label="% wins vs PLS", color=colors_pls, alpha=0.5, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, df.rate_tabpfn, w, label="% wins vs TabPFN-opt", color=colors_pls, alpha=0.95, edgecolor="black", linewidth=0.5)
    for i, (rp, rt) in enumerate(zip(df.rate_pls, df.rate_tabpfn)):
        ax.text(i - w/2, rp + 1, f'{rp:.0f}%', ha='center', fontsize=8)
        ax.text(i + w/2, rt + 1, f'{rt:.0f}%', ha='center', fontsize=8, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.variant, rotation=30, ha='right')
    ax.set_ylabel('Win fraction (%)')
    ax.set_title('Win rate vs PLS (light) and TabPFN-opt (dark) — across ~50 Stage A datasets')
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    fig.tight_layout()
    out = FIG_DIR / "fig_overview_winrate.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_median_relpls()
    fig2_win_matrix()
    fig3_cumulative_relpls()
    fig4_per_dataset_heatmap()
    fig5_win_fraction()
    print("\nAll overview figures generated.")
