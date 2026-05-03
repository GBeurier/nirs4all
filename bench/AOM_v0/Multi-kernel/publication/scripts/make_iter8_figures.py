"""Iter8 publication figures: cumulative dist + variant ranking + sparse ablation."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("bench/AOM_v0/Multi-kernel")
ITER8 = ROOT / "benchmark_runs/iter8_full54_champions/results.csv"
ITER7 = ROOT / "benchmark_runs/iter7_noretune/results.csv"
ITER5 = ROOT / "benchmark_runs/iter5_sparse/results.csv"
FIG_DIR = ROOT / "publication/figures"

VARIANT_LABELS = {
    "mkR-softmax_cv-asls-default-active15-sparse1": "mkR-asls-sparse1 (champion)",
    "mkR-softmax_cv-default-active15-sparse3": "mkR-default-sparse3",
    "mkR-softmax_cv-msc-default-active15-sparse3": "mkR-msc-sparse3",
    "MKM-reml-asls-default-active15": "MKM-reml-asls",
    "Ridge-raw": "Ridge (sklearn baseline)",
}
COLORS = {
    "mkR-softmax_cv-asls-default-active15-sparse1": "#d62728",
    "mkR-softmax_cv-default-active15-sparse3": "#1f77b4",
    "mkR-softmax_cv-msc-default-active15-sparse3": "#2ca02c",
    "MKM-reml-asls-default-active15": "#9467bd",
    "Ridge-raw": "#7f7f7f",
}

df = pd.read_csv(ITER8)
ok = df[df.status == "ok"]
ok_clean = ok[ok.dataset != "Quartz_spxy70"].copy()


# Figure 1: Cumulative distribution of rel-RMSEP vs TabPFN-opt
def fig1_cumulative_tabpfn():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v, label in VARIANT_LABELS.items():
        sub = ok_clean[ok_clean.variant == v]
        rel = sub.rel_rmsep_vs_tabpfn_opt.values
        sorted_rel = np.sort(rel)
        cum = np.arange(1, len(sorted_rel) + 1) / len(sorted_rel)
        ax.plot(sorted_rel, cum, label=label, color=COLORS[v],
                lw=2, alpha=0.85)
    ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax.text(1.0, 0.05, "TabPFN-opt parity", rotation=90, ha="right", va="bottom",
            fontsize=9, color="dimgray")
    ax.set_xlabel("RMSEP / TabPFN-opt RMSEP (lower is better)")
    ax.set_ylabel("Cumulative fraction of datasets")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Iter 8: cumulative distribution of relative RMSEP vs TabPFN-opt\n(Stage A, 50 datasets, n_train ≤ 1500)")
    ax.grid(alpha=0.3)
    out = FIG_DIR / "fig_iter8_cumulative_tabpfn.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out}")


# Figure 2: Box plot of rel-PLS across variants
def fig2_relrmsep_box():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    variants = list(VARIANT_LABELS.keys())
    data_pls = [ok_clean[ok_clean.variant == v].rel_rmsep_vs_pls.values for v in variants]
    data_tabpfn = [ok_clean[ok_clean.variant == v].rel_rmsep_vs_tabpfn_opt.values for v in variants]
    labels = [VARIANT_LABELS[v].split(" ")[0] for v in variants]

    for ax, data, title, parity in [
        (axes[0], data_pls, "vs PLS", 1.0),
        (axes[1], data_tabpfn, "vs TabPFN-opt", 1.0),
    ]:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                         showmeans=False, showfliers=False, whis=(5, 95))
        for patch, v in zip(bp["boxes"], variants):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.6)
        ax.axhline(parity, color="black", ls="--", lw=1, alpha=0.4)
        ax.set_ylabel(f"RMSEP / RMSEP({title.split()[1]})")
        ax.set_title(f"Iter 8: rel-RMSEP {title}")
        ax.set_ylim(0.5, 2.2)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "fig_iter8_boxplots.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out}")


# Figure 3: Sparse softmax ablation (iter5 vs iter7)
def fig3_sparse_mechanism():
    df5 = pd.read_csv(ITER5)
    df7 = pd.read_csv(ITER7)
    # Build dense reference using iter5 with -active15- (no sparse) — actually the dense
    # baseline is mkR-softmax_cv-asls-default-active15 from iter4
    df4 = pd.read_csv(ROOT / "benchmark_runs/iter4_score_methods/results.csv")
    target_ds = ["Beer_OriginalExtract_60_YbaseSplit", "TIC_spxy70",
                 "ALPINE_P_291_KS", "All_manure_MgO_SPXY_strat_Manure_type"]

    rows = []
    for ds in target_ds:
        # Dense: ASLS no sparse from iter4
        d_dense = df4[(df4.dataset == ds) & (df4.variant == "mkR-softmax_cv-asls-default-active15")]
        if len(d_dense):
            rows.append({"dataset": ds, "config": "dense", "rel_pls": d_dense.iloc[0].rel_rmsep_vs_pls})
        d_no = df7[(df7.dataset == ds) & (df7.variant == "mkR-softmax_cv-asls-default-active15-sparse2-noretune")]
        if len(d_no):
            rows.append({"dataset": ds, "config": "sparse2 (no retune)", "rel_pls": d_no.iloc[0].rel_rmsep_vs_pls})
        d_re = df5[(df5.dataset == ds) & (df5.variant == "mkR-softmax_cv-asls-default-active15-sparse2")]
        if len(d_re):
            rows.append({"dataset": ds, "config": "sparse2 (retune)", "rel_pls": d_re.iloc[0].rel_rmsep_vs_pls})

    abldf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = abldf.pivot(index="dataset", columns="config", values="rel_pls")
    pivot = pivot[["dense", "sparse2 (no retune)", "sparse2 (retune)"]]
    pivot.plot(kind="bar", ax=ax, color=["#bdbdbd", "#fdc086", "#d62728"], width=0.7)
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("rel-RMSEP vs PLS")
    ax.set_xlabel("")
    ax.set_title("Sparse softmax mechanism: pruning vs alpha re-tune (iter5/iter7 ablation)")
    ax.legend(loc="upper left", fontsize=9)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "fig_iter8_sparse_ablation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out}")


# Figure 4: Variant frequency in oracle (best-per-dataset)
def fig4_oracle_frequency():
    counts = []
    for ds in sorted(ok_clean.dataset.unique()):
        sub = ok_clean[ok_clean.dataset == ds]
        if len(sub) == 0:
            continue
        best = sub.sort_values("rel_rmsep_vs_pls").iloc[0]
        counts.append(best.variant)
    freq = pd.Series(counts).value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS.get(v, "gray") for v in freq.index]
    ax.barh([VARIANT_LABELS.get(v, v) for v in freq.index], freq.values, color=colors, alpha=0.8)
    for i, v in enumerate(freq.values):
        ax.text(v + 0.2, i, f"{v}/{len(counts)}", va="center", fontsize=10)
    ax.set_xlabel("Datasets won (out of 50)")
    ax.set_title("Iter 8: oracle variant frequency (best variant per dataset)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / "fig_iter8_oracle_frequency.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_cumulative_tabpfn()
    fig2_relrmsep_box()
    fig3_sparse_mechanism()
    fig4_oracle_frequency()
    print("\nAll iter8 figures generated.")
