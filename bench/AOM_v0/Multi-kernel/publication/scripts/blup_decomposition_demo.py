"""Generate the BLUP per-block decomposition demo figure.

Loads one of the smoke datasets (default ALPINE), fits BLUP-reml-asls,
and produces:

- ``fig_blup_decomposition.{pdf,png}`` — stacked bar chart of per-block
  contributions for the top-10 deviating test samples.
- ``fig_blup_variance_components.{pdf,png}`` — bar chart of relative
  variance contribution per AOM block.

Usage:

```bash
.venv/bin/python bench/AOM_v0/Multi-kernel/publication/scripts/blup_decomposition_demo.py \
  --dataset ALPINE/ALPINE_P_291_KS \
  --branch asls \
  --out-dir bench/AOM_v0/Multi-kernel/publication/figures
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup (same as the benchmark runner).
ROOT = Path(__file__).resolve()
PUBLICATION = ROOT.parent.parent
MULTI_KERNEL = PUBLICATION.parent
AOM_V0 = MULTI_KERNEL.parent
REPO_ROOT = AOM_V0.parent.parent
RIDGE_ROOT = AOM_V0 / "Ridge"
MKR_ROOT = MULTI_KERNEL / "MKR"
MKM_ROOT = MULTI_KERNEL / "MkM"
BLUP_ROOT = MULTI_KERNEL / "Blup"
for p in (RIDGE_ROOT, BLUP_ROOT, MKM_ROOT, MULTI_KERNEL, MKR_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_csv_array(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).any():
        col_mean = np.nanmean(arr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(col_mean, idx[1])
    return arr


def _load_csv_target(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep=";")
    return df.iloc[:, 0].astype(float).to_numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ALPINE/ALPINE_P_291_KS")
    parser.add_argument("--branch", default="asls",
                        choices=["none", "snv", "msc", "asls", "osc", "emsc1"])
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K deviating test samples to plot.")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("bench/AOM_v0/Multi-kernel/publication/figures"),
    )
    parser.add_argument("--all57-csv", type=Path,
                        default=Path("bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv"))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    db, ds = args.dataset.split("/", 1)
    cohort = pd.read_csv(args.all57_csv)
    sub = cohort[(cohort.database_name == db) & (cohort.dataset == ds)]
    if sub.empty:
        print(f"dataset {args.dataset} not found in cohort csv")
        return 1
    row = sub.iloc[0].to_dict()
    X_train = _load_csv_array(REPO_ROOT / row["train_path"])
    X_test = _load_csv_array(REPO_ROOT / row["test_path"])
    y_train = _load_csv_target(REPO_ROOT / row["ytrain_path"])
    y_test = _load_csv_target(REPO_ROOT / row["ytest_path"])

    from blup.estimator import AOMMultiKernelBLUP
    print(f"[demo] fitting BLUP-reml-{args.branch} on {args.dataset} "
          f"(n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, p={X_train.shape[1]})")
    model = AOMMultiKernelBLUP(
        operator_bank="compact",
        method="reml",
        n_random_restarts=3,
        max_iter=80,
        branch_preproc=args.branch,
        random_state=0,
    )
    model.fit(X_train, y_train)
    comps = model.predict_components(X_test)
    y_pred = comps["total"]
    print(f"[demo] RMSE_test = {float(np.sqrt(np.mean((y_test - y_pred) ** 2))):.4f}")
    print(f"[demo] sigma2_blocks: {dict(zip(model.block_names_, model.sigma2_blocks_))}")
    print(f"[demo] sigma2_residual: {model.sigma2_residual_:.4f}")

    import matplotlib.pyplot as plt

    # === Figure 1: per-block contributions for top-K deviating test samples
    deviation = np.abs(y_test - y_pred)
    top_idx = np.argsort(deviation)[::-1][: args.top_k]
    block_names = list(comps["random"].keys())
    contrib_matrix = np.array([comps["random"][b] for b in block_names])  # (B, n_test)
    fixed_part = comps["fixed"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(top_idx))
    cmap = plt.cm.tab10
    # Fixed-effect bar.
    ax.bar(
        range(len(top_idx)), fixed_part[top_idx], label="fixed (intercept)",
        color="lightgray", edgecolor="black", linewidth=0.5,
    )
    bottom += fixed_part[top_idx]
    for b, name in enumerate(block_names):
        c = contrib_matrix[b, top_idx]
        ax.bar(range(len(top_idx)), c, bottom=bottom, label=name,
               color=cmap(b % 10), edgecolor="black", linewidth=0.3)
        bottom += c
    # Overlay observed y as scatter.
    ax.scatter(
        range(len(top_idx)), y_test[top_idx], color="red",
        marker="o", zorder=10, label="observed y",
    )
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([f"#{i}" for i in top_idx], rotation=0)
    ax.set_xlabel("Test sample (top-{} deviating)".format(args.top_k))
    ax.set_ylabel("Contribution to predicted y")
    ax.set_title(
        f"BLUP per-block decomposition — {args.dataset} (branch={args.branch})\n"
        f"each bar = sum of fixed + random contributions = predict()"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.0), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"fig_blup_decomposition.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] wrote fig_blup_decomposition.{{pdf,png}}")

    # === Figure 2: relative variance components per block
    rel = model.relative_contributions_
    names = [k for k in rel if k != "_residual"]
    values = [rel[k] for k in names] + [rel["_residual"]]
    labels = names + ["_residual"]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(range(len(values)), values,
            color=["tab:blue"] * len(names) + ["lightgray"])
    ax2.set_xticks(range(len(values)))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("relative variance contribution h_b")
    ax2.set_title(
        f"MKM/BLUP variance decomposition — {args.dataset} (branch={args.branch})"
    )
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(args.out_dir / f"fig_blup_variance_components.{ext}", dpi=150)
    plt.close(fig2)
    print(f"[demo] wrote fig_blup_variance_components.{{pdf,png}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
