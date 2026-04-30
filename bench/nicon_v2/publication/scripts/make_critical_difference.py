"""Critical-difference diagram (Demsar 2006).

Friedman test across (variant × dataset) ranks, Nemenyi pairwise post-hoc.
For a benchmark with k variants on N datasets, the critical difference is

    CD = q_α · sqrt(k(k+1) / (6N))

where q_α is the studentized-range critical value at α=0.05.

Usage::

    python publication/scripts/make_critical_difference.py \
      --csv bench/nicon_v2/benchmark_runs/stack_curated/results.csv \
      --out bench/nicon_v2/publication/figures/cd_curated.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


# Studentized range distribution critical values q_α at α=0.05 (Demsar 2006 Table 5).
Q_ALPHA_05 = {
    2: 1.96, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
}


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    if alpha != 0.05:
        raise ValueError("only α=0.05 supported (q values from Demsar 2006)")
    q = Q_ALPHA_05.get(k)
    if q is None:
        raise ValueError(f"k={k} not supported (need 2 ≤ k ≤ 12)")
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def average_ranks(df: pd.DataFrame, metric: str = "rmsep") -> tuple[pd.Series, int, int]:
    pivot = df.pivot_table(index="dataset", columns="variant", values=metric, aggfunc="median").dropna()
    ranks = pivot.rank(axis=1, method="average")
    avg = ranks.mean(axis=0).sort_values()
    return avg, pivot.shape[1], pivot.shape[0]


def plot_cd_diagram(avg_ranks: pd.Series, cd: float, k: int, n: int, out: Path, title: str = "") -> None:
    variants = avg_ranks.index.tolist()
    ranks = avg_ranks.values
    n_var = len(variants)
    fig_h = max(3.0, 0.4 * n_var + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.set_xlim(1, k)
    ax.set_ylim(0, n_var + 1)
    ax.invert_yaxis()
    for i, (variant, r) in enumerate(zip(variants, ranks)):
        y = i + 0.5
        ax.plot([1, r], [y, y], color="gray", linewidth=0.5)
        ax.scatter([r], [y], color="black", s=20, zorder=3)
        ax.annotate(f"{variant}  ({r:.2f})", xy=(r + 0.05, y), va="center", fontsize=8)
    ax.set_xlabel("average rank (lower = better)")
    ax.set_yticks([])
    ax.set_title(title or f"Friedman ranking on {n} datasets (k={k}, CD={cd:.2f}, α=0.05)")
    ax.plot([1, 1 + cd], [0.2, 0.2], color="red", linewidth=2)
    ax.annotate(f"CD = {cd:.2f}", xy=(1 + cd / 2, 0.05), ha="center", color="red", fontsize=9)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="rmsep")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["status"].astype(str) == "OK"]
    df = df[df["variant"].apply(lambda v: not str(v).startswith("DECON")
                                  and not str(v).startswith("NICON-baseline"))]

    avg_ranks, k, n = average_ranks(df, metric=args.metric)
    if k > 12 or k < 2:
        raise ValueError(f"unsupported k={k} for CD diagram")
    if n < 5:
        print(f"Only n={n} datasets — Friedman is not informative; writing diagram anyway")

    pivot = df.pivot_table(index="dataset", columns="variant", values=args.metric, aggfunc="median").dropna()
    samples = [pivot[c].values for c in avg_ranks.index]
    try:
        stat, p = friedmanchisquare(*samples)
    except ValueError as exc:
        print(f"friedmanchisquare failed: {exc}")
        stat, p = float("nan"), float("nan")
    print(f"Friedman chi-sq = {stat:.3f}, p = {p:.3g}, k = {k}, n = {n}")

    cd = critical_difference(k, n, alpha=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_cd_diagram(avg_ranks, cd, k, n, args.out,
                    title=f"Friedman/Nemenyi CD (k={k}, n={n}, alpha=0.05)")
    print(f"wrote {args.out}")
    print("average ranks:")
    for v, r in avg_ranks.items():
        print(f"  {v:<32} {r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
