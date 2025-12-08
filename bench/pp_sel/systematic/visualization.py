"""Visualization functions for systematic selection results."""

import numpy as np
import pandas as pd


def plot_metrics_heatmap(
    df: pd.DataFrame,
    metrics: list = None,
    top_k: int = 20,
    title: str = "Metrics Heatmap",
    output_path: str = None,
    figsize: tuple = (12, 10),
    name_column: str = "name",
):
    """Create a heatmap of preprocessings vs metrics.

    Args:
        df: DataFrame with preprocessing results.
        metrics: List of metric columns to display. If None, uses defaults.
        top_k: Number of top preprocessings to display (by total_score or final_score).
        title: Title for the plot.
        output_path: Path to save the figure. If None, doesn't save.
        figsize: Figure size tuple.
        name_column: Column name containing preprocessing names.

    Returns:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    # Default metrics based on available columns
    if metrics is None:
        if "final_score" in df.columns:
            # Stage 3 or 4 data
            metrics = ["unsupervised_score", "ridge_r2", "knn_score", "xgb_score", "proxy_score", "final_score"]
        else:
            # Stage 1 data
            metrics = ["variance_ratio", "effective_dim", "snr", "roughness", "separation", "total_score"]

    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("Warning: No valid metrics found in DataFrame")
        return None

    # Sort by score and take top_k
    sort_col = "final_score" if "final_score" in df.columns else "total_score"
    if sort_col in df.columns:
        df_sorted = df.sort_values(sort_col, ascending=False).head(top_k)
    else:
        df_sorted = df.head(top_k)

    # Extract data for heatmap
    names = df_sorted[name_column].tolist()
    data = df_sorted[metrics].values

    # Normalize each column to [0, 1] for better visualization
    data_normalized = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        col = data[:, j]
        min_val, max_val = col.min(), col.max()
        if max_val - min_val > 1e-10:
            data_normalized[:, j] = (col - min_val) / (max_val - min_val)
        else:
            data_normalized[:, j] = 0.5

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(data_normalized, cmap="viridis", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Score", rotation=270, labelpad=15)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=9)

    # Add text annotations with original values
    for i in range(len(names)):
        for j in range(len(metrics)):
            value = data[i, j]
            # Format value based on magnitude
            if abs(value) < 0.01:
                text = f"{value:.1e}"
            elif abs(value) < 1:
                text = f"{value:.3f}"
            elif abs(value) < 100:
                text = f"{value:.2f}"
            else:
                text = f"{value:.1f}"

            # Choose text color based on background
            text_color = "white" if data_normalized[i, j] < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Preprocessing", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Heatmap saved to: {output_path}")

    return fig


def plot_distance_heatmap(
    distance_matrix: pd.DataFrame,
    title: str = "Distance Heatmap",
    output_path: str = None,
    figsize: tuple = (12, 10),
    annotate: bool = True,
):
    """Create a heatmap of pairwise distances between preprocessings.

    Args:
        distance_matrix: Square DataFrame with pairwise distances.
        title: Title for the plot.
        output_path: Path to save the figure. If None, doesn't save.
        figsize: Figure size tuple.
        annotate: Whether to add text annotations.

    Returns:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    names = distance_matrix.index.tolist()
    data = distance_matrix.values

    # Create heatmap
    im = ax.imshow(data, cmap="viridis", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Distance", rotation=270, labelpad=15)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # Add text annotations
    if annotate:
        for i in range(len(names)):
            for j in range(len(names)):
                value = data[i, j]
                text_color = "white" if value < (data.max() + data.min()) / 2 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=7)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Distance heatmap saved to: {output_path}")

    return fig


def plot_dual_distance_heatmap(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    names: list,
    label1: str = "Subspace",
    label2: str = "Geometry",
    title: str = "Dual Distance Heatmap",
    output_path: str = None,
    figsize: tuple = (14, 12),
):
    """Create a heatmap with two distance metrics shown in upper/lower triangles.

    Each cell is split diagonally with matrix1 in upper triangle and matrix2 in lower.

    Args:
        matrix1: First distance matrix (shown in upper triangle).
        matrix2: Second distance matrix (shown in lower triangle).
        names: List of preprocessing names.
        label1: Label for matrix1.
        label2: Label for matrix2.
        title: Title for the plot.
        output_path: Path to save the figure. If None, doesn't save.
        figsize: Figure size tuple.

    Returns:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.colors import Normalize

    n = len(names)
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize both matrices to same scale for consistent coloring
    vmin = min(matrix1.min(), matrix2.min())
    vmax = max(matrix1.max(), matrix2.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Get viridis colormap
    cmap = plt.get_cmap("viridis")

    # Create triangular patches for each cell
    upper_patches = []
    lower_patches = []
    upper_colors = []
    lower_colors = []

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal - full cell with one color (or skip)
                continue

            # Cell corners
            x0, y0 = j - 0.5, i - 0.5
            x1, y1 = j + 0.5, i + 0.5

            # Upper triangle (top-left to bottom-right diagonal)
            upper_tri = Polygon([(x0, y0), (x1, y0), (x1, y1)], closed=True)
            upper_patches.append(upper_tri)
            upper_colors.append(cmap(norm(matrix1[i, j])))

            # Lower triangle
            lower_tri = Polygon([(x0, y0), (x0, y1), (x1, y1)], closed=True)
            lower_patches.append(lower_tri)
            lower_colors.append(cmap(norm(matrix2[i, j])))

    # Add patches to axes
    for patch, color in zip(upper_patches, upper_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('white')
        patch.set_linewidth(0.3)
        ax.add_patch(patch)

    for patch, color in zip(lower_patches, lower_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('white')
        patch.set_linewidth(0.3)
        ax.add_patch(patch)

    # Add diagonal cells (self-distance = 0)
    for i in range(n):
        from matplotlib.patches import Rectangle
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor='lightgray',
                         edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

    # Set limits and aspect
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # Invert y-axis
    ax.set_aspect('equal')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
                continue

            val1 = matrix1[i, j]
            val2 = matrix2[i, j]

            # Upper triangle value (slightly up-right from center)
            color1 = "white" if norm(val1) < 0.5 else "black"
            ax.text(j + 0.15, i - 0.15, f"{val1:.2f}", ha="center", va="center",
                    fontsize=6, color=color1, fontweight="bold")

            # Lower triangle value (slightly down-left from center)
            color2 = "white" if norm(val2) < 0.5 else "black"
            ax.text(j - 0.15, i + 0.15, f"{val2:.2f}", ha="center", va="center",
                    fontsize=6, color=color2, fontweight="bold")

    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Distance", rotation=270, labelpad=15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0.7), edgecolor='black', label=f'↗ {label1}'),
        Patch(facecolor=cmap(0.3), edgecolor='black', label=f'↙ {label2}'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Dual distance heatmap saved to: {output_path}")

    return fig


def plot_results(
    stage1_df: pd.DataFrame,
    final_df: pd.DataFrame,
    distance_matrix: pd.DataFrame = None,
    output_path: str = "systematic_results.png",
):
    """Create comprehensive visualization.

    Args:
        stage1_df: DataFrame with Stage 1 unsupervised results.
        final_df: DataFrame with final ranking results.
        distance_matrix: Optional distance matrix from Stage 2.
        output_path: Path to save the figure.

    Returns:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Stage 1 - Score distribution by depth
    ax1 = fig.add_subplot(2, 3, 1)
    for depth in sorted(stage1_df["depth"].unique()):
        subset = stage1_df[stage1_df["depth"] == depth]
        ax1.hist(subset["total_score"], bins=20, alpha=0.5, label=f"Depth {depth}")
    ax1.set_xlabel("Unsupervised Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Stage 1: Score Distribution by Depth")
    ax1.legend()

    # Plot 2: Stage 1 - Top 15 pipelines
    ax2 = fig.add_subplot(2, 3, 2)
    top15 = stage1_df.head(15)
    viridis = plt.get_cmap("viridis")
    colors = viridis(np.linspace(0.8, 0.2, 15))
    ax2.barh(range(15), top15["total_score"], color=colors)
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(top15["name"], fontsize=8)
    ax2.set_xlabel("Unsupervised Score")
    ax2.set_title("Stage 1: Top 15 Pipelines")
    ax2.invert_yaxis()

    # Plot 3: Distance heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    if distance_matrix is not None:
        im = ax3.imshow(distance_matrix.values, cmap="RdYlBu", aspect="auto")
        ax3.set_xticks(range(len(distance_matrix.columns)))
        ax3.set_yticks(range(len(distance_matrix.index)))
        ax3.set_xticklabels(
            distance_matrix.columns, rotation=45, ha="right", fontsize=6
        )
        ax3.set_yticklabels(distance_matrix.index, fontsize=6)
        plt.colorbar(im, ax=ax3, label="Distance")
        ax3.set_title("Stage 2: Preprocessing Distances")
    else:
        ax3.text(0.5, 0.5, "No distance data", ha="center", va="center")

    # Plot 4: Metrics comparison (radar-like bar chart)
    ax4 = fig.add_subplot(2, 3, 4)
    top5 = stage1_df.head(5)
    metrics = ["variance_ratio", "effective_dim", "snr", "separation"]
    x = np.arange(len(metrics))
    width = 0.15

    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[m] if m != "effective_dim" else row[m] / 10 for m in metrics]
        values = [min(v, 1.0) for v in values]  # Normalize
        ax4.bar(x + i * width, values, width, label=row["name"][:15])

    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(["Variance", "Eff. Dim (÷10)", "SNR", "Separation"])
    ax4.set_ylabel("Score")
    ax4.set_title("Stage 1: Metrics Comparison (Top 5)")
    ax4.legend(fontsize=7, loc="upper right")

    # Plot 5: Final ranking by type
    ax5 = fig.add_subplot(2, 3, 5)
    type_colors = {
        "single": "steelblue",
        "stacked": "coral",
        "augmented_2": "green",
        "augmented_3": "purple",
    }

    top20 = final_df.head(20)
    colors = [type_colors.get(t, "gray") for t in top20["type"]]
    ax5.barh(range(len(top20)), top20["final_score"], color=colors)
    ax5.set_yticks(range(len(top20)))
    ax5.set_yticklabels(top20["name"], fontsize=7)
    ax5.set_xlabel("Final Score")
    ax5.set_title("Final Ranking (Top 20)")
    ax5.invert_yaxis()

    # Add legend for types
    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax5.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Plot 6: Proxy vs Unsupervised scores
    ax6 = fig.add_subplot(2, 3, 6)
    scatter_colors = [type_colors.get(t, "gray") for t in final_df["type"]]
    ax6.scatter(
        final_df["unsupervised_score"],
        final_df["proxy_score"],
        c=scatter_colors,
        alpha=0.6,
        s=50,
    )
    ax6.set_xlabel("Unsupervised Score")
    ax6.set_ylabel("Proxy Score")
    ax6.set_title("Unsupervised vs Proxy Performance")

    # Add labels for top 5
    for i in range(min(5, len(final_df))):
        row = final_df.iloc[i]
        ax6.annotate(
            row["name"][:12],
            (row["unsupervised_score"], row["proxy_score"]),
            fontsize=7,
            alpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Plot saved to: {output_path}")

    return fig
