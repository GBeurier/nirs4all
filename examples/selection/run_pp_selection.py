#!/usr/bin/env python
"""
Systematic Preprocessing Selection - Main Script
=================================================

A comprehensive, systematic evaluation of preprocessing pipelines:

1. Stage 1 - Exhaustive Unsupervised Evaluation:
   - All single preprocessings (depth 1)
   - All stacked pipelines depth 2 (A → B)
   - All stacked pipelines depth 3+ (A → B → C → ...)
   - Compute unsupervised metrics for each
   - Output: CSV + visualization
   - Take top_stage1 best preprocessings

2. Stage 2 - Diversity Analysis:
   - Take top_stage1 candidates from Stage 1
   - Compute pairwise distances (Grassmann, CKA)
   - Remove pipelines too similar to better-scored ones (similarity_ratio)
   - Keep top_stage2 diverse candidates

3. Stage 3 - Proxy Model Evaluation:
   - Evaluate all Stage 2 candidates with Ridge/KNN proxy models
   - Take top_stage3 best performers

4. Stage 4 - Augmentation Evaluation:
   - Generate concatenations from Stage 3 results (up to augmentation_order)
   - Evaluate with proxy models
   - Take top_stage4 augmentations

5. Final Ranking:
   - Combine Stage 3 and Stage 4 results
   - Take top_final best configurations

Usage:
    python run_selection.py [options]

    # Minimal test run (fast)
    python run_selection.py --depth 2 --top-stage1 3 --top-stage2 3 --top-stage3 3 --top-stage4 3 --top-final 6 --aug-order 2

    # Full run with defaults
    python run_selection.py --depth 3 --plots
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure parent directories are in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systematic import (
    SystematicSelector,
    ResultCache,
    load_data,
    plot_results,
    plot_metrics_heatmap,
    plot_distance_heatmap,
    plot_dual_distance_heatmap,
)


def main():
    """Main entry point for systematic preprocessing selection."""
    parser = argparse.ArgumentParser(
        description="Systematic Preprocessing Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (all tops at 3, order 2)
  python run_selection.py --depth 2 --top-stage1 3 --top-stage2 3 --top-stage3 3 --top-stage4 3 --top-final 6 --aug-order 2

  # Full run with default parameters
  python run_selection.py --depth 3 --plots
        """,
    )
    # Pipeline depth
    parser.add_argument(
        "--depth", type=int, default=3, help="Maximum pipeline depth (1-4)"
    )

    # Top-k parameters for each stage
    parser.add_argument(
        "--top-stage1",
        type=int,
        default=150,
        help="Number of top candidates from Stage 1 (unsupervised) to pass to Stage 2",
    )
    parser.add_argument(
        "--top-stage2",
        type=int,
        default=75,
        help="Number of diverse candidates from Stage 2 to pass to Stage 3",
    )
    parser.add_argument(
        "--top-stage3",
        type=int,
        default=20,
        help="Number of top candidates from Stage 3 (proxy evaluation)",
    )
    parser.add_argument(
        "--top-stage4",
        type=int,
        default=10,
        help="Number of top augmentations from Stage 4",
    )
    parser.add_argument(
        "--top-final",
        type=int,
        default=15,
        help="Number of final configurations to return",
    )

    # Similarity and augmentation parameters
    parser.add_argument(
        "--similarity-ratio",
        type=float,
        default=0.95,
        help="Similarity threshold for diversity filtering (0-1). Higher = more strict filtering",
    )
    parser.add_argument(
        "--aug-order",
        type=int,
        default=3,
        choices=[2, 3],
        help="Maximum augmentation order (2 = 2-way only, 3 = 2-way and 3-way)",
    )

    # Caching options
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching (reuse previous results)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: output_dir/.cache)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running",
    )

    # Data and output options
    parser.add_argument("--plots", action="store_true", help="Show plots")
    parser.add_argument("--full", action="store_true", help="Use full nitro dataset")
    parser.add_argument("--data", type=str, default=None, help="Custom data path")
    parser.add_argument(
        "--output", type=str, default="selection", help="Output directory"
    )
    args = parser.parse_args()

    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Determine data path
    if args.data:
        data_path = args.data
    elif args.full:
        data_path = str(script_dir / "nitro_regression_merged" / "Digestibility_0.8")
    else:
        data_path = str(script_dir / "nitro_regression_merged" / "Digestibility_0.8")

    # Ensure output directory exists
    output_dir = script_dir / args.output
    os.makedirs(output_dir, exist_ok=True)

    # Set up cache directory
    cache_dir = args.cache_dir if args.cache_dir else str(output_dir / ".cache")

    # Clear cache if requested
    if args.clear_cache:
        cache = ResultCache(cache_dir)
        cleared = cache.clear_cache()
        print(f"🗑️ Cleared {cleared} cache entries from {cache_dir}")

    # Load data
    print(f"\n📂 Data path: {data_path}")
    X, y = load_data(data_path)

    # Run selection
    selector = SystematicSelector(
        verbose=1,
        cache_dir=cache_dir,
        use_cache=args.cache,
    )
    results = selector.run_full_selection(
        X=X,
        y=y,
        max_depth=args.depth,
        top_stage1=args.top_stage1,
        top_stage2=args.top_stage2,
        top_stage3=args.top_stage3,
        top_stage4=args.top_stage4,
        top_final=args.top_final,
        similarity_ratio=args.similarity_ratio,
        augmentation_order=args.aug_order,
        output_dir=str(output_dir),
    )

    # Create plots
    fig = plot_results(
        results["stage1_df"],
        results["final_df"],
        results["distance_matrix"],
        output_path=os.path.join(str(output_dir), "systematic_results.png"),
    )

    # Create metrics heatmaps for Stage 1, 3, and 4
    print("\n📊 Generating heatmaps...")

    # Stage 1 heatmap: unsupervised metrics
    plot_metrics_heatmap(
        results["stage1_df"],
        metrics=["variance_ratio", "effective_dim", "snr", "roughness", "separation", "total_score"],
        top_k=20,
        title="Stage 1: Unsupervised Metrics Heatmap (Top 20)",
        output_path=os.path.join(str(output_dir), "heatmap_stage1_metrics.png"),
    )

    # Stage 3 heatmap: proxy model scores
    plot_metrics_heatmap(
        results["stage3_df"],
        metrics=["unsupervised_score", "ridge_r2", "knn_score", "xgb_score", "proxy_score", "final_score"],
        top_k=min(20, len(results["stage3_df"])),
        title="Stage 3: Proxy Model Scores Heatmap",
        output_path=os.path.join(str(output_dir), "heatmap_stage3_metrics.png"),
    )

    # Stage 4 heatmap: augmentation scores
    if len(results["stage4_df"]) > 0:
        plot_metrics_heatmap(
            results["stage4_df"],
            metrics=["unsupervised_score", "ridge_r2", "knn_score", "xgb_score", "proxy_score", "final_score"],
            top_k=min(20, len(results["stage4_df"])),
            title="Stage 4: Augmentation Scores Heatmap",
            output_path=os.path.join(str(output_dir), "heatmap_stage4_metrics.png"),
        )

    # Distance heatmaps from Stage 2
    if results["distance_matrix"] is not None:
        plot_distance_heatmap(
            results["distance_matrix"],
            title="Stage 2: Combined Distance Matrix",
            output_path=os.path.join(str(output_dir), "heatmap_distance_combined.png"),
        )

    # Dual distance heatmap (subspace vs geometry)
    if results["diversity_analysis"] is not None:
        diversity = results["diversity_analysis"]
        if diversity.subspace_matrix is not None and diversity.geometry_matrix is not None:
            plot_dual_distance_heatmap(
                diversity.subspace_matrix,
                diversity.geometry_matrix,
                diversity.pipeline_names,
                label1="Subspace",
                label2="Geometry",
                title="Stage 2: Subspace vs Geometry Distances",
                output_path=os.path.join(str(output_dir), "heatmap_distance_dual.png"),
            )

    if args.plots:
        import matplotlib.pyplot as plt

        plt.show()

    print("\n✅ Complete!")
    print(f"   - Stage 3: {len(results['top_stage3'])} top single/stacked pipelines")
    print(f"   - Stage 4: {len(results['top_stage4'])} top augmentations")
    print(f"   - Final: {len(results['final_results'])} total configurations")


if __name__ == "__main__":
    main()
