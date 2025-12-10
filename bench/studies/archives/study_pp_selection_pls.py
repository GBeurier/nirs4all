import argparse
import sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.analysis import TransferPreprocessingSelector


# =============================================================================
# Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Study Proto 1 - PLS Preprocessing Selection")
parser.add_argument("--show", action="store_true", help="Show plots interactively")
parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0-2)")
args = parser.parse_args()

trials = 25
top_k = 20
preset = "balanced"

DATA_PATH = Path(__file__).parent / "_datasets" / "redox" / "1700_Brix_StratGroupedKfold"

PP_SPEC = {
    "_cartesian_": [
        {"_or_": [None, "msc", "snv", "emsc", "rsnv"]},  # Stage 1: Scatter correction
        {"_or_": [None, "savgol", "savgol_15", "gaussian", "gaussian2", "msc", "snv", "emsc", "rsnv"]},  # Stage 2: Smoothing
        {"_or_": [None, "d1", "d2", "savgol_d1", "savgol15_d1", "savgol_d2"]},  # Stage 3: Derivatives
        {"_or_": [None, "haar", "detrend", "area_norm", "wav_sym5", "wav_coif3", "msc", "snv", "emsc"]},  # Stage 4: Post-processing
    ],
}


def main():
    """Run the preprocessing selection study."""
    print("=" * 70)
    print("STUDY PROTO 1: Automated PLS Preprocessing Selection")
    print("=" * 70)
    print()

    # =========================================================================
    print("Loading dataset configuration...")
    dataset_config = DatasetConfigs(str(DATA_PATH))

    # Create selector with generator spec
    selector = TransferPreprocessingSelector(
        preset="balanced",
        preprocessing_spec=PP_SPEC,
        verbose=args.verbose,
    )

    t0 = time.time()
    results = selector.fit(dataset_config)
    selection_time = time.time() - t0

    print()
    print(f"Selection completed in {selection_time:.1f}s")
    print()

    filtered_pp_list = results.to_preprocessing_list(top_k=top_k)

    print(f"Selected {len(filtered_pp_list)} preprocessings:")
    for i, pp_transforms in enumerate(filtered_pp_list, 1):
        result = results.ranking[i - 1]
        print(f"  {i:2d}. {result.name:<55} (score={result.transfer_score:.4f}, improvement={result.improvement_pct:.1f}%)")
    print()

    # =========================================================================

    pipeline = [
        {"y_processing": MinMaxScaler(feature_range=(0.05, 0.9))},

        {"feature_augmentation": {"_or_": filtered_pp_list, "pick": (1, 2)}},

        MinMaxScaler,

        {
            "model": PLSRegression(),
            "name": "PLS-Finetuned",
            "finetune_params": {
                "n_trials": trials,
                "verbose": 0,
                "approach": "grouped",
                "eval_mode": "avg",
                "sample": "tpe",
                "model_params": {
                    "n_components": ("int", 1, 40),
                },
            },
        },
    ]

    print(f"Pipeline configured with {len(filtered_pp_list)} preprocessings")
    print()

    # =========================================================================

    pipeline_config = PipelineConfigs(pipeline, "study_proto_1")

    runner = PipelineRunner(
        save_files=True,
        verbose=0,
        plots_visible=False,
    )

    t0 = time.time()
    predictions, _ = runner.run(pipeline_config, dataset_config)
    training_time = time.time() - t0

    print()
    print(f"Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    print()

    # =========================================================================
    # Results Analysis
    # =========================================================================
    print("=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    print()

    # Build transfer rank lookup from selection results
    transfer_rank_lookup = {}
    for rank, res in enumerate(results.ranking, 1):
        transfer_rank_lookup[res.name] = rank

    import math

    ranking_metric = "rmse"
    n_top = 10

    print(f"Top {n_top} models by {ranking_metric.upper()} (with transfer rank):")
    print("-" * 70)
    top_models = predictions.top(n=n_top, rank_metric=ranking_metric)

    for idx, prediction in enumerate(top_models, 1):
        # Get MSE and compute RMSE
        mse = prediction.get("test_score", prediction.get("val_score", None))
        rmse = math.sqrt(mse) if mse is not None else None
        preprocessing = prediction.get("preprocessings", "N/A")

        # Get transfer rank for this preprocessing
        transfer_rank = transfer_rank_lookup.get(preprocessing, "?")

        rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
        transfer_rank_str = f"#{transfer_rank}" if isinstance(transfer_rank, int) else transfer_rank

        print(f"{idx:2d}. RMSE={rmse_str:>8} | transfer rank={transfer_rank_str:>4} | {preprocessing}")

    print()

    # Best preprocessing summary
    print("-" * 70)
    print("Best Result:")
    print("-" * 70)
    best = top_models[0]
    best_mse = best.get("test_score", None)
    best_rmse = math.sqrt(best_mse) if best_mse is not None else "N/A"
    best_pp = best.get("preprocessings", "N/A")
    best_transfer_rank = transfer_rank_lookup.get(best_pp, "?")
    print(f"  Preprocessing: {best_pp}")
    print(f"  Transfer rank: #{best_transfer_rank}")
    print(f"  Test RMSE: {best_rmse:.4f}" if isinstance(best_rmse, float) else f"  Test RMSE: {best_rmse}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_candidates = 5 * 5 * 6 * 6  # 900 (with None filtering: ~899 unique)
    print(f"  Total preprocessing candidates: {total_candidates} (5×5×6×6)")
    print(f"  Filtered to top: {top_k}")
    print(f"  Selection time: {selection_time:.1f}s")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Total time: {selection_time + training_time:.1f}s")
    print()

    # =========================================================================
    # Visualizations
    # =========================================================================
    if args.show or True:  # Always save plots
        print("Generating visualizations...")
        output_dir = DATA_PATH.parent

        analyzer = PredictionAnalyzer(predictions)

        # Top K plot
        try:
            fig1 = analyzer.plot_top_k(k=5, rank_metric="rmse", rank_partition="test")
            if isinstance(fig1, list):
                for i, f in enumerate(fig1):
                    f.savefig(output_dir / f"study_proto_1_top_k_{i}.png", dpi=150, bbox_inches="tight")
            else:
                fig1.savefig(output_dir / "study_proto_1_top_k.png", dpi=150, bbox_inches="tight")
            print("  Saved: study_proto_1_top_k.png")
        except Exception as e:
            print(f"  Warning: Could not create top-k plot: {e}")

        # Transfer selection ranking plot
        try:
            fig_transfer = results.plot_ranking(top_k=15)
            fig_transfer.savefig(output_dir / "study_proto_1_transfer_ranking.png", dpi=150, bbox_inches="tight")
            print("  Saved: study_proto_1_transfer_ranking.png")
        except Exception as e:
            print(f"  Warning: Could not create transfer ranking plot: {e}")

        print()

    print("=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
