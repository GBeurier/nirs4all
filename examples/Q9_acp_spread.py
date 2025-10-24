"""
Q9 Example - Preprocessing Quality Analysis with PCA Geometry
============================================================
Evaluates how well different preprocessing techniques preserve the geometric
structure of NIRS data using PCA-based metrics.
"""

# Standard library imports
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.utils.PCA_analyzer import PreprocPCAEvaluator

# Configuration variables
feature_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
]

# Build the pipeline
pipeline = [
    MinMaxScaler(),
    # {"feature_augmentation": {"_or_": preprocessing_options, "size": [3, (1, 3)], "count": 10}},  # no adapted to procrustes
    {"_or_": preprocessing_options, "size": (2, 4), "count": 10},
    # "2d_chart",
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "PCA_dist")
data_path = ['sample_data/regression', 'sample_data/classification', 'sample_data/binary']
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
print("🔄 Running preprocessing pipeline...")
runner = PipelineRunner(save_files=False, verbose=0, keep_datasets=True, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Get datasets (no manual pivot needed - evaluator handles it!)
datasets_raw = runner.raw_data  # {dataset_name: np_array(n_samples, n_features)}
datasets_pp = runner.pp_data  # {dataset_name: {preproc_name: np_array(n_samples, n_features)}}

# Create evaluator and fit
print("\n📊 Analyzing PCA geometry preservation...")


evaluator = PreprocPCAEvaluator(r_components=12, knn=10)
evaluator.fit(datasets_raw, datasets_pp)                    # Automatic structure detection!

# Display results table
print("\n" + "="*120)
print("PREPROCESSING QUALITY METRICS")
print("="*120)
print(evaluator.df_.sort_values(["dataset", "preproc"]).to_string(index=False))
print("\n" + "="*120)
print("INTERPRETATION GUIDE:")
print("  - evr_pre: Explained variance ratio (higher is better, >0.95 is good)")
print("  - cka, rv: Similarity to raw PCA structure (higher is better, >0.9 is excellent)")
print("  - procrustes: Distance from raw PCA (lower is better, <0.1 is excellent)")
print("  - trustworthiness: k-NN preservation (higher is better, >0.95 is excellent)")
print("  - grassmann: NaN when feature dimensions change (e.g., feature augmentation)")
print("="*120)


# ============================================================================
# NEW: Cross-Dataset Analysis for Multi-Machine Compatibility
# ============================================================================
print("\n" + "="*120)
print("CROSS-DATASET ANALYSIS - Evaluating Multi-Machine Compatibility")
print("="*120)
print("\n🔬 Analyzing how preprocessing affects inter-dataset distances...")
print("   (Goal: Find preprocessing that brings different datasets closer together)")

if not evaluator.cross_dataset_df_.empty:
    # Display cross-dataset summary
    summary = evaluator.get_cross_dataset_summary(metric='centroid_improvement')
    print("\n📊 Cross-Dataset Distance Summary:")
    print("   (Sorted by centroid improvement - higher is better)")
    print("-" * 120)
    print(summary.to_string(index=False))
    print("\n" + "="*120)
    print("CROSS-DATASET INTERPRETATION GUIDE:")
    print("  - centroid_improv_mean: How much preprocessing reduced dataset centroid distance")
    print("                          (positive = datasets closer, negative = datasets farther)")
    print("  - spread_improv_mean: How much preprocessing reduced distribution spread distance")
    print("  - centroid_dist_pp: Absolute distance between dataset centroids after preprocessing")
    print("  - spread_dist_pp: Absolute distribution distance after preprocessing")
    print("  → Look for HIGH improvement values and LOW absolute distances for best cross-prediction!")
    print("="*120)

    # Print top 3 preprocessing methods for cross-dataset compatibility
    print("\n🏆 TOP 3 PREPROCESSING FOR CROSS-DATASET COMPATIBILITY:")
    top3 = summary.head(3)
    for idx, row in top3.iterrows():
        pp_name = row['preproc'].split('|')[-1].replace('MinMax>', '').replace('>', ' → ')
        print(f"\n   {idx + 1}. {pp_name}")
        print(f"      Centroid Improvement: {row['centroid_improv_mean']:.4f} (±{row['centroid_improv_std']:.4f})")
        print(f"      Spread Improvement:   {row['spread_improv_mean']:.4f} (±{row['spread_improv_std']:.4f})")

    # Quality metric convergence analysis
    print("\n" + "="*120)
    print("QUALITY METRIC CONVERGENCE ANALYSIS")
    print("="*120)
    print("\n🔬 Analyzing how preprocessing affects quality metric homogeneity across datasets...")
    print("   (Goal: Find preprocessing that makes datasets behave similarly in quality metrics)")

    quality_convergence = evaluator.get_quality_metric_convergence()
    quality_metrics = ['evr_pre', 'cka', 'rv', 'procrustes', 'trustworthiness', 'grassmann']

    # Compute average convergence score
    avg_convergence = quality_convergence[[f'{m}_convergence' for m in quality_metrics]].mean(axis=1)
    quality_convergence['avg_convergence'] = avg_convergence
    quality_convergence_sorted = quality_convergence.sort_values('avg_convergence', ascending=False)

    print("\n📊 Quality Metric Convergence Summary:")
    print("   (Sorted by average convergence - higher is better)")
    print("-" * 120)
    display_cols = ['preproc'] + [f'{m}_convergence' for m in quality_metrics] + ['avg_convergence']
    print(quality_convergence_sorted[display_cols].to_string(index=False))

    print("\n" + "="*120)
    print("QUALITY CONVERGENCE INTERPRETATION GUIDE:")
    print("  - Positive convergence: Preprocessing reduces variance = datasets more similar")
    print("  - Negative convergence: Preprocessing increases variance = datasets more different")
    print("  - evr_pre_convergence: Explained variance becomes more consistent")
    print("  - cka/rv_convergence: PCA structure similarity becomes more consistent")
    print("  - procrustes/trustworthiness_convergence: Geometric preservation becomes more consistent")
    print("  → Look for HIGH convergence values for robust cross-dataset models!")
    print("="*120)

    print("\n🏆 TOP 3 PREPROCESSING FOR QUALITY HOMOGENEITY:")
    top3_quality = quality_convergence_sorted.head(3)
    for idx, row in top3_quality.iterrows():
        pp_name = row['preproc'].split('|')[-1].replace('MinMax>', '').replace('>', ' → ')
        print(f"\n   {idx + 1}. {pp_name}")
        print(f"      Average Convergence: {row['avg_convergence']:.4f}")
        print(f"      EVR Convergence:   {row['evr_pre_convergence']:.4f}")
        print(f"      CKA Convergence:   {row['cka_convergence']:.4f}")
else:
    print("\n⚠️  Not enough datasets for cross-dataset analysis (need at least 2)")

# Generate all visualizations
print("\n🎨 Generating visualizations...")

# NEW: Key visualizations for transfer learning
if not evaluator.cross_dataset_df_.empty:
    print("\n   🎯 KEY VISUALIZATIONS FOR TRANSFER LEARNING:")

    print("   📊 1. All datasets in same PCA space (raw + all preprocessings)...")
    evaluator.plot_all_datasets_pca()

    print("   📏 2. Centroid distance matrices (raw vs all preprocessings)...")
    evaluator.plot_distance_matrices(metric='centroid')

    print("   🏆 3. Preprocessing ranking for transfer learning (centroid distance)...")
    evaluator.plot_distance_reduction_ranking(metric='centroid', log_scale=True)

    print("   📐 4. Spread distance matrices (raw vs all preprocessings)...")
    evaluator.plot_distance_matrices(metric='spread')

    print("   🎖️  5. Preprocessing ranking for transfer learning (spread distance)...")
    evaluator.plot_distance_reduction_ranking(metric='spread', log_scale=True)

    print("   📊 6. Quality metric convergence across datasets...")
    evaluator.plot_quality_metric_convergence()
else:
    print("\n⚠️  Not enough datasets for cross-dataset analysis (need at least 2)")

# Optional: Original within-dataset analysis
print("\n   📈 7. Within-dataset structure preservation metrics...")
evaluator.plot_preservation_summary(by="preproc")

# plt.show(block=True)

print("\n✅ Analysis complete!")
print("="*120)
