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
    # feature_scaler,
    # {"feature_augmentation": {"_or_": preprocessing_options, "size": [1, (1, 2)], "count": 15}}, # no adapted to procrustes
    {"_or_": preprocessing_options, "size": (1, 3), "count": 20}
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "PCA_dist")
data_path = ['sample_data/regression', 'sample_data/regression_2', 'sample_data/regression_3']
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
print("🔄 Running preprocessing pipeline...")
runner = PipelineRunner(save_files=False, verbose=0, keep_datasets=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Get datasets (no manual pivot needed - evaluator handles it!)
datasets_raw = runner.raw_data
datasets_pp = runner.pp_data

# Create evaluator and fit
print("\n📊 Analyzing PCA geometry preservation...")
evaluator = PreprocPCAEvaluator(r_components=12, knn=10)
evaluator.fit(datasets_raw, datasets_pp)  # Automatic structure detection!

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

# Generate all visualizations
print("\n🎨 Generating visualizations...")
evaluator.plot_pca_scatter()
evaluator.plot_summary(by="preproc")
evaluator.plot_distance_network(metric='cka')
plt.show()

print("\n✅ Analysis complete!")
