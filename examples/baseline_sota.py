from dotenv import load_dotenv
from pathlib import Path
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import SavitzkyGolay, ASLSBaseline
from nirs4all.operators.filters import YOutlierFilter, HighLeverageFilter, SpectralQualityFilter, XOutlierFilter
from nirs4all.operators.filters.base import CompositeFilter

from huggingface_hub import login
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Hugging Face login for TabPFN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env or environment. TabPFN may not work properly.")

# Clear GPU cache if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##========================== Main Pipeline Configuration =========================##
# Configuration variables
DATA_PATH = [
    'sample_data/Hiba/LDMC_vera',
    # 'sample_data/Hiba/LDMC_hiba',
    # 'sample_data/Hiba/SLA_vera',
    # 'sample_data/Hiba/SLA_hiba'
]

AGGREGATION_KEY = "ID"  # None
TASK_TYPE = "regression"  # "classification" or "regression" or "auto" or "binary"

filter_quality = SpectralQualityFilter(
    max_nan_ratio=0.15,
    max_zero_ratio=0.4,
    min_variance=1e-6,
    max_value=4.0,
    min_value=-0.5,
)

filter_iqr = YOutlierFilter(method="iqr", threshold=1.5, reason="iqr")
filter_zscore = YOutlierFilter(method="zscore", threshold=3.0, reason="zscore")


TabPFNModel = TabPFNRegressor if TASK_TYPE == "regression" else TabPFNClassifier
tabpfn_real_path = 'tabpfn-v2.5-regressor-v2.5_real.ckpt' if TASK_TYPE == "regression" else 'tabpfn-v2.5-classifier-v2.5_real.ckpt'
# Define the pipeline
pipeline = [
    {
        "sample_filter": {
            "filters": [HighLeverageFilter, XOutlierFilter(method="pca_residual", n_components=30)],
            "mode": "any",
            "report": True,  # Print filtering report
        }
    },
    ASLSBaseline(),
    {"split": SPXYGFold(n_splits=1, random_state=42), "group": AGGREGATION_KEY},  # COMMENT IF TRAIN AND TEST ARE PROVIDED
    {"split": SPXYGFold(n_splits=3, random_state=42), "group": AGGREGATION_KEY},
    # {"chart_y": {"include_excluded": True, "highlight_excluded": True}},
    # {"chart_2d": {"include_excluded": True, "highlight_excluded": True}},
    # {"y_processing": [QuantileTransformer(n_quantiles=150, output_distribution='normal', random_state=42), StandardScaler()]},
    {"y_processing": StandardScaler()},
    # StandardScaler(),
    # SavitzkyGolay(),

    PCA(n_components=0.99, random_state=42, whiten=True),
    # PCA(50, random_state=42, whiten=True),
    StandardScaler(),
    PowerTransformer(),
    # {
    #     'model': {
    #         'framework': 'autogluon',
    #         'params': {
    #             'presets': 'extreme_quality',
    #             'time_limit': 3600,
    #             'num_bag_folds': 5,
    #             'random_state': 42,
    #         }
    #     },
    #     "name": "AutoGluon",
    # },
    # {
    #     "model": TabPFNModel(n_estimators=16, device='cuda', random_state=42),
    #     "name": "TabPFN",
    # },
    {
        "model": TabPFNModel(n_estimators=16, device='cuda', random_state=42, model_path=tabpfn_real_path),
        "name": "TabPFN-real",
    },
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "SOTA")
dataset_config = DatasetConfigs(DATA_PATH, task_type=TASK_TYPE)

# Run the pipeline
runner = PipelineRunner(save_files=True, verbose=0, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analyze and display top performing models
best_model_count = 5
rank_metric = 'rmse' if TASK_TYPE == "regression" else 'balanced_accuracy'
display_metrics = ['rmse', 'r2', 'mape', 'nrmse'] if TASK_TYPE == "regression" else ['accuracy', 'balanced_accuracy', 'f1', 'recall']

for dataset_name, dataset_prediction in predictions_per_dataset.items():
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    dataset_predictions = dataset_prediction['run_predictions']

    # Display top performing models
    print("Top Predictions (per row):")
    print("-" * 80)

    print(f"\nTop {best_model_count} models based on validation {rank_metric.upper()}:")
    top_models_val = dataset_predictions.top(best_model_count, rank_metric, rank_partition='val', display_metrics=display_metrics)
    for idx, prediction in enumerate(top_models_val):
        print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")

    print(f"\nTop {best_model_count} models based on test {rank_metric.upper()}:")
    top_models_test = dataset_predictions.top(best_model_count, rank_metric, rank_partition='test', display_metrics=display_metrics)
    for idx, prediction in enumerate(top_models_test):
        print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")

    # Print aggregated results if aggregation_key is provided
    if AGGREGATION_KEY is not None:
        print("*" * 80)
        print(f"\n Top Predictions (aggregated by {AGGREGATION_KEY}):")
        print("-" * 80)

        print(f"\nTop {best_model_count} models based on validation {rank_metric.upper()}:")
        top_models_val = dataset_predictions.top(best_model_count, rank_metric, rank_partition='val', display_metrics=display_metrics, aggregate=AGGREGATION_KEY)
        for idx, prediction in enumerate(top_models_val):
            print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")

        print(f"\nTop {best_model_count} models based on test {rank_metric.upper()}:")
        top_models_test = dataset_predictions.top(best_model_count, rank_metric, rank_partition='test', display_metrics=display_metrics, aggregate=AGGREGATION_KEY)
        for idx, prediction in enumerate(top_models_test):
            print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")