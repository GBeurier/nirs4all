from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression


import nirs4all
from nirs4all.data.synthetic import SyntheticNIRSGenerator, ComponentLibrary, SyntheticDatasetBuilder
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay,
    Detrend,
    Gaussian,
    Haar,
    Wavelet,
    ASLSBaseline
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)
from nirs4all.operators.splitters import SPXYGFold

from nirs4all.operators.models import FCKPLS
from fckpls_torch import FCKPLSTorch, TrainConfig

def get_synthetic_dataset(n_samples=100, n_features=50, random_state=42):
    """Create reproducible test dataset."""
    return (
        SyntheticDatasetBuilder(n_samples=n_samples, random_state=random_state)
        .with_features(
            wavelength_range=(1000, 2500),
            complexity="realistic",
            components=["water", "protein", "lipid", "starch"]
        )
        .with_targets(
            distribution="lognormal",
            range=(5, 50),
            component="protein"
        )
        .with_nonlinear_targets(
            interactions="polynomial",
            interaction_strength=0.3,
            hidden_factors=3
        )
        .with_partitions(train_ratio=0.8)
        .build()
    )

synthetic_dataset_small = get_synthetic_dataset(n_samples=150, n_features=500)
synthetic_dataset_mid = get_synthetic_dataset(n_samples=150, n_features=2500)
synthetic_dataset_large = get_synthetic_dataset(n_samples=1500, n_features=2500)


print("Datasets created:")
print(" - SMALL:\n", synthetic_dataset_small)
print(" - MID:\n", synthetic_dataset_mid)
print(" - LARGE:\n", synthetic_dataset_large)


n_trials = 25
n_pp_variants = 10
pls_max_components = 25

# PLS hyperparameter search space
pls_params = {
    "n_components": ('int', 3, pls_max_components),
}

pipeline = [
    ASLSBaseline(),
    {"y_processing": MinMaxScaler},
    SPXYGFold(n_splits=3),
    {
        "_cartesian_": [
            {"_or_": [None, SNV(), EMSC(), Detrend()]},
            {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
            {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
            {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("coif3")]},
        ],
        "count": n_pp_variants,
    },
    {
        "model": PLSRegression,
        "name": "PLS-raw_tuned",
        "finetune_params": {
            "n_trials": n_trials,
            "sample": "tpe",
            "verbose": 2,
            "approach": "grouped",
            "model_params": pls_params,
        },
    },
]

# result = nirs4all.run(
#     pipeline=pipeline,
#     dataset=synthetic_dataset_small,  #[synthetic_dataset_small, synthetic_dataset_mid, synthetic_dataset_large],
#     name="BasicMulti",
#     verbose=1,
#     random_state=42
# )

# print(f"\nNumber of model configurations: {result.num_predictions}")
# print(f"Best RMSE: {result.best_score:.4f}")

# # Show all results
# print("\nAll results (ranked by RMSE):")
# for i, pred in enumerate(result.top(10, display_metrics=['rmse', 'r2']), 1):
#     model_name = pred.get('model_name', 'Unknown')
#     print(f"   {i}. {model_name}: RMSE={pred.get('rmse', 0):.4f}")






pipeline = [
    ASLSBaseline(),
    {"y_processing": MinMaxScaler},
    SPXYGFold(n_splits=3),
    FCKPLSTorch
]

result = nirs4all.run(
    pipeline=pipeline,
    dataset=synthetic_dataset_small,  #[synthetic_dataset_small, synthetic_dataset_mid, synthetic_dataset_large],
    name="BasicMulti",
    verbose=1,
    random_state=42
)

print(f"\nNumber of model configurations: {result.num_predictions}")
print(f"Best RMSE: {result.best_score:.4f}")

# Show all results
print("\nAll results (ranked by RMSE):")
for i, pred in enumerate(result.top(10, display_metrics=['rmse', 'r2']), 1):
    model_name = pred.get('model_name', 'Unknown')
    print(f"   {i}. {model_name}: RMSE={pred.get('rmse', 0):.4f}")