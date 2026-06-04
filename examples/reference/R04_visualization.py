"""
R04 - Prediction Analysis & Visualization Reference
===================================================

Reference for inspecting predictions and producing result charts with the
public ``Predictions`` object and ``PredictionAnalyzer`` API.

This reference covers:

* Accessing predictions from a ``RunResult`` (``result.predictions``)
* The ``PredictionAnalyzer`` chart methods:
  ``plot_top_k``, ``plot_heatmap``, ``plot_candlestick``, ``plot_histogram``
* Ranking options (``rank_metric``, ``rank_partition``)

This example is self-contained: it generates synthetic data with
``nirs4all.generate.regression`` so it needs no on-disk dataset, and it runs
plot-free by default (pass ``--plots`` to render figures).

Prerequisites
-------------
Complete :ref:`U04_visualization` for the tutorial walkthrough.

Duration: ~1 minute
Difficulty: **
"""

import argparse

import nirs4all
from nirs4all.visualization.predictions import PredictionAnalyzer

parser = argparse.ArgumentParser(description="R04 Prediction Analysis & Visualization Reference")
parser.add_argument("--plots", action="store_true", help="Render figures")
parser.add_argument("--show", action="store_true", help="Display figures interactively")
args = parser.parse_args()

print("=" * 60)
print("R04 - Prediction Analysis & Visualization Reference")
print("=" * 60)

# -----------------------------------------------------------------------------
# 1. Produce predictions: several models so the charts have something to compare
# -----------------------------------------------------------------------------
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.linear_model import ElasticNet, Ridge  # noqa: E402
from sklearn.model_selection import ShuffleSplit  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        {"y_processing": MinMaxScaler()},
        ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
        {"model": PLSRegression(n_components=5), "name": "PLS-5"},
        {"model": PLSRegression(n_components=10), "name": "PLS-10"},
        {"model": Ridge(alpha=1.0), "name": "Ridge"},
        {"model": ElasticNet(alpha=0.1), "name": "ElasticNet"},
    ],
    dataset=nirs4all.generate.regression(n_samples=300, random_state=42),
    name="R04_visualization",
    verbose=0,
    plots_visible=False,
)

# -----------------------------------------------------------------------------
# 2. The Predictions object lives on the result
# -----------------------------------------------------------------------------
predictions = result.predictions
print(f"\nresult.num_predictions = {result.num_predictions}")
print(f"best model = {result.best['model_name']}  (RMSE={result.best_rmse:.4f})")

# -----------------------------------------------------------------------------
# 3. PredictionAnalyzer chart reference
# -----------------------------------------------------------------------------
analyzer = PredictionAnalyzer(predictions, save=args.plots or args.show)

# Top-K predicted-vs-actual, ranked by validation RMSE
analyzer.plot_top_k(k=3, rank_metric="rmse")
print("\nplot_top_k(k=3, rank_metric='rmse')              - top 3 models, pred vs actual")

# Rank instead by R2 on the test partition
analyzer.plot_top_k(k=3, rank_metric="r2", rank_partition="test")
print("plot_top_k(k=3, rank_metric='r2', rank_partition='test')")

# Heatmap of model vs preprocessing
analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings", rank_metric="rmse", display_metric="rmse")
print("plot_heatmap(x_var='model_name', y_var='preprocessings')")

# Candlestick: score distribution per model
analyzer.plot_candlestick(variable="model_name", rank_metric="rmse")
print("plot_candlestick(variable='model_name')          - score spread per model")

# Histogram of scores
analyzer.plot_histogram(rank_metric="rmse")
print("plot_histogram(rank_metric='rmse')               - score distribution")

if args.show:
    import matplotlib.pyplot as plt

    plt.show()

print("\nDone. (pass --plots to render figures, --show to display them)")
