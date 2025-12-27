# Metrics Reference

This page documents all evaluation metrics available in NIRS4ALL.

## Overview

NIRS4ALL automatically computes appropriate metrics based on the task type:

| Task Type | Default Metric | Direction |
|-----------|----------------|-----------|
| Regression | MSE | Lower is better ↓ |
| Binary Classification | Balanced Accuracy | Higher is better ↑ |
| Multiclass Classification | Balanced Accuracy | Higher is better ↑ |

## Regression Metrics

### Core Metrics

| Metric | Abbreviation | Formula | Range | Direction |
|--------|--------------|---------|-------|-----------|
| **MSE** | MSE | $\frac{1}{n}\sum(y_{true} - y_{pred})^2$ | [0, ∞) | ↓ |
| **RMSE** | RMSE | $\sqrt{MSE}$ | [0, ∞) | ↓ |
| **MAE** | MAE | $\frac{1}{n}\sum\|y_{true} - y_{pred}\|$ | [0, ∞) | ↓ |
| **R²** | R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | (-∞, 1] | ↑ |
| **MAPE** | MAPE | $\frac{100}{n}\sum\|\frac{y_{true} - y_{pred}}{y_{true}}\|$ | [0, ∞) | ↓ |

### NIRS-Specific Metrics

| Metric | Abbreviation | Description | Range | Direction |
|--------|--------------|-------------|-------|-----------|
| **Bias** | Bias | Mean error: $\bar{y_{pred} - y_{true}}$ | (-∞, ∞) | → 0 |
| **SEP** | SEP | Standard Error of Prediction | [0, ∞) | ↓ |
| **RPD** | RPD | Ratio of Performance to Deviation: $\frac{SD(y_{true})}{SEP}$ | [0, ∞) | ↑ |
| **Consistency** | Cons | $1 - \frac{RMSE}{SD(y_{true})}$ | (-∞, 1] | ↑ |

### Additional Metrics

| Metric | Abbreviation | Description | Range | Direction |
|--------|--------------|-------------|-------|-----------|
| **Explained Variance** | ExpVar | Proportion of variance explained | (-∞, 1] | ↑ |
| **Max Error** | MaxErr | Maximum absolute error | [0, ∞) | ↓ |
| **Median AE** | MedAE | Median absolute error | [0, ∞) | ↓ |
| **NRMSE** | NRMSE | RMSE / (max - min) | [0, ∞) | ↓ |
| **NMSE** | NMSE | MSE / variance | [0, ∞) | ↓ |
| **NMAE** | NMAE | MAE / (max - min) | [0, ∞) | ↓ |
| **Pearson R** | Pearson | Pearson correlation coefficient | [-1, 1] | ↑ |
| **Spearman R** | Spearman | Spearman rank correlation | [-1, 1] | ↑ |

### Metric Descriptions

#### MSE (Mean Squared Error)
Measures the average squared difference between predictions and true values. Penalizes large errors more than small ones.

```python
# In NIRS4ALL
metrics = result.top(n=5, display_metrics=['mse'])
```

#### RMSE (Root Mean Squared Error)
Square root of MSE, in the same units as the target variable. Most commonly used regression metric.

```python
# Default ranking metric for regression
result.top(n=5)  # Ranks by RMSE
```

#### R² (Coefficient of Determination)
Proportion of variance in the target explained by the model. R² = 1 is perfect, R² = 0 means no better than mean.

```python
result.top(n=5, display_metrics=['r2'])
```

#### RPD (Ratio of Performance to Deviation)
Common in NIRS literature. Indicates model quality:

| RPD Value | Interpretation |
|-----------|----------------|
| < 1.5 | Not usable |
| 1.5 - 2.0 | Rough screening |
| 2.0 - 2.5 | Good screening |
| 2.5 - 3.0 | Good quantification |
| > 3.0 | Excellent quantification |

#### SEP (Standard Error of Prediction)
Standard deviation of prediction errors. Indicates spread of errors around bias.

#### Bias
Mean error. Positive bias means model over-predicts on average.

---

## Classification Metrics

### Core Metrics

| Metric | Abbreviation | Description | Range | Direction |
|--------|--------------|-------------|-------|-----------|
| **Accuracy** | Acc | Correct predictions / total | [0, 1] | ↑ |
| **Balanced Accuracy** | BalAcc | Mean recall per class | [0, 1] | ↑ |
| **Precision** | Prec | TP / (TP + FP), weighted | [0, 1] | ↑ |
| **Recall** | Rec | TP / (TP + FN), weighted | [0, 1] | ↑ |
| **F1 Score** | F1 | Harmonic mean of precision & recall | [0, 1] | ↑ |
| **Specificity** | Spec | TN / (TN + FP) | [0, 1] | ↑ |

### Advanced Metrics

| Metric | Abbreviation | Description | Range | Direction |
|--------|--------------|-------------|-------|-----------|
| **ROC AUC** | AUC | Area under ROC curve | [0, 1] | ↑ |
| **MCC** | MCC | Matthews correlation coefficient | [-1, 1] | ↑ |
| **Cohen's Kappa** | Kappa | Agreement adjusted for chance | [-1, 1] | ↑ |
| **Log Loss** | LogLoss | Cross-entropy loss | [0, ∞) | ↓ |
| **Jaccard** | Jaccard | Intersection over union | [0, 1] | ↑ |
| **Hamming Loss** | Hamming | Fraction of wrong labels | [0, 1] | ↓ |

### Averaging Methods

For multiclass problems, metrics use different averaging:

| Suffix | Method | Description |
|--------|--------|-------------|
| (none) | Weighted | Weighted by class frequency (default) |
| `_micro` | Micro | Global TP, FP, FN counts |
| `_macro` | Macro | Unweighted mean per class |
| `balanced_*` | Macro | Same as macro average |

```python
# Available multiclass metrics
result.top(n=5, display_metrics=['accuracy', 'balanced_accuracy', 'f1_macro'])
```

### Metric Descriptions

#### Balanced Accuracy
Mean of recall for each class. Handles imbalanced datasets better than accuracy.

```python
# Default for classification
result.top(n=5)  # Uses balanced_accuracy
```

#### MCC (Matthews Correlation Coefficient)
Correlation between predicted and true classes. Considers all four confusion matrix quadrants. Recommended for imbalanced datasets.

| MCC Value | Interpretation |
|-----------|----------------|
| +1 | Perfect prediction |
| 0 | Random prediction |
| -1 | Inverse prediction |

#### ROC AUC
Area under the Receiver Operating Characteristic curve. Measures discrimination ability across all classification thresholds.

---

## Using Metrics in Code

### Accessing Metrics in Results

```python
result = nirs4all.run(pipeline, dataset)

# Get top results with specific metrics
for pred in result.top(n=5, display_metrics=['rmse', 'r2', 'mae']):
    print(f"RMSE: {pred['rmse']:.4f}, R²: {pred['r2']:.4f}, MAE: {pred['mae']:.4f}")
```

### Ranking by Different Metrics

```python
# Rank by RMSE (default for regression)
top_by_rmse = result.top(n=5, rank_metric='rmse')

# Rank by R²
top_by_r2 = result.top(n=5, rank_metric='r2')

# Rank by custom metric
top_by_mae = result.top(n=5, rank_metric='mae')
```

### Metric Abbreviations

NIRS4ALL provides abbreviations for display:

```python
from nirs4all.core.metrics import abbreviate_metric

abbreviate_metric('balanced_accuracy')  # Returns 'BalAcc'
abbreviate_metric('mean_squared_error')  # Returns 'MSE'
abbreviate_metric('r2')                  # Returns 'R²'
```

### Computing Metrics Manually

```python
from nirs4all.core.metrics import eval, eval_multi

# Single metric
rmse = eval(y_true, y_pred, 'rmse')

# All metrics for task type
metrics = eval_multi(y_true, y_pred, 'regression')
# Returns: {'mse': 0.01, 'rmse': 0.1, 'mae': 0.08, 'r2': 0.95, ...}
```

### Getting Available Metrics

```python
from nirs4all.core.metrics import get_available_metrics, get_default_metrics

# All available
all_reg = get_available_metrics('regression')
all_cls = get_available_metrics('binary_classification')

# Commonly used
default_reg = get_default_metrics('regression')
# ['r2', 'rmse', 'mse', 'sep', 'mae', 'rpd', 'bias', ...]
```

---

## Metric Selection Guidelines

### For Regression

| Scenario | Recommended Metrics |
|----------|---------------------|
| General purpose | RMSE, R², MAE |
| NIRS literature | RMSE, R², RPD, SEP |
| Outlier-sensitive | MAE, Median AE |
| Relative errors | MAPE, NRMSE |
| Correlation focus | Pearson R, Spearman R |

### For Classification

| Scenario | Recommended Metrics |
|----------|---------------------|
| Balanced classes | Accuracy, F1 |
| Imbalanced classes | Balanced Accuracy, MCC, ROC AUC |
| Cost-sensitive | Precision or Recall (depending on cost) |
| Binary problems | Accuracy, AUC, F1 |
| Multiclass problems | Balanced Accuracy, F1 Macro |

---

## Complete Example

```python
import nirs4all
from nirs4all.core.metrics import eval_multi, get_default_metrics

# Run pipeline
result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        ShuffleSplit(n_splits=5),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset="sample_data/regression",
    verbose=1
)

# View multiple metrics
print("📊 Top 5 Models by RMSE:")
for pred in result.top(n=5, display_metrics=['rmse', 'r2', 'mae', 'sep', 'rpd']):
    print(f"  {pred['model_name']}:")
    print(f"    RMSE: {pred['rmse']:.4f}")
    print(f"    R²: {pred['r2']:.4f}")
    print(f"    MAE: {pred['mae']:.4f}")
    print(f"    SEP: {pred.get('sep', 'N/A')}")
    print(f"    RPD: {pred.get('rpd', 'N/A')}")

# Compute all metrics for best model
best = result.best
y_true = best['y_true']
y_pred = best['y_pred']
all_metrics = eval_multi(y_true, y_pred, 'regression')

print("\n📈 All Regression Metrics:")
for metric, value in all_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

## See Also

- {doc}`/user_guide/models/training` - Model training basics
- {doc}`/reference/predictions_api` - Working with prediction results
- {doc}`/user_guide/visualization/prediction_charts` - Visualizing metrics
