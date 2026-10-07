# Splitters Reference

Your independent unit may be a specimen, subject, batch or instrument. The splitter assigns these units to training and validation. Random splitting, stratification, group splitting and calibration sampling answer different questions. The tables enumerate each strategy and its parameters.

## Choose what must remain unseen before choosing a splitter

Read {doc}`nodes/split` for the worked result figure, expected dimensions and exercises. Choose a family below to compare the enumerated operators.

## Same worked recipe in JSON, YAML and Python

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "pipeline": [
    {
      "split": {
        "class": "sklearn.model_selection.KFold",
        "params": {
          "n_splits": 3,
          "shuffle": true,
          "random_state": 42
        }
      }
    },
    {
      "model": {
        "class": "sklearn.cross_decomposition.PLSRegression",
        "params": {
          "n_components": 2
        }
      }
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
pipeline:
- split:
    class: sklearn.model_selection.KFold
    params:
      n_splits: 3
      shuffle: true
      random_state: 42
- model:
    class: sklearn.cross_decomposition.PLSRegression
    params:
      n_components: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    KFold(n_splits=3, shuffle=True, random_state=42),
    {"model": PLSRegression(n_components=2)},
]
```
:::

::::

These are Python SDK operators. Native R/Octave/WASM recipes use their own method IDs and facade; see {doc}`/guide/languages`.

---

## NIRS-Specific Splitters

All splitters below are imported from `nirs4all.operators.splitters`.

### Single-Split Methods

These produce a single train/test split.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `KennardStoneSplitter` | `test_size`, `pca_components=None`, `metric="euclidean"` | Kennard-Stone algorithm: selects maximally diverse training samples using max-min distance criterion |
| `SPXYSplitter` | `test_size`, `pca_components=None`, `metric="euclidean"` | SPXY (Sample set Partitioning based on joint X and Y distances): Kennard-Stone extended to include target information |
| `KMeansSplitter` | `test_size`, `pca_components=None`, `metric="euclidean"` | K-Means clustering-based split: clusters samples and assigns cluster centers to training |
| `KBinsStratifiedSplitter` | `test_size`, `n_bins=10`, `strategy="uniform"`, `encode="ordinal"` | Stratified sampling using KBins discretization of continuous targets |
| `SystematicCircularSplitter` | `test_size` | Systematic circular sampling: orders by y-value, then selects at regular intervals |
| `SPlitSplitter` | `test_size` | Data twinning algorithm (Vakayil & Joseph 2022): selects a statistically representative subset |

### K-Fold Methods

These produce multiple folds for cross-validation.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `SPXYFold` | `n_splits=5`, `metric="euclidean"`, `y_metric="euclidean"`, `pca_components=None` | SPXY-based K-Fold: assigns samples to folds using joint X-Y distances for spatially representative folds |
| `SPXYGFold` | `n_splits=5`, `metric="euclidean"`, `y_metric="euclidean"`, `pca_components=None` | Group-aware SPXY K-Fold: respects group boundaries while using SPXY distance criterion |
| `BinnedStratifiedGroupKFold` | `n_splits=5`, `n_bins=10`, `strategy="quantile"`, `shuffle=False` | Stratified Group K-Fold with binned continuous targets: ensures balanced target distribution across folds while respecting groups |

### Group Wrapper

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `GroupedSplitterWrapper` | `splitter`, `aggregation="mean"`, `y_aggregation=None` | Wraps any sklearn splitter to add group-awareness; aggregates samples by group and ensures no group leakage |

Usage:
::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "class": "nirs4all.operators.splitters.grouped_wrapper.GroupedSplitterWrapper",
  "params": {
    "splitter": "sklearn.model_selection._split.KFold"
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
class: nirs4all.operators.splitters.grouped_wrapper.GroupedSplitterWrapper
params:
  splitter: sklearn.model_selection._split.KFold
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.operators.splitters import GroupedSplitterWrapper
from sklearn.model_selection import KFold

# Any splitter becomes group-aware
wrapper = GroupedSplitterWrapper(KFold(n_splits=5))
```
:::

::::

---

## Commonly Used sklearn Splitters

These are imported from `sklearn.model_selection` and work directly in nirs4all pipelines.

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `KFold` | `n_splits=5`, `shuffle=False` | Standard K-Fold cross-validation |
| `StratifiedKFold` | `n_splits=5`, `shuffle=False` | K-Fold with stratification on target classes |
| `ShuffleSplit` | `n_splits=10`, `test_size=0.1` | Random train/test splits with configurable sizes |
| `RepeatedKFold` | `n_splits=5`, `n_repeats=10` | Repeated K-Fold cross-validation |
| `LeaveOneOut` | *(none)* | Leave-one-out cross-validation |
| `GroupKFold` | `n_splits=5` | K-Fold respecting group boundaries |
| `StratifiedGroupKFold` | `n_splits=5`, `shuffle=False` | Stratified K-Fold respecting groups |

---

## SPXYFold Parameters Detail

`SPXYFold` supports several configurations for different use cases:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `y_metric` | `"euclidean"` | For regression (continuous y) -- default SPXY behavior |
| `y_metric` | `"hamming"` | For classification (categorical y) |
| `y_metric` | `None` | Ignore Y (pure Kennard-Stone, X-only selection) |
| `pca_components` | `int` or `None` | Apply PCA dimensionality reduction before distance computation |

---

## See Also

- {doc}`../reference/pipeline_keywords` -- Pipeline keyword syntax
- {doc}`../reference/filters` -- Sample filtering operators
- {doc}`../reference/models` -- Built-in models reference

## Choose the independent unit before the algorithm

```{figure} /assets/guide/split.svg
:alt: Each validation fold holds out distinct independent observations.

Educational workflow result; read the accompanying explanation for scope and interpretation.
```

The split defines the generalization claim. New samples, new physical
subjects, future acquisitions and unseen instruments are different questions.
Repeated scans, pixels from one specimen, augmented copies and repeated
source observations do not become independent because they occupy different
rows. Group them by the appropriate origin before assigning folds.

| Strategy | Why choose it | What it does not establish |
|---|---|---|
| Shuffled KFold/ShuffleSplit | Exchangeable independent samples from one population | Generalization to unseen batches or future drift |
| StratifiedKFold | Preserve class coverage across classification folds | Independence between repeated measurements |
| GroupKFold/GroupedSplitterWrapper | Hold out complete origins, subjects or batches | Class or target balance without a separate balancing policy |
| BinnedStratifiedGroupKFold | Balance continuous-target bins while preserving groups | Exact balance when groups are large or scarce |
| Kennard–Stone | Cover feature-space diversity in calibration rows | Random-population test performance or unseen-target extrapolation |
| SPXY/SPXYFold/SPXYGFold | Represent joint X/y space, optionally respecting groups | Prospective selection without labels for future samples |
| KMeansSplitter | Representative calibration based on cluster structure | A guarantee of coverage for small rare subgroups |
| KBinsStratified/SystematicCircular | Spread target ranges across partitions | A deployment drift test |
| SPlitSplitter | Representative subset through data twinning | Independence if repeated origins were not grouped |
| TimeSeriesSplit | Respect temporal ordering in ordered observations | Protection against overlap when one origin spans dates |
| LeaveOneOut | Train on nearly all available independent samples | Low variance of the estimated score or freedom from selection bias |

Kennard–Stone uses a max-min distance criterion to spread selected calibration
samples. The metric, scaling and optional PCA therefore define what “diverse”
means. SPXY adds target distances: selection uses known targets and must be
described as retrospective calibration design. Group-aware variants avoid
splitting origins, but still require sufficient independent groups in every
fold. Stratification and grouping solve different problems.

## Folds are part of the experiment

The number of folds changes training size and compute. Repeated folds increase
resampling evidence but are correlated because rows recur. Report how scores
are aggregated; an unweighted mean of fold RMSE is not pooled RMSE when folds
have unequal sizes. Avoid selecting a preprocessing or model using the final
test fold. Use nested selection or a separate development/holdout design.

Record row IDs, group/origin relations, partition roles, seeds and any
distance-based representation used to create the folds. Reusing identical
folds makes candidate comparisons easier to interpret. Resume and artifact
audit also depend on these identities, not merely on `random_state=42`.

For runnable group binding see `U02_group_splitting.py`; for comparisons run
`U01_cv_strategies.py`. The `split` node, fold files and group metadata are
documented in {doc}`nodes/split`. Check {doc}`/guide/evaluation` before adding
nested search, ensembles or calibration on top of a splitter.
