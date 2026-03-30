# Datasets

Every pipeline operates on a **SpectroDataset** -- the core data container in
nirs4all. It holds feature matrices, target values, metadata, and fold
assignments in a single, self-describing object.

---

## What Is a SpectroDataset?

At its simplest, a SpectroDataset is a bundle of arrays and indices:

| Component    | Description                                           |
|------------- |-------------------------------------------------------|
| **X**        | Feature matrix -- spectral data (samples x features)  |
| **y**        | Target vector -- the values you want to predict       |
| **metadata** | Sample-level information (IDs, groups, dates, etc.)   |
| **folds**    | Cross-validation fold assignments (written by the splitter) |

Think of it as a smart DataFrame that knows about spectroscopy: it tracks
signal types, wavelength headers, processing history, and sample identity.

```
SpectroDataset
  |-- X:  (200, 1050)   200 samples, 1050 wavelengths
  |-- y:  (200,)         200 target values
  |-- metadata:          sample IDs, site labels, dates
  |-- folds:             fold 0..4 train/val assignments
```

---

## Partitions

Data is organised into three partitions:

| Partition | Purpose                                           |
|-----------|---------------------------------------------------|
| **train** | Used for cross-validation (split into folds)      |
| **test**  | Held-out data for final evaluation                |
| **val**   | Created automatically by the splitter from train  |

The **test** partition is never used during model selection or training. It is
evaluated once after refit to produce the final performance score (RMSEP in
chemometrics terminology).

The **val** partition is temporary: it exists only within a given
cross-validation fold. In fold 0, some training samples become the validation
set; in fold 1, different samples take that role.

:::{note}
If you provide only training data (no test split), nirs4all still runs
cross-validation. You will get RMSECV but no RMSEP.
:::

---

## Loading Data

nirs4all accepts many input formats. You do not need to build a
SpectroDataset by hand -- the library creates one for you:

| Input                     | Example                                       |
|---------------------------|-----------------------------------------------|
| Folder path               | `"sample_data/regression"`                    |
| NumPy arrays              | `(X, y)` or `X` alone                        |
| Dictionary                | `{"X": X, "y": y, "metadata": meta}`         |
| SpectroDataset instance   | A pre-built dataset object                    |
| DatasetConfigs object     | Full configuration with loader settings       |

```python
# Simplest: point at a folder
result = nirs4all.run(pipeline=pipeline, dataset="sample_data/regression")

# From arrays
result = nirs4all.run(pipeline=pipeline, dataset=(X, y))
```

Folder-based loading auto-detects file formats (CSV, Parquet, Excel, NumPy,
MATLAB) and infers which files contain features, targets, and metadata. See
{doc}`/user_guide/data/loading_data` for details.

---

## Multi-Source Data

A dataset can hold more than one feature source. This is common when you
combine data from different instruments or different measurement types:

```
Source 0: NIR spectra        (200, 1050)
Source 1: chemical markers   (200, 20)
```

Each source can have:
- A different number of features (1050 wavelengths vs. 20 markers).
- Different wavelength headers or column names.
- Independent preprocessing chains.

Sources share the same samples and the same target vector. You can process
them separately (using source-specific branches) and merge the results before
the model step, or feed them into a multi-input model.

See {doc}`/user_guide/pipelines/multi_source` for pipeline patterns with
multi-source datasets.

---

## Signal Types

Spectral data comes in several physical representations. nirs4all tracks the
signal type of each source so that transforms and conversions behave
correctly:

| Signal type     | Description                                         |
|-----------------|-----------------------------------------------------|
| Absorbance      | Most common for NIRS. Related to concentration via Beer-Lambert law. |
| Reflectance     | Raw reflectance (0 to 1 or 0 to 100%).              |
| Transmittance   | Fraction of light transmitted through the sample.    |
| Log(1/R)        | Logarithmic reflectance, approximation of absorbance.|

Signal type can be **auto-detected** based on value ranges and distribution,
or **set manually**:

```python
# Auto-detection
signal_type, confidence, reason = dataset.detect_signal_type(src=0)

# Manual override
dataset.set_signal_type("absorbance", src=0, forced=True)

# Conversion
dataset.convert_to_absorbance(src=0)
```

See {doc}`/user_guide/data/signal_types` for detection logic and conversion
details.

---

## Repetitions

In NIRS, it is common to measure the same physical sample multiple times.
These repeated measurements must be kept together during cross-validation --
otherwise the model can see repetitions of a sample in both the training and
validation folds, which causes data leakage and optimistic scores.

SpectroDataset handles this through a **repetition column**:

```python
dataset.set_repetition("Sample_ID")
```

Once set:
- Splitters group all repetitions of a sample into the same fold.
- Predictions can be aggregated at the sample level (mean, median, or vote).

:::{warning}
Without repetition-aware splitting, a dataset with 50 soil samples measured 3
times each (150 spectra) would allow the same soil sample to appear in both
train and val. This leads to optimistic evaluation because the model partly
memorises individual samples.
:::

See {doc}`/user_guide/data/aggregation` for aggregation methods and outlier
handling.

---

## Synthetic Data

For testing and prototyping, nirs4all can generate realistic synthetic NIRS
spectra:

```python
import nirs4all

dataset = nirs4all.generate.regression(n_samples=500)
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
```

Synthetic data is useful for verifying that a pipeline runs correctly before
applying it to real data, or for benchmarking new preprocessing methods. See
{doc}`/user_guide/data/synthetic_data` for the full generation API.

---

## Key Properties

Once a SpectroDataset is loaded, it exposes several useful properties:

| Property             | Returns                                      |
|----------------------|----------------------------------------------|
| `dataset.X`         | Feature matrix for the default source        |
| `dataset.y`         | Target vector                                |
| `dataset.n_samples` | Number of samples                            |
| `dataset.n_features`| Number of features in the default source     |
| `dataset.task_type` | `REGRESSION`, `BINARY_CLASSIFICATION`, or `MULTICLASS_CLASSIFICATION` |
| `dataset.signal_type()` | Signal type of a given source            |
| `dataset.repetition_stats` | Repetition group statistics            |

---

## Next Steps

- {doc}`pipelines` -- understand the steps that operate on datasets.
- {doc}`cross_validation` -- learn how folds are created from the train
  partition.
- {doc}`/user_guide/data/loading_data` -- complete loading reference.
- {doc}`/user_guide/data/signal_types` -- signal detection and conversion.
