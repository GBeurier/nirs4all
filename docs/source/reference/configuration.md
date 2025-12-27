# Configuration Reference

This page provides the complete specification for `PipelineConfigs` and `DatasetConfigs`.

## PipelineConfigs

`PipelineConfigs` defines the processing pipeline: preprocessing steps, cross-validation, and models.

### Constructor

```python
from nirs4all.pipeline import PipelineConfigs

config = PipelineConfigs(
    definition,                   # Pipeline definition (list, dict, or path)
    name="",                      # Pipeline name
    description="",               # Optional description
    max_generation_count=10000    # Maximum pipeline variants to generate
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `definition` | list, dict, str | *required* | Pipeline steps as list, dict with `pipeline` key, or path to YAML/JSON |
| `name` | str | `""` | Pipeline name (used in artifacts and results) |
| `description` | str | `""` | Human-readable description |
| `max_generation_count` | int | `10000` | Maximum pipeline variants from generators |

### Definition Formats

#### List of Steps (Recommended)

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

pipeline = PipelineConfigs([
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=10)}
], name="MyPipeline")
```

#### Dictionary with Pipeline Key

```python
pipeline = PipelineConfigs({
    "pipeline": [
        MinMaxScaler(),
        ShuffleSplit(n_splits=3),
        {"model": PLSRegression(n_components=10)}
    ]
}, name="MyPipeline")
```

#### YAML File Path

```python
pipeline = PipelineConfigs("config/pipeline.yaml", name="MyPipeline")
```

**pipeline.yaml:**
```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 3
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

#### JSON File Path

```python
pipeline = PipelineConfigs("config/pipeline.json", name="MyPipeline")
```

### Step Serialization

Steps are serialized to a canonical format:

| Input | Serialized Form |
|-------|-----------------|
| `MinMaxScaler()` | `{"class": "sklearn.preprocessing.MinMaxScaler"}` |
| `PLSRegression(n_components=10)` | `{"class": "...", "params": {"n_components": 10}}` |
| `{"model": PLSRegression()}` | `{"model": {"class": "..."}}` |

### Accessing Pipeline Configurations

```python
pipeline = PipelineConfigs([...], name="MyPipeline")

# Access expanded configurations (list of step lists)
pipeline.steps           # List of step configurations

# Access names (includes hash for uniqueness)
pipeline.names           # ["MyPipeline_a1b2c3"]

# Check if generators were used
pipeline.has_configurations  # True if _or_, _range_ expanded
```

---

## DatasetConfigs

`DatasetConfigs` defines how to load and configure datasets.

### Constructor

```python
from nirs4all.data import DatasetConfigs

dataset = DatasetConfigs(
    configurations,              # Path(s) or configuration dict(s)
    task_type="auto",            # Force task type
    signal_type=None,            # Override signal type
    aggregate=None,              # Aggregation column or True
    aggregate_method=None,       # Aggregation method
    aggregate_exclude_outliers=None  # Exclude outliers before aggregation
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `configurations` | str, dict, list | *required* | Path, config dict, or list of either |
| `task_type` | str, list | `"auto"` | Task type per dataset |
| `signal_type` | str, list | `None` | Signal type override |
| `aggregate` | str, bool, list | `None` | Aggregation setting |
| `aggregate_method` | str, list | `None` | Method: "mean", "median", "vote" |
| `aggregate_exclude_outliers` | bool, list | `None` | Exclude outliers via T² |

### Configuration Dictionary Keys

#### Data File Keys

| Key | Description | Example |
|-----|-------------|---------|
| `train_x` | Training features | `"spectra_train.csv"` |
| `train_y` | Training targets | `"targets_train.csv"` |
| `train_m` | Training metadata | `"metadata_train.csv"` |
| `test_x` | Test features | `"spectra_test.csv"` |
| `test_y` | Test targets | `"targets_test.csv"` |
| `test_m` | Test metadata | `"metadata_test.csv"` |

#### Parameter Keys

| Key | Description |
|-----|-------------|
| `train_x_params` | Parameters for `train_x` file |
| `train_y_params` | Parameters for `train_y` file |
| `test_x_params` | Parameters for `test_x` file |
| `global_params` | Parameters applied to all files |

#### File Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delimiter` | str | `","` | Column separator |
| `decimal_separator` | str | `"."` | Decimal point character |
| `has_header` | bool | `True` | First row is header |
| `header_unit` | str | `"auto"` | Header interpretation |
| `signal_type` | str | `"auto"` | Spectral signal type |
| `na_policy` | str | `"drop"` | Missing value handling |
| `target_column` | str | `None` | Target column name (combined files) |
| `sheet_name` | str | `None` | Excel sheet name |

#### Header Unit Options

| Value | Description |
|-------|-------------|
| `"nm"` | Wavelengths in nanometers |
| `"cm-1"` | Wavenumbers in cm⁻¹ |
| `"none"` | No header row |
| `"text"` | Text labels (ignored) |
| `"index"` | Numeric indices |
| `"auto"` | Automatic detection |

#### Signal Type Options

| Value | Description |
|-------|-------------|
| `"absorbance"` | Absorbance values |
| `"reflectance"` | Reflectance 0-1 |
| `"reflectance%"` | Reflectance 0-100 |
| `"transmittance"` | Transmittance 0-1 |
| `"transmittance%"` | Transmittance 0-100 |
| `"auto"` | Automatic detection |

#### NA Policy Options

| Value | Description |
|-------|-------------|
| `"drop"` | Drop rows with missing values |
| `"fill_mean"` | Fill with column mean |
| `"fill_median"` | Fill with column median |
| `"fill_zero"` | Fill with zeros |

### Task Type Options

| Value | Description |
|-------|-------------|
| `"auto"` | Auto-detect from targets |
| `"regression"` | Continuous target prediction |
| `"binary_classification"` | Two-class classification |
| `"multiclass_classification"` | Multi-class classification |

### Configuration Examples

#### Simple Path

```python
dataset = DatasetConfigs("path/to/data/")
```

#### Explicit Files

```python
dataset = DatasetConfigs({
    "train_x": "spectra_train.csv",
    "train_y": "targets_train.csv",
    "test_x": "spectra_test.csv",
    "test_y": "targets_test.csv"
})
```

#### With Parameters

```python
dataset = DatasetConfigs({
    "train_x": "spectra.csv",
    "train_y": "targets.csv",
    "train_x_params": {
        "header_unit": "nm",
        "signal_type": "reflectance",
        "delimiter": ";"
    },
    "train_y_params": {
        "has_header": True
    }
})
```

#### Multi-Source Dataset

```python
dataset = DatasetConfigs({
    "train_x": ["nir_spectra.csv", "markers.csv"],
    "train_y": "targets.csv",
    "train_x_params": [
        {"header_unit": "nm", "signal_type": "reflectance"},
        {"header_unit": "text"}
    ]
})
```

#### Multiple Datasets

```python
dataset = DatasetConfigs([
    "dataset1/",
    "dataset2/",
    {"train_x": "custom/spectra.csv", "train_y": "custom/targets.csv"}
])
```

#### With Aggregation

```python
dataset = DatasetConfigs(
    "path/to/data/",
    aggregate="sample_id",           # Column name in metadata
    aggregate_method="mean",         # "mean", "median", or "vote"
    aggregate_exclude_outliers=True  # Remove outliers before aggregating
)
```

### Accessing Dataset Data

```python
dataset = DatasetConfigs("path/to/data/")

# Iterate over datasets
for ds in dataset.iter_datasets():
    print(f"Dataset: {ds.name}")
    print(f"  Samples: {len(ds)}")
    print(f"  Features: {ds.n_features}")
    print(f"  Task: {ds.task_type}")

# Get specific dataset by index
ds = dataset.get_dataset_at(0)

# Get all datasets as list
all_datasets = dataset.get_datasets()
```

---

## Complete Examples

### Full Pipeline Configuration

```python
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import StandardNormalVariate

# Pipeline configuration
pipeline = PipelineConfigs([
    MinMaxScaler(),
    StandardNormalVariate(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=10)}
], name="ProductionPipeline", description="NIR protein prediction model")

# Dataset configuration
dataset = DatasetConfigs({
    "train_x": "data/spectra.csv",
    "train_y": "data/protein.csv",
    "train_m": "data/samples.csv",
    "train_x_params": {
        "header_unit": "nm",
        "signal_type": "reflectance",
        "delimiter": ","
    }
}, task_type="regression", aggregate="sample_id")

# Run
runner = PipelineRunner(verbose=1, save_artifacts=True)
predictions, per_dataset = runner.run(pipeline, dataset)
```

### YAML Configuration File

**pipeline.yaml:**
```yaml
pipeline:
  # Preprocessing
  - class: sklearn.preprocessing.MinMaxScaler

  - class: nirs4all.operators.transforms.StandardNormalVariate

  # Target scaling
  - y_processing:
      class: sklearn.preprocessing.MinMaxScaler

  # Cross-validation
  - class: sklearn.model_selection.ShuffleSplit
    params:
      n_splits: 5
      test_size: 0.25
      random_state: 42

  # Model
  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

**dataset.yaml:**
```yaml
train_x: data/spectra.csv
train_y: data/targets.csv
train_x_params:
  header_unit: nm
  signal_type: reflectance
  delimiter: ","
task_type: regression
```

**Python usage:**
```python
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs

pipeline = PipelineConfigs("config/pipeline.yaml", name="YAMLPipeline")
dataset = DatasetConfigs("config/dataset.yaml")

runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(pipeline, dataset)
```

## See Also

- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- {doc}`/reference/generator_keywords` - Generator syntax (`_or_`, `_range_`)
- {doc}`/user_guide/data/loading_data` - Data loading guide
- {doc}`/getting_started/concepts` - Core concepts overview
