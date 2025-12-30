# nirs4all.data.synthetic

Low-level synthetic NIRS data generation module.

```{eval-rst}
.. currentmodule:: nirs4all.data.synthetic
```

## Overview

This module provides the core classes for generating synthetic NIRS spectra. For most use cases, prefer the high-level `nirs4all.generate()` API.

## Core Classes

### SyntheticNIRSGenerator

```{eval-rst}
.. autoclass:: SyntheticNIRSGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```

### SyntheticDatasetBuilder

```{eval-rst}
.. autoclass:: SyntheticDatasetBuilder
   :members:
   :undoc-members:
   :show-inheritance:
```

## Spectral Components

### NIRBand

```{eval-rst}
.. autoclass:: NIRBand
   :members:
```

### SpectralComponent

```{eval-rst}
.. autoclass:: SpectralComponent
   :members:
```

### ComponentLibrary

```{eval-rst}
.. autoclass:: ComponentLibrary
   :members:
   :undoc-members:
```

## Configuration Classes

### SyntheticDatasetConfig

```{eval-rst}
.. autoclass:: SyntheticDatasetConfig
   :members:
```

### FeatureConfig

```{eval-rst}
.. autoclass:: FeatureConfig
   :members:
```

### TargetConfig

```{eval-rst}
.. autoclass:: TargetConfig
   :members:
```

### MetadataConfig

```{eval-rst}
.. autoclass:: MetadataConfig
   :members:
```

### PartitionConfig

```{eval-rst}
.. autoclass:: PartitionConfig
   :members:
```

### BatchEffectConfig

```{eval-rst}
.. autoclass:: BatchEffectConfig
   :members:
```

## Metadata Generation

### MetadataGenerator

```{eval-rst}
.. autoclass:: MetadataGenerator
   :members:
```

### MetadataGenerationResult

```{eval-rst}
.. autoclass:: MetadataGenerationResult
   :members:
```

## Target Generation

### TargetGenerator

```{eval-rst}
.. autoclass:: TargetGenerator
   :members:
```

### ClassSeparationConfig

```{eval-rst}
.. autoclass:: ClassSeparationConfig
   :members:
```

## Multi-Source Generation

### MultiSourceGenerator

```{eval-rst}
.. autoclass:: MultiSourceGenerator
   :members:
```

### SourceConfig

```{eval-rst}
.. autoclass:: SourceConfig
   :members:
```

### MultiSourceResult

```{eval-rst}
.. autoclass:: MultiSourceResult
   :members:
```

## Export Utilities

### DatasetExporter

```{eval-rst}
.. autoclass:: DatasetExporter
   :members:
```

### CSVVariationGenerator

```{eval-rst}
.. autoclass:: CSVVariationGenerator
   :members:
```

## Real Data Fitting

### RealDataFitter

```{eval-rst}
.. autoclass:: RealDataFitter
   :members:
```

### FittedParameters

```{eval-rst}
.. autoclass:: FittedParameters
   :members:
```

### SpectralProperties

```{eval-rst}
.. autoclass:: SpectralProperties
   :members:
```

## Validation Utilities

### validate_spectra

```{eval-rst}
.. autofunction:: validate_spectra
```

### validate_wavelengths

```{eval-rst}
.. autofunction:: validate_wavelengths
```

### validate_concentrations

```{eval-rst}
.. autofunction:: validate_concentrations
```

## Constants

### PREDEFINED_COMPONENTS

Dictionary of predefined spectral components:

- `"water"`: O-H overtones and combinations
- `"protein"`: N-H and C-H bonds
- `"lipid"`: C-H stretching modes
- `"starch"`: O-H and C-O combinations
- `"cellulose"`: Cellulose fingerprint
- `"chlorophyll"`: Plant pigments
- `"oil"`: Unsaturated C-H bonds
- `"nitrogen_compound"`: N-H combinations

### COMPLEXITY_PARAMS

Default parameters for each complexity level:

| Parameter | simple | realistic | complex |
|-----------|--------|-----------|---------|
| noise_scale | 0.001 | 0.005 | 0.02 |
| baseline_degree | 1 | 2 | 3 |
| scatter_intensity | 0.0 | 0.02 | 0.1 |

## See Also

- {doc}`/user_guide/data/synthetic_data` - User guide
- {doc}`nirs4all.api.generate` - High-level generate API
