# Augmentation

NIRS datasets are often small. Collecting and measuring physical samples is
expensive and time-consuming. Augmentation helps by generating synthetic
variations of your training data, giving the model more examples to learn from
without collecting new samples.

---

## Why Augment?

A model trained on 50 spectra may overfit to the specific noise patterns of
those 50 measurements. Augmentation introduces controlled variability --
small spectral shifts, additive noise, baseline drift -- so the model learns
to be robust to these effects rather than memorising them.

:::{note}
Augmentation is applied **only during training**. Validation and test samples
are never augmented, so your evaluation scores remain honest.
:::

---

## When Augmentation Runs

During cross-validation, augmentation happens inside each fold:

```
Fold k:
  1. Select train subset
  2. Apply augmentation -> expanded train subset
  3. Fit transforms and model on expanded train
  4. Predict on val and test (no augmentation)
```

Augmented samples are tagged with an `origin_id` that links them back to their
base sample. This tagging ensures that:

- Augmented samples never leak into the validation or test set.
- Splitters ignore augmented samples when assigning folds.
- Predictions can be traced back to the original measurement.

---

## Sample Augmentation

Sample augmentation generates new training rows by applying random
perturbations to existing spectra.

```python
from nirs4all.operators.augmentation import GaussianAdditiveNoise

pipeline = [
    {"sample_augmentation": GaussianAdditiveNoise(sigma=0.01)},
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(10)},
]
```

Each base sample produces one or more augmented copies. The augmented copies
share the same target value (y) as the original.

### Augmentation families

nirs4all includes a library of NIRS-specific augmentation operators grouped
by the type of variation they simulate:

| Family         | Examples                                       | What it simulates            |
|----------------|------------------------------------------------|------------------------------|
| **Noise**      | GaussianAdditiveNoise, MultiplicativeNoise, SpikeNoise | Instrument noise          |
| **Baseline**   | LinearBaselineDrift, PolynomialBaselineDrift    | Baseline shifts              |
| **Wavelength** | WavelengthShift, WavelengthStretch, LocalWavelengthWarp | Wavelength calibration drift |
| **Spectral**   | SmoothMagnitudeWarp, BandPerturbation, BandMasking | Spectral shape variation   |
| **Mixup**      | MixupAugmenter, LocalMixupAugmenter            | Weighted sample blending     |
| **Physical**   | PathLengthAugmenter, BatchEffectAugmenter       | Physical measurement effects |
| **Scatter**    | ScatterSimulationMSC                            | Scatter variation            |

See {doc}`/reference/operator_catalog` for the full list of augmentation
operators and their parameters.

### Standard vs. balanced mode

By default, augmentation produces a fixed number of copies per sample
(**standard mode**). For classification tasks with imbalanced classes, you can
switch to **balanced mode**, which generates more copies for under-represented
classes:

```python
{"sample_augmentation": {
    "transformers": [GaussianAdditiveNoise(sigma=0.01)],
    "balance": "y",
    "target_size": 100,
}}
```

This is especially useful when one class has many fewer samples than another.

---

## Feature Augmentation

Feature augmentation does not create new rows. Instead, it creates new
**feature columns** by applying different transforms to the same spectra and
concatenating the results.

```python
{"feature_augmentation": [SNV(), SavitzkyGolay()], "action": "extend"}
```

The `action` parameter controls how the new features are combined with
existing ones:

| Action      | Effect                                                 |
|-------------|--------------------------------------------------------|
| `"extend"`  | Add new processing chains alongside existing ones      |
| `"add"`     | Add derived features to the current processing chain   |
| `"replace"` | Replace the current processing with the new one        |

Feature augmentation is useful when you want a model to see both raw spectra
and derived features (e.g., SNV-corrected, first-derivative) simultaneously,
without branching.

---

## concat_transform

A simpler alternative to feature augmentation for a common case: apply
several transforms independently and concatenate all results into one wide
feature matrix.

```python
{"concat_transform": [SNV(), SavitzkyGolay(deriv=1), SavitzkyGolay(deriv=2)]}
```

The example above produces a feature matrix that is three times wider than
the original -- one copy per transform. This gives the model access to
multiple representations of the same spectra in a single flat input.

---

## Choosing an Approach

| Technique             | New rows? | New features? | Best for                           |
|-----------------------|-----------|---------------|------------------------------------|
| Sample augmentation   | Yes       | No            | Small datasets, noise robustness   |
| Feature augmentation  | No        | Yes           | Multiple spectral views, deep learning |
| concat_transform      | No        | Yes           | Quick multi-representation input   |

You can combine all three in the same pipeline. Sample augmentation runs
first (creating new rows), then feature augmentation or concat_transform adds
feature columns.

---

## Practical Tips

:::{tip}
Start with mild augmentation (low noise, small shifts) and increase gradually.
Aggressive augmentation can distort spectral features and hurt performance.
:::

:::{tip}
Monitor the gap between RMSEC (calibration error) and RMSECV
(cross-validation error). If RMSEC is much lower, the model is overfitting
and augmentation may help. If both are similar, augmentation is less likely to
improve results.
:::

---

## Next Steps

- {doc}`pipelines` -- understand pipeline structure and where augmentation
  steps fit.
- {doc}`cross_validation` -- learn how augmented samples interact with folds.
- {doc}`/user_guide/augmentation/sample_augmentation_guide` -- detailed sample
  augmentation guide.
- {doc}`/reference/operator_catalog` -- full list of augmentation operators.
