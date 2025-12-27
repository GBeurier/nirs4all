# Deployment

This section covers exporting trained models and using them in production.

```{toctree}
:maxdepth: 2

export_bundles
prediction_model_reuse
retrain_transfer
```

## Overview

Once you've trained a successful pipeline, NIRS4ALL makes it easy to deploy to production or share with collaborators.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📦 Export Bundles
:link: export_bundles
:link-type: doc

Package trained pipelines as standalone `.n4a` bundles.

+++
{bdg-primary}`Production`
:::

:::{grid-item-card} 🔄 Model Reuse
:link: prediction_model_reuse
:link-type: doc

Load and use exported models for new predictions.

+++
{bdg-success}`Inference`
:::

:::{grid-item-card} 🔧 Transfer Learning
:link: retrain_transfer
:link-type: doc

Adapt models to new instruments or sample types.

+++
{bdg-warning}`Adaptation`
:::

::::

## Deployment Workflow

```{mermaid}
graph LR
    A[Train Pipeline] --> B[Evaluate]
    B --> C{Good?}
    C -->|Yes| D[Export .n4a]
    C -->|No| A
    D --> E[Production Use]
    D --> F[Share/Collaborate]
```

## Quick Example

### Export a Model

```python
import nirs4all

# Train and get results
result = nirs4all.run(pipeline, dataset="data/")

# Export the best model
result.export("exports/best_model.n4a")
```

### Use an Exported Model

```python
import nirs4all

# Load and predict
predictions = nirs4all.predict(
    bundle="exports/best_model.n4a",
    data="new_samples/"
)
```

## Bundle Contents

A `.n4a` bundle contains:
- Trained model weights
- Preprocessing transformers (fitted)
- Pipeline configuration
- Metadata (training date, metrics, etc.)

## See Also

- {doc}`/reference/cli` - CLI commands for bundle management
- {doc}`/reference/workspace` - Workspace and artifact structure
