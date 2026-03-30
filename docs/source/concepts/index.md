# Concepts

This section explains the core ideas behind nirs4all. Each page covers one
concept at a high level -- what it is, why it matters, and how the pieces fit
together. If you prefer to learn by reading code, jump to
{doc}`/examples/index`. If you need exact parameter lists, see
{doc}`/reference/index`.

## Suggested Reading Order

Start with **Pipelines** and **Datasets** -- they are the two objects you
pass to every `nirs4all.run()` call. Then read **Cross-Validation** to
understand how performance is measured. The remaining pages build on those
foundations.

```
Pipelines --> Datasets --> Cross-Validation
    |                           |
    +---> Branching & Merging   |
    +---> Generators            |
    +---> Augmentation          |
                                |
              Predictions & Deployment
```

## Pages

::::{grid} 2
:gutter: 3

:::{grid-item-card} Pipelines
:link: pipelines
:link-type: doc

What a pipeline is, the four step types, and how steps are executed.
:::

:::{grid-item-card} Datasets
:link: datasets
:link-type: doc

SpectroDataset, partitions, multi-source data, signal types, and repetitions.
:::

:::{grid-item-card} Cross-Validation
:link: cross_validation
:link-type: doc

Fold-based evaluation, out-of-fold predictions, scoring, and refit.
:::

:::{grid-item-card} Branching and Merging
:link: branching_and_merging
:link-type: doc

Parallel sub-pipelines, separation branches, merge strategies, and stacking.
:::

:::{grid-item-card} Generators
:link: generators
:link-type: doc

Express a search space in one pipeline and let nirs4all expand it into many
variants.
:::

:::{grid-item-card} Augmentation
:link: augmentation
:link-type: doc

Expand small datasets with synthetic training samples and feature views.
:::

:::{grid-item-card} Predictions and Deployment
:link: predictions_and_deployment
:link-type: doc

RunResult, export, predict, explain, retrain, and the Session API.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

pipelines
datasets
cross_validation
branching_and_merging
generators
augmentation
predictions_and_deployment
```
