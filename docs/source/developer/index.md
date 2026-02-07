# Developer Guide

Documentation for developers who want to understand NIRS4ALL internals or contribute to the project.

```{toctree}
:maxdepth: 2

architecture
pipeline_architecture
controllers
caching
artifacts
artifacts_internals
metadata
outputs_vs_artifacts
synthetic
testing
```

## Overview

This section is for developers who want to:

- **Understand** the internal architecture of NIRS4ALL
- **Extend** the library with custom controllers and operators
- **Contribute** to the project

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🏗️ Architecture
:link: architecture
:link-type: doc

High-level overview of the pipeline execution engine, data flow, and component interactions.

+++
{bdg-primary}`System Design`
:::

:::{grid-item-card} 🔄 Pipeline Architecture
:link: pipeline_architecture
:link-type: doc

Detailed pipeline execution flow and internal mechanics.

+++
{bdg-info}`Internals`
:::

:::{grid-item-card} 🎮 Controllers
:link: controllers
:link-type: doc

The controller registry system that dispatches pipeline steps to appropriate handlers.

+++
{bdg-success}`Extension Point`
:::

:::{grid-item-card} 📦 Artifacts & Storage
:link: artifacts
:link-type: doc

User guide for the artifact storage system.

+++
{bdg-warning}`Storage`
:::

:::{grid-item-card} Caching & Memory
:link: caching
:link-type: doc

Block-based storage, copy-on-write branches, step caching, and memory observability.

+++
{bdg-info}`Performance`
:::

::::

## Additional Developer Topics

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔧 Artifacts Internals
:link: artifacts_internals
:link-type: doc

Internal implementation details and extension points for the artifacts system.

+++
{bdg-secondary}`Deep Dive`
:::

:::{grid-item-card} 📝 Metadata
:link: metadata
:link-type: doc

Pipeline and dataset metadata handling.

+++
{bdg-secondary}`Data`
:::

:::{grid-item-card} 📤 Outputs vs Artifacts
:link: outputs_vs_artifacts
:link-type: doc

Understanding the difference between outputs and artifacts.

+++
{bdg-secondary}`Concepts`
:::

:::{grid-item-card} 🧪 Synthetic Data
:link: synthetic
:link-type: doc

Synthetic data generator internals and extension.

+++
{bdg-info}`New in 0.6`
:::

:::{grid-item-card} 🧪 Testing Guide
:link: testing
:link-type: doc

Running tests, markers, fixtures, and writing new tests.

+++
{bdg-primary}`Quality`
:::

::::

## Quick Start: Custom Controller

Here's a minimal example of creating a custom controller:

```python
from nirs4all.controllers import register_controller, OperatorController

@register_controller
class MyController(OperatorController):
    priority = 50  # Lower = higher priority

    @classmethod
    def matches(cls, step, operator, keyword) -> bool:
        return keyword == "my_custom_keyword"

    @classmethod
    def use_multi_source(cls) -> bool:
        return False

    def execute(self, step_info, dataset, context, runtime_context, **kwargs):
        # Your implementation here
        return context, output
```

See {doc}`controllers` for the complete guide.

## See Also

- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
- {doc}`/examples/index` - Working examples including developer examples
- [GitHub Repository](https://github.com/GBeurier/nirs4all) - Source code and issues
