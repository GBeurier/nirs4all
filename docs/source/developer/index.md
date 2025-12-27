# Developer Guide

Documentation for developers who want to understand NIRS4ALL internals or contribute to the project.

```{toctree}
:maxdepth: 2

architecture
controllers
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

:::{grid-item-card} 🎮 Controllers
:link: controllers
:link-type: doc

The controller registry system that dispatches pipeline steps to appropriate handlers.

+++
{bdg-success}`Extension Point`
:::

::::

## Coming Soon

The following developer guides are planned:

- **Custom Extensions** - Step-by-step guide to writing your own controllers and operators
- **Artifacts** - The artifact storage system for saving and loading pipeline state
- **Deep Learning** - Integration patterns for TensorFlow, PyTorch, Keras, and JAX models
- **Contributing** - Guidelines for contributing code, documentation, and bug reports

:::{tip}
For now, see the examples in `examples/developer/` for advanced usage patterns.
:::

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
