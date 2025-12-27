# Reference

Complete reference documentation for NIRS4ALL APIs, syntax, and operators.

```{toctree}
:maxdepth: 2

pipeline_syntax
operator_catalog
cli
```

## Overview

This section provides detailed reference documentation:

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📄 Pipeline Syntax
:link: pipeline_syntax
:link-type: doc

Complete guide to writing pipelines including all step types, generator syntax, and options.

+++
{bdg-primary}`Comprehensive`
:::

:::{grid-item-card} 🧰 Operator Catalog
:link: operator_catalog
:link-type: doc

Complete listing of all available operators: preprocessors, splitters, models, and augmentations.

+++
{bdg-success}`All Operators`
:::

:::{grid-item-card} 💻 CLI Reference
:link: cli
:link-type: doc

Command-line interface documentation for all `nirs4all` workspace commands.

+++
{bdg-info}`Commands`
:::

:::{grid-item-card} 🐍 API Reference
:link: /api/module_api
:link-type: doc

Module-level API: `nirs4all.run()`, `nirs4all.predict()`, and related functions.

+++
{bdg-warning}`Python API`
:::

::::

## API Documentation

Detailed API documentation is also available:

- {doc}`/api/module_api` - Module-level API (`nirs4all.run()`, `nirs4all.predict()`, etc.)
- {doc}`/api/sklearn_integration` - sklearn-compatible `NIRSPipeline` wrapper

## See Also

- {doc}`/user_guide/index` - Step-by-step how-to guides
- {doc}`/developer/architecture` - Architecture overview for developers
- {doc}`/examples/index` - Working examples organized by topic
