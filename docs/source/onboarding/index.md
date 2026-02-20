# Understanding NIRS4ALL

Welcome to the NIRS4ALL deep dive! This section helps you build a mental model of the library's architecture, design philosophy, and internal workflows.

Think of NIRS4ALL as more than just a tool — it's a comprehensive system for spectroscopy data analysis that brings together:
- Smart data management (handling multi-source spectra with complex metadata)
- Flexible pipeline execution (supporting countless ML/DL frameworks)
- Reproducible experiment tracking (with DuckDB-backed persistence)
- Production-ready deployment (export and replay capabilities)

## Choose Your Learning Path

::::{grid} 3
:gutter: 3

:::{grid-item-card} 🔬 Researcher
:link: persona_paths
:link-type: doc

I want to understand pipelines and data concepts to analyze spectroscopy data effectively.

+++
Start with Mental Models → Data Workflow
:::

:::{grid-item-card} 💻 Developer
:link: persona_paths
:link-type: doc

I want to extend nirs4all with custom operators or understand the execution engine.

+++
Start with Controllers → Pipeline Workflow
:::

:::{grid-item-card} ⚙️ Operator Author
:link: persona_paths
:link-type: doc

I want to create custom preprocessing or modeling operators for the library.

+++
Start with Controllers → Extension Patterns
:::

::::

```{seealso}
**Related Examples:**
- {doc}`/examples/index` - Browse all 67 examples by topic and difficulty
- [U01: Hello World](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U01_hello_world.py) - See your first pipeline in action
- [U02: Basic Regression](https://github.com/GBeurier/nirs4all/blob/main/examples/user/01_getting_started/U02_basic_regression.py) - Complete workflow example
```

## Core Concepts

```{toctree}
:maxdepth: 2

mental_models
data_workflow
pipeline_workflow
workspace_intro
controllers_intro
persona_paths
```

## What Makes NIRS4ALL Different?

**Spectroscopy-first**: Unlike general ML libraries, NIRS4ALL understands spectroscopy-specific challenges like signal types, wavelength semantics, repeated measurements, and multi-source data.

**Controller-driven**: The execution engine uses a flexible controller pattern, allowing you to extend behavior without modifying core code.

**Workspace-backed**: Every experiment is persisted in a DuckDB store with content-addressed artifacts, giving you full traceability and reproducibility.

**Multi-framework**: Seamlessly mix sklearn, TensorFlow, PyTorch, JAX, and AutoGluon in a single pipeline.

## How to Use This Section

Each page in this section covers one major architectural concept:

1. **Mental Models** — The big picture and why the library is designed this way
2. **Data Workflow** — How SpectroDataset works and why it matters
3. **Pipeline Workflow** — How execution flows from user code to results
4. **Workspace Intro** — How experiments are persisted and replayed
5. **Controllers Intro** — The extension mechanism that powers everything
6. **Persona Paths** — Tailored learning paths based on your role

::::{tip}
If you're completely new, start with [Mental Models](mental_models.md) to build intuition, then dive into the specific areas that interest you.
::::
