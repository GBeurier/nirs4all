# Persona Learning Paths

Different users have different goals. This page provides **tailored learning paths** based on your role and objectives.

## Choose Your Persona

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔬 Research User
I want to analyze spectroscopy data and build predictive models.

+++
[Jump to path](#research-user-path)
:::

:::{grid-item-card} 💻 Pipeline Engineer
I want to build advanced pipelines with branching, stacking, and custom operators.

+++
[Jump to path](#pipeline-engineer-path)
:::

:::{grid-item-card} ⚙️ Platform/MLOps
I want to deploy models, manage workspaces, and automate workflows.

+++
[Jump to path](#platformmlops-path)
:::

:::{grid-item-card} 🛠️ Maintainer
I want to contribute to the library, fix bugs, and understand the internals.

+++
[Jump to path](#maintainer-path)
:::

::::

---

## Research User Path

**Goal**: Understand pipelines and data concepts to analyze spectroscopy data effectively.

### Learning Path

1. **Read core concepts** (2-3 hours)
   - [Mental Models](mental_models.md) — Understand the big picture
   - [Data Workflow](data_workflow.md) — Learn how SpectroDataset works
   - [Pipeline Workflow](pipeline_workflow.md) — Understand execution flow

2. **Run examples** (1-2 hours)
   - Start with `examples/user/01_getting_started/`
   - Try `examples/user/02_data_handling/`
   - Explore `examples/user/03_preprocessing/`

3. **Build your first pipeline** (1 hour)
   - Load your dataset
   - Start with one baseline pipeline (e.g., SNV + PLS)
   - Add controlled preprocessing alternatives via `_or_`
   - Use prediction ranking (`result.top()`)

4. **Explore advanced patterns** (ongoing)
   - Multi-source data (`examples/user/02_data_handling/U02_multi_source.py`)
   - Branching workflows (`examples/developer/01_branching/`)
   - Stacking (`examples/developer/01_branching/D04_stacking.py`)

### Key Resources

- **User Guide**: [Getting Started](../user_guide/index.md)
- **API Reference**: [nirs4all.run()](../api/run.md)
- **Examples**: `examples/user/`

### Common Questions

<details>
<summary><strong>How do I handle repeated measurements?</strong></summary>

Set `repetition_column` in your dataset config:

```python
dataset = nirs4all.load_dataset(
    "my_data.csv",
    repetition_column="sample_id",  # Group by this column
    aggregate_predictions="mean",   # Average predictions across repetitions
)
```

This ensures group-aware splitting and honest validation metrics.

</details>

<details>
<summary><strong>How do I compare preprocessing methods?</strong></summary>

Use `_or_` generators:

```python
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},  # Try all three
    PLSRegression(n_components=10),
]

result = nirs4all.run(pipeline, dataset)
print(result.top(5))  # Show top 5 by RMSE
```

</details>

<details>
<summary><strong>How do I export my best model?</strong></summary>

```python
result.export("best_model.n4a")
```

Then use it for prediction:

```python
predictions = nirs4all.predict("best_model.n4a", new_data)
```

</details>

---

## Pipeline Engineer Path

**Goal**: Build advanced pipelines with branching, stacking, custom operators, and performance optimization.

### Learning Path

1. **Read core architecture** (3-4 hours)
   - [Pipeline Workflow](pipeline_workflow.md) — Execution flow
   - [Controllers Intro](controllers_intro.md) — Extension mechanism
   - [Data Workflow](data_workflow.md) — Data semantics
   - [Workspace Intro](workspace_intro.md) — Persistence layer

2. **Study branching and merging** (2-3 hours)
   - Read: [Pipeline Workflow - Branching Section](pipeline_workflow.md#branching-workflow)
   - Examples: `examples/developer/01_branching/`
   - Try: Duplication vs separation branches

3. **Implement custom operator** (2-4 hours)
   - Study existing operators in `nirs4all/operators/`
   - Follow sklearn conventions (fit/transform/predict)
   - Create controller if needed (see [Controllers Intro](controllers_intro.md))
   - Test train/predict symmetry

4. **Profile and optimize** (ongoing)
   - Use cache configuration (`examples/developer/06_internals/D03_cache_performance.py`)
   - Monitor memory usage (`verbose=2`)
   - Optimize generator expansion strategies

### Key Resources

- **Developer Guide**: [Extending NIRS4ALL](../developer/extending.md)
- **API Reference**: [Controllers](../api/controllers.md)
- **Examples**: `examples/developer/`

### Common Questions

<details>
<summary><strong>How do I create a custom operator?</strong></summary>

Follow sklearn conventions:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransform(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param

    def fit(self, X, y=None):
        # Learn parameters from training data
        return self

    def transform(self, X):
        # Apply transformation
        return X * self.param
```

Then use in pipeline:

```python
pipeline = [MyTransform(param=2.0), PLSRegression(10)]
```

</details>

<details>
<summary><strong>How do I implement stacking?</strong></summary>

Use duplication branches + predictions merge:

```python
pipeline = [
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), RandomForestRegressor()],
    ]},
    {"merge": "predictions"},  # OOF predictions as features
    {"model": Ridge()},        # Meta-model
]
```

See `examples/developer/01_branching/D04_stacking.py` for details.

</details>

<details>
<summary><strong>How do I optimize memory usage?</strong></summary>

1. Enable cache with memory limits:

```python
from nirs4all.config.cache_config import CacheConfig

result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    cache=CacheConfig(
        step_cache_enabled=True,
        step_cache_max_mb=2048,  # 2 GB limit
        use_cow_snapshots=True,   # Copy-on-write snapshots
    ),
)
```

2. Use `verbose=2` to monitor per-step memory:

```python
result = nirs4all.run(pipeline, dataset, verbose=2)
```

</details>

---

## Platform/MLOps Path

**Goal**: Deploy models, manage workspaces, automate workflows, and ensure reproducibility.

### Learning Path

1. **Read operational concepts** (2-3 hours)
   - [Workspace Intro](workspace_intro.md) — Persistence and lifecycle
   - [Pipeline Workflow - Execution Modes](pipeline_workflow.md#execution-modes) — Train/predict/explain/refit
   - [Mental Models](mental_models.md) — Architecture overview

2. **Master workspace management** (2-3 hours)
   - Initialize workspace (`nirs4all workspace init`)
   - Query runs and predictions (`nirs4all workspace list-runs`)
   - Cleanup workflows (`nirs4all artifacts cleanup`)
   - Study: `nirs4all/cli/` for CLI commands

3. **Define deployment workflows** (2-4 hours)
   - Export pipelines (`result.export("model.n4a")`)
   - Replay predictions (`nirs4all.predict("model.n4a", data)`)
   - Automate validation gates (test export/replay consistency)
   - Version control workspace configurations

4. **Implement observability** (ongoing)
   - Structured logging (`verbose=1,2,3`)
   - Custom queries on workspace DB
   - Monitor run metrics and trends
   - Set up alerting for failed runs

### Key Resources

- **CLI Reference**: `nirs4all --help`
- **Workspace API**: [WorkspaceStore](../api/workspace.md)
- **Examples**: `examples/developer/06_internals/`

### Common Questions

<details>
<summary><strong>How do I set up a shared workspace?</strong></summary>

Set environment variable:

```bash
export NIRS4ALL_WORKSPACE="/shared/nirs4all-workspace"
```

Or specify in code:

```python
result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    workspace="/shared/nirs4all-workspace",
)
```

</details>

<details>
<summary><strong>How do I automate cleanup?</strong></summary>

Create a cleanup script:

```bash
#!/bin/bash
# cleanup.sh

# Delete runs older than 30 days
nirs4all workspace cleanup --older-than 30d

# Garbage collect orphaned artifacts
nirs4all artifacts cleanup

# Vacuum database
nirs4all workspace vacuum
```

Run monthly via cron.

</details>

<details>
<summary><strong>How do I query workspace data?</strong></summary>

Use DuckDB directly:

```python
import duckdb

conn = duckdb.connect("workspace/store.duckdb", read_only=True)

# Query top pipelines
result = conn.execute("""
    SELECT pipeline_name, AVG(rmse) as avg_rmse, COUNT(*) as n_runs
    FROM predictions
    WHERE dataset_name = 'my_dataset'
    GROUP BY pipeline_name
    ORDER BY avg_rmse
    LIMIT 10
""").fetchdf()

print(result)
```

</details>

---

## Maintainer Path

**Goal**: Contribute to the library, understand internals, fix bugs, and harden contracts.

### Learning Path

1. **Deep dive into architecture** (1-2 days)
   - [Pipeline Workflow](pipeline_workflow.md) — Execution engine
   - [Controllers Intro](controllers_intro.md) — Controller registry
   - [Data Workflow](data_workflow.md) — Dataset internals
   - [Workspace Intro](workspace_intro.md) — Persistence layer
   - [Mental Models](mental_models.md) — Design philosophy

2. **Map complexity hotspots** (1 day)
   - Study branching and merge controllers (`nirs4all/controllers/branch*.py`)
   - Review cache implementation (`nirs4all/cache/`)
   - Inspect workspace store (`nirs4all/workspace/`)
   - Read technical debt notes in library overview

3. **Harden contract boundaries** (ongoing)
   - Add type hints where missing
   - Write unit tests for edge cases
   - Document invariants in docstrings
   - Reduce ambiguous behavior through explicit error handling

4. **Contribute** (ongoing)
   - Pick an issue from GitHub
   - Fix bugs or add features
   - Write tests (`pytest tests/`)
   - Run examples (`cd examples && ./run.sh`)
   - Submit PR

### Key Resources

- **Source Code**: `nirs4all/` directory
- **Tests**: `tests/unit/` and `tests/integration/`
- **Examples**: Run with `./run.sh` in `examples/`
- **CI**: GitHub Actions (`.github/workflows/`)

### First Week Plan

| Day | Activity |
| --- | --- |
| 1 | Run a small training example from `examples/` |
| 2 | Trace execution through runner, orchestrator, and executor |
| 3 | Study one controller family in detail (e.g., model controllers) |
| 4 | Inspect workspace tables and artifacts from your run |
| 5 | Implement a small non-critical extension or fix |

### Common Questions

<details>
<summary><strong>How do I run tests?</strong></summary>

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=nirs4all
```

</details>

<details>
<summary><strong>How do I add a new controller?</strong></summary>

1. Create controller in `nirs4all/controllers/`
2. Inherit from `OperatorController`
3. Implement `matches()`, `execute()`, etc.
4. Decorate with `@register_controller`
5. Import in `nirs4all/controllers/__init__.py`
6. Write tests in `tests/unit/controllers/`

See [Controllers Intro](controllers_intro.md) for details.

</details>

<details>
<summary><strong>How do I debug execution flow?</strong></summary>

Use verbose logging:

```python
result = nirs4all.run(pipeline, dataset, verbose=3)
```

Or add breakpoints in:
- `nirs4all/pipeline/executor.py` — main execution loop
- `nirs4all/pipeline/step_runner.py` — step routing
- `nirs4all/controllers/` — specific controller

</details>

---

## New Contributor First Week Plan

A structured plan for your first week contributing to NIRS4ALL:

### Day 1: Get Oriented

**Morning:**
- [ ] Clone repository
- [ ] Set up development environment (Python 3.11+, install dependencies)
- [ ] Run installation verification: `nirs4all --test-install`

**Afternoon:**
- [ ] Run a simple example: `examples/user/01_getting_started/U01_minimal.py`
- [ ] Read [Mental Models](mental_models.md)
- [ ] Browse `nirs4all/api/` — module-level entry points

### Day 2: Trace Execution

**Morning:**
- [ ] Add print statements or breakpoints to:
  - `nirs4all/api/run.py` → `run()` function
  - `nirs4all/pipeline/runner.py` → `PipelineRunner.run()`
  - `nirs4all/pipeline/orchestrator.py` → `execute()`

**Afternoon:**
- [ ] Trace a simple pipeline through:
  - `PipelineExecutor` → step loop
  - `StepRunner` → parse/route/execute
  - `ControllerRouter` → priority-based selection

**Output**: Diagram of execution flow in your notebook

### Day 3: Deep Dive into Controllers

**Morning:**
- [ ] Read [Controllers Intro](controllers_intro.md)
- [ ] Study `nirs4all/controllers/controller.py` — base class
- [ ] Pick one family (e.g., model controllers)

**Afternoon:**
- [ ] Study `SklearnModelController` in detail
- [ ] Trace how it:
  - Matches sklearn estimators
  - Fits models per fold
  - Emits artifacts
  - Loads artifacts in predict mode

**Output**: Document controller lifecycle for one family

### Day 4: Inspect Workspace

**Morning:**
- [ ] Run example with workspace: `examples/user/01_getting_started/U02_workspace.py`
- [ ] Inspect `workspace/store.duckdb` with DuckDB CLI:
  ```bash
  duckdb workspace/store.duckdb
  .tables
  SELECT * FROM runs;
  SELECT * FROM predictions LIMIT 10;
  ```

**Afternoon:**
- [ ] Read [Workspace Intro](workspace_intro.md)
- [ ] Explore artifact storage: `workspace/artifacts/`
- [ ] Query predictions with custom SQL

**Output**: Document workspace schema and lifecycle

### Day 5: Implement Small Fix

**Morning:**
- [ ] Pick a "good first issue" from GitHub
- [ ] Or fix a small bug you noticed
- [ ] Write tests in `tests/unit/`

**Afternoon:**
- [ ] Run tests: `pytest tests/`
- [ ] Run examples: `cd examples && ./run.sh`
- [ ] Submit PR (or keep in draft for feedback)

**Output**: Your first contribution!

---

## Next Steps

After completing your persona's learning path:

- **Explore examples**: `examples/` directory has 50+ examples
- **Read full docs**: [User Guide](../user_guide/index.md) and [Developer Guide](../developer/index.md)
- **Join community**: GitHub Discussions for questions and feedback
- **Contribute**: Pick an issue and submit a PR!

::::{tip}
Learning is iterative. Don't try to understand everything at once. Start with your role's path, then explore adjacent areas as needed.
::::

```{seealso}
**Related Examples:**
- [Examples Index](../../examples/index.md) - Browse all 67 examples organized by topic and difficulty
- [U01-U04: Getting Started](https://github.com/GBeurier/nirs4all/tree/main/examples/user/01_getting_started) - First examples for all personas
- [Developer Examples](https://github.com/GBeurier/nirs4all/tree/main/examples/developer) - Advanced examples for extending nirs4all
```
