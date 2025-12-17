# Logging System User Guide

This guide explains how to use the nirs4all logging system for structured, configurable output.

## Overview

The nirs4all logging system provides:

- **Human-readable console output** optimized for researchers
- **Machine-parseable file logging** for automation and analysis
- **Progress bars** with TTY-aware display
- **Context tracking** for runs, branches, and sources
- **ASCII-safe output** for HPC/cluster environments

## Quick Start

```python
from nirs4all.pipeline import PipelineRunner

# Basic usage - logging is configured automatically
runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(pipeline, dataset)

# With logging options
runner = PipelineRunner(
    verbose=2,              # Detailed output
    log_file=True,          # Write to workspace/logs/
    log_format="pretty",    # Human-readable format
    use_unicode=True,       # Use Unicode symbols
    use_colors=True,        # ANSI colors
)
```

## Verbosity Levels

| Level | Name | Use Case |
|-------|------|----------|
| 0 | Quiet | Silent operation, errors only. Best for production/notebooks |
| 1 | Standard | Key milestones and results. **Recommended for research** |
| 2 | Debug | Detailed operation, troubleshooting |
| 3 | Trace | Full trace with per-fold/per-step details |

### What you see at each level

**verbose=0 (Quiet)**
```
Only warnings and errors - no progress information.
```

**verbose=1 (Standard)**
```
> Loading data...
  [OK] Loaded dataset: 3,482 samples x 2,150 features

> Evaluating pipelines...
  * Progress: 21/42 (50%) -- best RMSE: 0.389
  [OK] Evaluation complete

> Training best model...
  [OK] Model trained: CV_RMSE=0.381
```

**verbose=2 (Debug)**
```
Everything from verbose=1, plus:
- Configuration details (seeds, versions)
- Pipeline generation/pruning statistics
- Cache hits/misses
- Per-pipeline evaluation summaries
- Memory/GPU usage warnings
```

## Configuration Options

### PipelineRunner Parameters

```python
runner = PipelineRunner(
    # Verbosity
    verbose=1,              # 0-3, controls log level

    # File logging
    log_file=True,          # Write logs to files
    log_format="pretty",    # "pretty", "minimal", or "json"
    json_output=False,      # Also write JSON Lines file

    # Display settings
    use_unicode=True,       # Unicode symbols (False for ASCII)
    use_colors=True,        # ANSI colors (auto-detect TTY)
    show_progress_bar=True, # Show progress bars
)
```

### Environment Variables

Override settings via environment variables:

```bash
# Override log level
export NIRS4ALL_LOG_LEVEL=DEBUG

# Force ASCII-only output (for clusters)
export NIRS4ALL_ASCII_ONLY=1

# Disable colors
export NIRS4ALL_NO_COLOR=1
```

## Progress Bars

The logging system includes TTY-aware progress bars that automatically adapt to terminal capabilities.

### Basic Usage

```python
from nirs4all.core.logging import ProgressBar, EvaluationProgress

# Simple progress bar
with ProgressBar(total=100, description="Processing") as pbar:
    for i in range(100):
        # do work
        pbar.update(1)

# With iterator
for item in ProgressBar.wrap(items, description="Processing"):
    process(item)
```

### ML-Specific Evaluation Progress

```python
from nirs4all.core.logging import EvaluationProgress

# Track pipeline evaluation with best score
with EvaluationProgress(
    total_pipelines=42,
    metric_name="RMSE",
    higher_is_better=False
) as progress:
    for pipeline in pipelines:
        score = evaluate(pipeline)
        is_new_best = progress.update(score=score, pipeline_name=pipeline.name)
        if is_new_best:
            print(f"New best: {score}")
```

### Multi-Level Progress

For nested operations (datasets → pipelines → folds):

```python
from nirs4all.core.logging import MultiLevelProgress

progress = MultiLevelProgress(run_total=5, run_description="Datasets")

with progress.run_level() as run_pbar:
    for dataset in datasets:
        with progress.pipeline_level(total=10) as pipe_pbar:
            for pipeline in pipelines:
                with progress.fold_level(total=5) as fold_pbar:
                    for fold in folds:
                        # evaluate
                        fold_pbar.update(1)
                pipe_pbar.update(1)
        run_pbar.update(1)
```

### Spinner for Unknown Duration

```python
from nirs4all.core.logging import spinner

with spinner("Loading large dataset") as s:
    data = load_dataset()
    s.update("Parsing...")
    parsed = parse(data)
```

## File Logging

### Log File Location

When `log_file=True`, logs are written to:
```
{workspace}/logs/{run_id}.log      # Human-readable
{workspace}/logs/{run_id}.jsonl    # JSON Lines (if json_output=True)
```

### Log Rotation

Logs are automatically rotated based on:

- **Count**: Keep last N runs (default: 100)
- **Age**: Remove logs older than N days (default: 30)
- **Size**: Rotate when file exceeds N bytes (optional)

Old logs are compressed with gzip to save space.

```python
from nirs4all.core.logging import configure_logging

configure_logging(
    log_file=True,
    log_dir="./workspace/logs",
    max_log_runs=50,        # Keep last 50 runs
    max_log_age_days=14,    # Remove after 14 days
    max_log_bytes=10_000_000,  # Rotate at 10MB
    compress_logs=True,     # Gzip old logs
)
```

### JSON Lines Format

For integration with log aggregation systems (ELK, Loki, etc.):

```python
runner = PipelineRunner(
    log_file=True,
    json_output=True  # Write .jsonl file
)
```

JSON log entries look like:
```json
{"ts": "2025-12-16T19:12:03.041+01:00", "level": "INFO", "run_id": "R-20251216-191203", "message": "Loading data...", "phase": "data"}
{"ts": "2025-12-16T19:12:05.882+01:00", "level": "INFO", "run_id": "R-20251216-191203", "message": "Data loaded", "samples": 3482, "features": 2150}
```

## Context Tracking

### Run Context

Track entire runs for reproducibility:

```python
from nirs4all.core.logging import LogContext, get_logger

logger = get_logger(__name__)

with LogContext(run_id="experiment-001", project="protein-analysis"):
    logger.info("Starting analysis")
    # All logs include run_id
```

### Branch Context

Track pipeline branches:

```python
with LogContext.branch("snv", index=0, total=4):
    logger.info("Processing SNV preprocessing")
    # Output: [branch:snv] Processing SNV preprocessing
```

### Source Context

Track multi-source pipelines:

```python
with LogContext.source("NIR", index=0, total=3):
    logger.info("Processing NIR spectra")
    # Output: [source:0/NIR] Processing NIR spectra
```

## Module-Level Logging

For library code, use module-level loggers:

```python
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

def my_function():
    logger.info("Starting processing")
    logger.debug("Detailed info for debugging")
    logger.warning("Something unexpected happened")
    logger.success("Operation completed")  # [OK] prefix
```

### Available Methods

| Method | Level | Symbol | Use |
|--------|-------|--------|-----|
| `logger.info()` | INFO | (none) | General information |
| `logger.debug()` | DEBUG | (none) | Detailed debugging |
| `logger.warning()` | WARNING | `[!]` | Non-fatal issues |
| `logger.error()` | ERROR | `[X]` | Fatal errors |
| `logger.success()` | INFO | `[OK]` | Successful completion |
| `logger.starting()` | INFO | `>` | Starting an operation |
| `logger.progress()` | INFO | `*` | Progress updates (throttled) |

## HPC/Cluster Environments

For HPC systems without Unicode support:

```python
runner = PipelineRunner(
    use_unicode=False,  # ASCII-only symbols
    use_colors=False,   # No ANSI escape codes
)
```

Or set environment variables:
```bash
export NIRS4ALL_ASCII_ONLY=1
export NIRS4ALL_NO_COLOR=1
```

## Example Output

### Standard Run (verbose=1)

```
================================================================================
  nirs4all run: wheat_protein_analysis
  Started: 2025-12-16 19:12:03
================================================================================

> Loading data...
  [OK] Loaded wheat_nir: 3,482 samples x 2,150 features

> Building cross-validation splits...
  [OK] 5-fold GroupKFold ready

> Evaluating pipelines...
  * Progress: 21/42 (50%) -- best RMSE: 0.389
  [OK] Evaluation complete

> Training best model...
  [OK] Model trained: CV_RMSE=0.381

================================================================================
  [OK] Run completed in 2m 5.9s

  Best pipeline: SavGol(w=11) -> PCA(n=150) -> TabPFN
  Metrics: RMSE=0.381  R2=0.82
================================================================================
```

### With Branching (verbose=2)

```
> Entering branch block (4 branches)...
  |
  |-- [branch:snv] SNV preprocessing
  |   * fold 1/5: RMSE=0.412
  |   * fold 2/5: RMSE=0.398
  |   [OK] CV_RMSE=0.405
  |
  |-- [branch:msc] MSC preprocessing
  |   [OK] CV_RMSE=0.392
  |
  |-- [branch:savgol] Savitzky-Golay
  |   [OK] CV_RMSE=0.381  <- best
  |

> Branch comparison:
  +------------+----------+-------+
  | Branch     | CV_RMSE  | Rank  |
  +------------+----------+-------+
  | savgol     | 0.381    | 1     |
  | msc        | 0.392    | 2     |
  | snv        | 0.405    | 3     |
  +------------+----------+-------+
```

## Troubleshooting

### Logs not appearing

Check verbosity level:
```python
runner = PipelineRunner(verbose=1)  # INFO level
```

### Progress bars not working

Progress bars require a TTY. In non-interactive environments (notebooks, CI), they fall back to line-based updates.

### Unicode errors on cluster

```python
runner = PipelineRunner(use_unicode=False)
```

### Finding log files

```python
from nirs4all.core.logging import get_config

config = get_config()
if config._file_handler:
    print(f"Log file: {config._file_handler.get_log_file_path()}")
```
