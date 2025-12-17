# NIRS4ALL Logging Specification

**Version:** 1.1
**Status:** Draft
**Date:** December 2024
**Authors:** nirs4all team

---

## 1. Executive Summary

This specification defines a unified logging system for nirs4all, **replacing all `print()` statements** with a structured, configurable logging infrastructure. The design prioritizes **human readability** for scientists and ML practitioners while maintaining machine-parseable output for automation pipelines.

### Design Philosophy

1. **Science-first readability**: Logs should be instantly understandable by researchers, not just DevOps engineers
2. **Progressive disclosure**: Show what matters at each verbosity level, no information overload
3. **Minimal ceremony**: Clean output without excessive timestamps or structured noise in interactive use
4. **Reproducibility support**: Enable tracking of runs and configurations for scientific reproducibility
5. **Dual output**: Human-friendly console + optional structured file logging for tooling
6. **Cluster-safe**: Full support for ASCII-only output on HPC/Unix environments without Unicode support

---

## 2. Rationale & Context

### Current State

The nirs4all library currently uses:
- Direct `print()` calls with emoji prefixes
- A `verbose` parameter (0-2) passed through pipeline components
- Scattered logging in some modules (`logging.getLogger(__name__)`)
- Environment variable `DISABLE_EMOJI` for terminal compatibility (incomplete)
- The `nirs4all/utils/emoji.py` module with emoji constants

### Problems to Solve

| Issue | Impact |
|-------|--------|
| Inconsistent output format | Hard to parse, grep, or analyze logs |
| No log levels | Cannot filter noise vs. critical information |
| No file logging | Runs are not recorded for later analysis |
| No run identification | Cannot correlate outputs to specific executions |
| Mixed print/logging | Unpredictable behavior when integrating with other tools |
| Progress overwhelm | Long-running jobs flood terminal with updates |
| Unicode/emoji failures | Cluster and remote environments crash or show garbage |
| Complex pipelines poorly logged | Branching, multisource, stacking hard to follow |

### Goals

1. **Replace all library `print()` with logging calls**
2. **Remove `nirs4all/utils/emoji.py`** and all direct emoji usage
3. **Provide human-readable console output by default**
4. **Enable structured file logging for automation/analysis**
5. **Support scientific workflows** (reproducibility, traceability)
6. **Handle complex pipeline structures** (branching, multisource, stacking)
7. **Full ASCII fallback** for HPC/cluster environments

---

## 3. Code to Remove/Replace

### Files to Delete

- `nirs4all/utils/emoji.py` - Replace with logging symbols in formatter

### Code to Remove from `nirs4all/__init__.py`

Remove the entire emoji stripping mechanism:

```python
# DELETE THIS BLOCK:
os.environ['DISABLE_EMOJIS'] = '1'
if os.environ.get('DISABLE_EMOJIS') == '1':
    import re
    original_print = __builtins__['print']
    def strip_emojis(text): ...
    def emoji_free_print(*args, **kwargs): ...
    __builtins__['print'] = emoji_free_print
```

### Pattern Replacements

| Old Pattern | New Pattern |
|------------|-------------|
| `print(f"{ROCKET} Starting...")` | `logger.info("Starting...")` |
| `print(f"{WARNING} Issue detected")` | `logger.warning("Issue detected")` |
| `print(f"{CHECK} Done")` | `logger.info("Done", status="success")` |
| `if verbose > 0: print(...)` | `logger.info(...)` (level handles filtering) |
| `if verbose > 1: print(...)` | `logger.debug(...)` |

---

## 4. Verbosity Levels & Use Cases

### Level Mapping

| Verbose | Logging Level | Use Case | Target Audience |
|---------|---------------|----------|-----------------|
| 0 | WARNING+ | Silent operation, errors only | Production / notebooks |
| 1 | INFO | Standard operation, key milestones | Researchers |
| 2 | DEBUG | Detailed operation, troubleshooting | Developers / debugging |
| 3 | TRACE* | Full trace, per-fold/per-step details | Deep debugging |

*TRACE is implemented as DEBUG with additional filter, not a separate Python level.

### What to Log at Each Level

#### verbose=0 (Quiet / Production)
```
Only warnings, errors, and critical failures.
No progress information.
```

#### verbose=1 (Standard / Researcher)
```
- Run start/end with summary metrics
- Major phase transitions (data loading, training, evaluation)
- Final results and best model information
- Branch/source context when relevant
- Warnings (domain shift, missing data, etc.)
- Artifacts saved (paths)
```

#### verbose=2 (Debug / Developer)
```
Everything from verbose=1, plus:
- Configuration details (seeds, versions, hashes)
- Pipeline generation/pruning statistics
- Cache hits/misses
- Per-pipeline evaluation summaries
- Per-branch and per-source details
- Memory/GPU usage warnings
- Detailed timing information
```

#### verbose=3 (Trace / Deep Debug)
```
Everything from verbose=2, plus:
- Per-fold CV results
- Individual step execution details
- Full preprocessing chain information
- Artifact registry operations
- Search space exploration details
- Branch path tracing
- Stacking layer details
```

---

## 5. Console Output Format

### Design Principles

1. **Clean and scannable**: Minimal visual noise, easy to skim
2. **Aligned structure**: Consistent column alignment for visual parsing
3. **Progressive indentation**: Hierarchy through indentation, not verbose prefixes
4. **Smart timestamps**: Time elapsed, not wall-clock (except for file logs)
5. **ASCII-safe symbols**: All decorators have ASCII fallbacks
6. **Pipeline structure awareness**: Branch/source context embedded naturally

### Symbol System (ASCII-Safe)

All symbols have two modes controlled by `use_unicode` setting:

| Purpose | Unicode | ASCII | Description |
|---------|---------|-------|-------------|
| Starting | `>` | `>` | Beginning of a phase |
| Success | `[OK]` | `[OK]` | Successful completion |
| Progress | `*` | `*` | In progress |
| Skipped | `-` | `-` | Skipped (cached) |
| Warning | `[!]` | `[!]` | Non-fatal issue |
| Error | `[X]` | `[X]` | Fatal error |
| Branch | `\|` | `\|` | Branch indicator |
| Indent | `  ` | `  ` | Hierarchy level |
| Arrow | `->` | `->` | Flow/result |

### Format Specification

```
[ELAPSED] [STATUS] MESSAGE                                    [CONTEXT]
```

Where:
- **ELAPSED**: Time since run start (optional, configurable)
- **STATUS**: Symbol indicating event type
- **MESSAGE**: Human-readable action/status
- **CONTEXT**: Optional key=value pairs for important details

### Example Output: verbose=1 (ASCII mode)

```
================================================================================
  nirs4all run: maize_protein_analysis
  Started: 2025-12-16 19:12:03
================================================================================

> Loading data...
  [OK] Loaded maize_nir_v7: 3,482 samples x 2,150 features

> Building cross-validation splits...
  [OK] 5-fold GroupKFold ready (groups=genotype, stratify=site)

> Generating pipeline candidates...
  [OK] 42 pipelines generated (86 pruned: duplicates, invalid)

> Evaluating pipelines...
  * Progress: 8/42 (19%) -- best RMSE: 0.412
  * Progress: 21/42 (50%) -- best RMSE: 0.389
  * Progress: 42/42 (100%) -- best RMSE: 0.381
  [OK] Evaluation complete

> Training best model on full dataset...
  [OK] Model trained: train_RMSE=0.344, CV_RMSE=0.381

> Generating predictions for external dataset...
  [!] Domain shift detected (MMD=0.23) -- predictions may be degraded
  [OK] 412 predictions exported

> Saving artifacts...
  [OK] Model saved: workspace/maize/models/best_model.pkl
  [OK] Report saved: workspace/maize/reports/report.pdf

================================================================================
  [OK] Run completed in 2m 5.9s

  Best pipeline: SavGol(w=11) -> PCA(n=150) -> TabPFN
  Metrics: RMSE=0.381  R2=0.82  Pearson=0.91
================================================================================
```

### Example Output: verbose=2 with Branching Pipeline

```
================================================================================
  nirs4all run: multi_preprocessing_comparison
  Started: 2025-12-16 19:12:03
  ----------------------------------------------------------------------------
  Environment:
    Python 3.12.2 | sklearn 1.5.2 | torch 2.6.0+cu128 | jax 0.4.33
    CUDA 12.8 | GPU: RTX 3090 Ti (24GB)
  Reproducibility:
    seed=1337 | git=8f3c9a1 | config_hash=2b31c7d1
================================================================================

> Loading data...
  | Source: s3://naomics/maize/v7/
  | Format: X=OPUS, Y=CSV, meta=Parquet
  [OK] Loaded: 3,482 samples x 2,150 features (2.77s)

> Building cross-validation splits...
  [OK] 5-fold ready: [697, 696, 697, 696, 696] (1.34s)

> Executing shared preprocessing...
  [OK] MinMaxScaler applied

> Entering branch block (4 branches)...
  |
  |-- [branch:snv] SNV preprocessing
  |   [OK] StandardNormalVariate applied (0.12s)
  |
  |-- [branch:msc] MSC preprocessing
  |   [OK] MultiplicativeScatterCorrection applied (0.18s)
  |
  |-- [branch:savgol] Savitzky-Golay preprocessing
  |   [OK] SavitzkyGolay(window=11, poly=2) applied (0.09s)
  |
  |-- [branch:derivative] Derivative preprocessing
  |   [OK] FirstDerivative applied (0.08s)
  |

> Evaluating models per branch...
  |
  |-- [branch:snv] Evaluating PLS(n=5)...
  |   * fold 1/5: RMSE=0.412
  |   * fold 2/5: RMSE=0.398
  |   * fold 3/5: RMSE=0.421
  |   * fold 4/5: RMSE=0.405
  |   * fold 5/5: RMSE=0.389
  |   [OK] CV_RMSE=0.405, R2=0.81
  |
  |-- [branch:msc] Evaluating PLS(n=5)...
  |   [OK] CV_RMSE=0.392, R2=0.83
  |
  |-- [branch:savgol] Evaluating PLS(n=5)...
  |   [OK] CV_RMSE=0.381, R2=0.85  <- best
  |
  |-- [branch:derivative] Evaluating PLS(n=5)...
  |   [OK] CV_RMSE=0.398, R2=0.82
  |

> Branch comparison summary:
  +------------+----------+-------+--------+
  | Branch     | CV_RMSE  | R2    | Rank   |
  +------------+----------+-------+--------+
  | savgol     | 0.381    | 0.85  | 1      |
  | msc        | 0.392    | 0.83  | 2      |
  | derivative | 0.398    | 0.82  | 3      |
  | snv        | 0.405    | 0.81  | 4      |
  +------------+----------+-------+--------+

================================================================================
  [OK] Run completed in 1m 42.3s

  Best branch: savgol
  Best pipeline: MinMaxScaler -> SavitzkyGolay(w=11) -> PLS(n=5)
  Metrics: RMSE=0.381  R2=0.85  Pearson=0.92
================================================================================
```

### Example Output: Multi-Source Pipeline

```
> Loading multi-source dataset...
  | Source[0]: NIR spectra (1,200 samples x 2,150 features)
  | Source[1]: MIR spectra (1,200 samples x 1,800 features)
  | Source[2]: Raman spectra (1,200 samples x 3,200 features)
  [OK] 3 sources loaded, 1,200 samples total

> Executing per-source preprocessing...
  |
  |-- [source:0/NIR] Preprocessing...
  |   [OK] SNV + SavGol applied
  |
  |-- [source:1/MIR] Preprocessing...
  |   [OK] MSC + Baseline applied
  |
  |-- [source:2/Raman] Preprocessing...
  |   [OK] Baseline + Normalize applied
  |

> Concatenating sources for fusion model...
  [OK] Fused features: 1,200 x 7,150
```

### Example Output: Stacking Pipeline

```
> Collecting predictions from branches for stacking...
  | Branches: snv, msc, savgol, derivative
  | Branch predictions shape: (3482, 4)
  [OK] Stacking features collected

> Training meta-model...
  | Meta-model: Ridge(alpha=1.0)
  | Input shape: (3482, 4) -> (3482, 1)
  [OK] Meta-model trained (0.34s)
    +-- Stacking CV_RMSE: 0.362 (improvement: +5.0% vs best branch)

> Stacking comparison:
  +----------------+----------+-------+
  | Model          | CV_RMSE  | R2    |
  +----------------+----------+-------+
  | Stacking-Ridge | 0.362    | 0.87  |  <- best
  | savgol-PLS     | 0.381    | 0.85  |
  | msc-PLS        | 0.392    | 0.83  |
  +----------------+----------+-------+
```

### Example Output: Nested Branching

For complex nested branch structures (branch within branch):

```
> Entering outer branch block (2 branches)...
  |
  |-- [branch:preprocessing/scatter] Scatter correction variants
  |   |
  |   |-- [branch:preprocessing/scatter/snv]
  |   |   [OK] SNV applied
  |   |
  |   |-- [branch:preprocessing/scatter/msc]
  |   |   [OK] MSC applied
  |   |
  |
  |-- [branch:preprocessing/derivative] Derivative variants
  |   |
  |   |-- [branch:preprocessing/derivative/first]
  |   |   [OK] FirstDerivative applied
  |   |
  |   |-- [branch:preprocessing/derivative/second]
  |   |   [OK] SecondDerivative applied
  |   |
  |

> Total branch combinations: 4
  | preprocessing/scatter/snv
  | preprocessing/scatter/msc
  | preprocessing/derivative/first
  | preprocessing/derivative/second
```

---

## 6. File Logging Format

### Purpose

File logs serve different needs than console output:
- **Searchable**: Full timestamps, structured data
- **Complete**: All levels captured (not filtered by verbosity)
- **Parseable**: JSON Lines format for tooling integration
- **Archivable**: Run history for reproducibility

### Output Location

```
{workspace}/logs/{run_id}.log      # Human-readable log
{workspace}/logs/{run_id}.jsonl    # Machine-readable log (optional)
```

### Human-Readable Log Format

```
2025-12-16T19:12:03.041+01:00 [INFO ] Loading data...
2025-12-16T19:12:05.882+01:00 [INFO ] [OK] Loaded maize_nir_v7: 3,482 samples x 2,150 features
2025-12-16T19:12:05.889+01:00 [DEBUG] Cache hit for dataset hash: 4d01...
2025-12-16T19:14:02.880+01:00 [WARN ] Domain shift detected (MMD=0.23)
```

### JSON Lines Format (Optional)

For integration with log aggregation systems (ELK, Loki, etc.):

```json
{"ts": "2025-12-16T19:12:03.041+01:00", "level": "INFO", "run_id": "R-20251216-191203", "phase": "data", "event": "load_start", "dataset": "maize_nir_v7"}
{"ts": "2025-12-16T19:12:05.882+01:00", "level": "INFO", "run_id": "R-20251216-191203", "phase": "data", "event": "load_complete", "samples": 3482, "features": 2150, "duration_ms": 2841}
{"ts": "2025-12-16T19:12:07.239+01:00", "level": "INFO", "run_id": "R-20251216-191203", "phase": "branch", "branch_name": "snv", "branch_path": ["preprocessing", "scatter", "snv"], "branch_index": 0, "event": "start"}
```

---

## 7. Event Categories

### Phase Events

Major workflow phases for high-level tracking:

| Phase | Description | Key Events |
|-------|-------------|------------|
| `init` | Run initialization | start, config_load, environment |
| `data` | Data loading/validation | load, validate, transform |
| `split` | CV split creation | build, leakage_check |
| `generate` | Pipeline generation | expand, prune, deduplicate |
| `branch` | Branch execution | enter, step, exit, compare |
| `source` | Multi-source processing | load, preprocess, concat |
| `evaluate` | Pipeline evaluation | start, progress, complete |
| `stack` | Stacking/ensemble | collect, train_meta, predict |
| `train` | Final model training | start, complete |
| `predict` | Prediction generation | start, complete |
| `export` | Artifact saving | model, report, predictions |
| `complete` | Run completion | summary, cleanup |

### Pipeline Structure Events

For complex pipelines:

| Event | Context Fields | Description |
|-------|---------------|-------------|
| `branch_enter` | `branch_name`, `branch_path`, `branch_index`, `total_branches` | Entering a branch |
| `branch_exit` | `branch_name`, `duration_ms`, `metrics` | Exiting a branch |
| `branch_compare` | `rankings`, `best_branch` | Branch comparison summary |
| `source_process` | `source_index`, `source_name`, `total_sources` | Per-source processing |
| `source_concat` | `final_shape`, `source_shapes` | Source concatenation |
| `stack_collect` | `n_models`, `branch_sources` | Collecting for stacking |
| `stack_meta_train` | `meta_model`, `input_shape` | Training meta-model |
| `nested_branch_enter` | `parent_branch`, `child_branch`, `depth` | Entering nested branch |

---

## 8. Progress Reporting

### Throttling Strategy

To avoid flooding terminals during long operations:

1. **Time-based**: Max 1 update per 5 seconds
2. **Percentage-based**: Update at 10%, 25%, 50%, 75%, 90%, 100%
3. **Event-based**: Always report new best results
4. **Batch-based**: For small batches, report start/end only
5. **Branch-aware**: Show branch context in progress updates

### Progress Format

```
  * Progress: 21/42 (50%) -- best RMSE: 0.389 [branch:savgol, fold:3]
```

For nested structures:
```
  * [branch:snv] fold 3/5: RMSE=0.421
```

---

## 9. Implementation Architecture

### Module Structure

```
nirs4all/
|-- core/
|   +-- logging/
|       |-- __init__.py          # Public API
|       |-- config.py            # Configuration & initialization
|       |-- context.py           # Run context (run_id, phase, branch tracking)
|       |-- formatters.py        # Console & file formatters (ASCII/Unicode)
|       |-- handlers.py          # Custom handlers (throttle, file rotation)
|       +-- events.py            # Event types & structured logging
```

### Public API

```python
from nirs4all.core.logging import get_logger, configure_logging, LogContext

# Module-level logger (standard Python pattern)
logger = get_logger(__name__)

# Configuration at application entry point
configure_logging(
    verbose=1,                    # 0-3 verbosity level
    log_file=True,                # Enable file logging
    log_format="pretty",          # "pretty" | "json" | "minimal"
    show_progress=True,           # Show progress updates
    use_unicode=False,            # False for ASCII-only (cluster-safe)
    use_colors=True,              # Use ANSI colors (auto-detect TTY)
)

# Context management for run tracking
with LogContext(run_id="my-experiment", project="protein-analysis"):
    logger.info("Starting analysis")

    with LogContext.branch("snv", index=0, total=4):
        logger.info("Processing SNV branch")
```

### PipelineRunner Integration

```python
class PipelineRunner:
    def __init__(
        self,
        workspace_path: Optional[Union[str, Path]] = None,
        verbose: int = 0,
        mode: str = "train",
        save_files: bool = True,
        enable_tab_reports: bool = True,
        continue_on_error: bool = False,
        show_spinner: bool = True,
        keep_datasets: bool = True,
        plots_visible: bool = False,
        random_state: Optional[int] = None,
        # NEW: Logging configuration
        log_file: bool = True,              # Write logs to workspace/logs/
        log_format: str = "pretty",         # "pretty" | "json" | "minimal"
        use_unicode: bool = True,           # False for ASCII-only output
        use_colors: bool = True,            # ANSI colors in terminal
    ):
        """Initialize pipeline runner.

        The `verbose` parameter controls log level:
        - 0: WARNING+ only (silent operation)
        - 1: INFO (standard, recommended for research)
        - 2: DEBUG (detailed, for troubleshooting)
        - 3: TRACE (full trace, per-fold details)

        For HPC/cluster environments without Unicode support, set `use_unicode=False`
        to use ASCII-only symbols and box characters.
        """
        # Configure logging based on runner settings
        configure_logging(
            verbose=verbose,
            log_file=log_file,
            log_dir=workspace_path / "logs" if workspace_path else None,
            log_format=log_format,
            use_unicode=use_unicode,
            use_colors=use_colors,
        )
```

### Logger Usage Examples

```python
from nirs4all.core.logging import get_logger

logger = get_logger(__name__)

# Standard logging
logger.info("Loading dataset", dataset="maize_v7", samples=3482)
logger.warning("Domain shift detected", mmd=0.23, threshold=0.15)
logger.error("Training failed", exception=str(e))

# Phase markers
logger.phase_start("evaluate", pipelines=42, metric="RMSE")
logger.phase_complete("evaluate", duration=93.8, best_score=0.381)

# Branch context (automatic indentation)
with logger.branch_context("snv", index=0, total=4):
    logger.info("Applying SNV preprocessing")
    logger.progress("fold", current=3, total=5, rmse=0.421)

# Nested branch context
with logger.branch_context("preprocessing", index=0, total=2):
    with logger.branch_context("scatter", index=0, total=2, parent="preprocessing"):
        logger.info("Processing scatter/snv combination")

# Multi-source context
with logger.source_context("NIR", index=0, total=3):
    logger.info("Processing NIR spectra", features=2150)

# Stacking context
with logger.stack_context(n_branches=4, meta_model="Ridge"):
    logger.info("Collecting branch predictions for stacking")

# Progress updates (auto-throttled)
logger.progress("evaluate", current=21, total=42, best_score=0.389)

# Metrics (structured for downstream analysis)
logger.metric("RMSE", value=0.381, scope="cv", fold=None, pipeline="pip:19be")

# Artifacts
logger.artifact("model", path="models/best.pkl", size_bytes=44_300_000)
```

---

## 10. Configuration

### PipelineRunner Parameters (Primary Configuration)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | int | 0 | Verbosity level (0-3) |
| `log_file` | bool | True | Write logs to workspace/logs/ |
| `log_format` | str | "pretty" | Output format: "pretty", "json", "minimal" |
| `use_unicode` | bool | True | Use Unicode symbols (False for ASCII-only) |
| `use_colors` | bool | True | Use ANSI colors (auto-disabled if not TTY) |

### Environment Variables (Override Only)

Environment variables can override programmatic settings for deployment scenarios:

| Variable | Default | Description |
|----------|---------|-------------|
| `NIRS4ALL_LOG_LEVEL` | (from verbose) | Override minimum log level |
| `NIRS4ALL_LOG_FILE` | (from log_file) | Override log file path |
| `NIRS4ALL_NO_COLOR` | `0` | Force disable ANSI colors |
| `NIRS4ALL_ASCII_ONLY` | `0` | Force ASCII-only output |

**Note**: Programmatic configuration via `PipelineRunner` or `configure_logging()` takes precedence over environment variables by default.

---

## 11. Migration Strategy

### Phase 1: Infrastructure (Week 1)

1. Create `nirs4all/core/logging/` module
2. Implement logger with formatters (ASCII and Unicode modes)
3. Add `configure_logging()` and `get_logger()` APIs
4. Add branch/source/stacking context support

### Phase 2: Core Migration (Week 2)

1. **Delete** `nirs4all/utils/emoji.py`
2. **Remove** emoji stripping code from `nirs4all/__init__.py`
3. Replace `print()` calls in:
   - `nirs4all/pipeline/` - Runner, Orchestrator, Predictor
   - `nirs4all/controllers/` - All controllers (including flow, models, charts)
   - `nirs4all/analysis/` - Selector, results

### Phase 3: Peripheral Migration (Week 3)

Replace `print()` calls in:
- `nirs4all/cli/` - CLI commands
- `nirs4all/optimization/` - Optuna integration
- `nirs4all/visualization/` - Chart generation

### Phase 4: Polish (Week 4)

1. Add progress bar support (optional, TTY-aware)
2. Implement file rotation
3. Add JSON Lines output
4. Update documentation
5. Update all examples to use new runner parameters

### Migration Checklist per File

- [ ] Remove `from nirs4all.utils.emoji import ...`
- [ ] Replace `print(f"...")` with `logger.info("...", key=value)`
- [ ] Replace `print(f"{WARNING}...")` with `logger.warning("...")`
- [ ] Replace `if verbose > N: print(...)` with appropriate logger level
- [ ] Add `logger = get_logger(__name__)` at module top
- [ ] Add branch/source context where applicable
- [ ] Test with `verbose=0,1,2` and `use_unicode=False`

---

## 12. Appendix: ASCII-Safe Format Reference

### Separators and Borders

```
================================================================================
Section Title
================================================================================

----------------------------------------------------------------------------
Subsection
----------------------------------------------------------------------------

+------------+----------+-------+
| Column 1   | Column 2 | Col 3 |
+------------+----------+-------+
| value      | value    | value |
+------------+----------+-------+
```

### Indentation and Hierarchy

```
> Phase start
  | Context line
  | Another context
  [OK] Completion message
    +-- Sub-item 1
    +-- Sub-item 2

> Branch block
  |
  |-- [branch:name] Description
  |   [OK] Branch result
  |
  |-- [branch:other] Description
  |   [OK] Branch result
  |
```

### Nested Branch Hierarchy

```
> Entering nested branch structure...
  |
  |-- [branch:outer1] Outer branch 1
  |   |
  |   |-- [branch:outer1/inner1] Inner branch 1
  |   |   [OK] Result
  |   |
  |   |-- [branch:outer1/inner2] Inner branch 2
  |   |   [OK] Result
  |   |
  |
  |-- [branch:outer2] Outer branch 2
  |   |
  |   |-- [branch:outer2/inner1] Inner branch 1
  |   |   [OK] Result
  |   |
  |
```

### Status Indicators

| Status | Symbol | Example |
|--------|--------|---------|
| Starting | `>` | `> Loading data...` |
| Success | `[OK]` | `[OK] Data loaded` |
| In progress | `*` | `* Processing 50%` |
| Warning | `[!]` | `[!] Missing values` |
| Error | `[X]` | `[X] Failed to load` |
| Skipped | `-` | `- Cached, skipping` |

---

## 13. Complex Pipeline Logging Examples

### Generator Syntax with _or_

When using generator syntax like `{"_or_": [...], "pick": N}`:

```
> Generating pipeline variants from generator syntax...
  | Generator: feature_augmentation with _or_ (4 options, pick 2)
  | Combinations: 6 pipelines generated
  [OK] Generator expanded

> Evaluating generated pipelines...
  |
  |-- [variant:1/6] SNV + SavGol
  |   [OK] CV_RMSE=0.381
  |
  |-- [variant:2/6] SNV + Gaussian
  |   [OK] CV_RMSE=0.395
  |
  ...
```

### Cross-Branch Stacking

```
> Validating cross-branch stacking compatibility...
  | Branch type: generator (_or_ syntax)
  | Compatibility: FULL (same samples, model variants)
  [OK] Cross-branch stacking validated

> Collecting predictions from 4 branches...
  | branch:snv predictions: (3482, 1)
  | branch:msc predictions: (3482, 1)
  | branch:savgol predictions: (3482, 1)
  | branch:derivative predictions: (3482, 1)
  [OK] Stacking matrix: (3482, 4)

> Training meta-model on stacked features...
  [OK] Ridge meta-model trained
    +-- Improvement: +5.2% vs best individual branch
```

### Sample Partitioning Branches

```
> Entering sample_partitioner branch block...
  | Partition key: sample_type
  | Partitions: ['calibration', 'validation', 'external']
  |
  |-- [partition:calibration] Training on calibration samples (n=2400)
  |   [OK] Model trained, CV_RMSE=0.352
  |
  |-- [partition:validation] Testing on validation samples (n=800)
  |   [OK] Test RMSE=0.378
  |
  |-- [partition:external] Predicting on external samples (n=282)
  |   [!] Domain shift detected (MMD=0.21)
  |   [OK] Predictions exported
  |
```

---

## 14. Open Questions

1. **Log rotation**: Implement automatic rotation for long-running processes?
2. **Remote logging**: Support for Weights & Biases, MLflow integration?
3. **Notebook detection**: Auto-detect Jupyter and adjust output format?
4. **Metrics integration**: Tie logging into a metrics collection system?

---

## 15. References

- Python logging documentation: https://docs.python.org/3/library/logging.html
- structlog library: https://www.structlog.org/
- Rich library (for pretty output): https://rich.readthedocs.io/
- loguru library (alternative): https://loguru.readthedocs.io/
