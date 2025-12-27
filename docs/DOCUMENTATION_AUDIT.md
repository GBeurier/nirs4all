# Documentation Audit & Reorganization Plan

**Date**: December 26, 2025
**Status**: Audit Complete - Action Plan Pending
**Author**: Documentation Review

---

## Table of Contents

1. [Part 1: Current State Evaluation](#part-1-current-state-evaluation)
2. [Part 2: Proposed Documentation Architecture](#part-2-proposed-documentation-architecture)
3. [Part 3: Action Plan for Professional RTD](#part-3-action-plan-for-professional-rtd)

---

# Part 1: Current State Evaluation

## Overview of Current Structure

The documentation is spread across multiple directories with significant overlap, outdated content, and incomplete RTD integration:

```
docs/
├── source/                    # RTD source (Sphinx)
│   ├── conf.py               # Sphinx config (uses myst_parser for .md)
│   ├── index.md              # Main landing page
│   ├── tutorials.md          # Points to examples
│   ├── reference.md          # Reference overview
│   ├── architecture.md       # Basic architecture intro
│   ├── api/                  # Auto-generated API docs
│   └── archives/             # OLD .rst files (deprecated)
│
├── api/                       # Standalone API docs (NOT in RTD)
├── cli/                       # CLI documentation
├── explanations/              # Deep-dive explanations (developer)
├── reference/                 # Reference materials
├── specifications/            # Design specs (internal)
├── user_guide/                # User tutorials (NOT in RTD)
└── nirs4all_v2_design/        # Future v2 architecture (internal)
```

## File-by-File Evaluation

### `/docs/source/` (RTD Source Directory)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `conf.py` | ⚠️ Outdated | Essential | Version shows 0.4.2 (current is 0.5.x), has RTD theme + myst_parser |
| `index.md` | ✅ Good | Essential | Good landing page, feature list, installation, citation |
| `tutorials.md` | ⚠️ Partial | Essential | Links to new examples (U01-U27, D01-D22) but examples may not exist yet |
| `reference.md` | ⚠️ Partial | Essential | Links to R01-R04 reference examples |
| `architecture.md` | ⚠️ Outdated | Important | Uses old PipelineRunner API, needs update to new `nirs4all.run()` API |
| `api/module_api.md` | ✅ Good | Essential | Documents new module-level API |
| `api/sklearn_integration.md` | ✅ Good | Essential | NIRSPipeline wrapper documentation |
| `archives/` | ❌ Deprecated | Remove | Old .rst files from previous Sphinx setup |

### `/docs/api/` (Standalone API Docs - NOT in RTD)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `module_api.md` | ⚠️ Duplicate | Merge | Duplicate of `source/api/module_api.md` |
| `sklearn_integration.md` | ⚠️ Duplicate | Merge | Duplicate of `source/api/sklearn_integration.md` |
| `storage.md` | ✅ Good | Developer | Detailed ArtifactRegistry/ArtifactLoader API |
| `workspace.md` | ⚠️ Very Long | Split | 800+ lines, workspace architecture spec (design doc, not user doc) |

### `/docs/cli/` (CLI Documentation)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `workspace_commands.md` | ✅ Good | User | CLI workspace commands, well-documented |

### `/docs/explanations/` (Deep-Dive Explanations)

| File | Status | Utility | Audience | Notes |
|------|--------|---------|----------|-------|
| `artifacts_developer_guide.md` | ✅ Good | Developer | Dev | Artifact system internals |
| `metadata.md` | ❓ Unknown | Check | Dev | Need to verify content |
| `pipeline_architecture.md` | ✅ Good | Developer | Dev | Controller system, custom extensions |
| `pls_study.md` | ⚠️ Unclear | Check | User? | PLS methods comparison |
| `resampler.md` | ⚠️ Unclear | Check | User/Dev | Resampling explanation |
| `shap.md` | ⚠️ Unclear | Check | User | SHAP integration |
| `snv.md` | ⚠️ Unclear | Check | User | SNV preprocessing explanation |

### `/docs/reference/` (Reference Materials)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `analyzer_charts.md` | ⚠️ Check | User | Visualization reference |
| `artifacts_system_v2.md` | ✅ Good | Developer | Artifact system v2 design |
| `augmentations.md` | ✅ Good | Developer | Augmentation operator guidelines |
| `branching.md` | ⚠️ Check | User/Dev | Branching reference |
| `combination_generator.md` | ⚠️ Check | Developer | Generator internals |
| `generator_keywords.md` | ⚠️ Check | User | Generator syntax keywords |
| `operator_catalog.md` | ✅ Good | User/Dev | Complete operator listing |
| `outputs_vs_artifacts.md` | ⚠️ Check | Developer | Terminology clarification |
| `prediction_results_list.md` | ⚠️ Check | Developer | Predictions API |
| `quick_reference_prediction_results_list.md` | ⚠️ Check | User | Quick reference |
| `synthetic_nirs_generator.md` | ⚠️ Check | Developer | Synthetic data generation |
| `transfer_preprocessing_selector_cheatsheet.md` | ⚠️ Check | User | Transfer learning cheatsheet |
| `writing_pipelines.md` | ✅ Excellent | Essential | Comprehensive pipeline syntax guide |

### `/docs/specifications/` (Design Specifications - Internal)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `api_design_v2.md` | 🔒 Internal | Future | v2 API design |
| `api_v2_migration_roadmap.md` | 🔒 Internal | Future | Migration planning |
| `asymmetric_sources_design.md` | 🔒 Internal | Future | Multi-source design |
| `branching_generation_proposal.md` | 🔒 Internal | Future | Branching design |
| `concat_augmentation_specification.md` | 🔒 Internal | Dev | Concat transform spec |
| `config_format.md` | ⚠️ Outdated | Update | Config format (may be superseded) |
| `cross_dataset_metrics.md` | 🔒 Internal | Dev | Metrics design |
| `dataset_config_roadmap.md` | 🔒 Internal | Future | Dataset config evolution |
| `dataset_config_specification.md` | ✅ Good | Dev | Dataset config spec |
| `generator_analysis.md` | 🔒 Internal | Dev | Generator analysis |
| `generator_selection_semantics.md` | 🔒 Internal | Dev | Generator semantics |
| `hash_uniqueness.md` | 🔒 Internal | Dev | Hash collision analysis |
| `logging_specification.md` | 🔒 Internal | Dev | Logging design |
| `manifest.md` | ✅ Good | Dev | Manifest format spec |
| `merge_syntax.md` | ✅ Good | User | Branch/merge syntax |
| `method_preprocessings.md` | 🔒 Internal | Dev | Preprocessing methods |
| `nested_cv.md` | 🔒 Internal | Dev | Nested CV design |
| `pipeline_review.md` | 🔒 Internal | Dev | Pipeline architecture review |
| `pipeline_syntax.md` | ✅ Good | User | Pipeline syntax (with fixes) |
| `pipeline_syntax_duplicate.md` | ❌ Delete | N/A | Duplicate file |
| `prediction_reload_design.md` | 🔒 Internal | Dev | Prediction reload |
| `random_remainder_selection.md` | 🔒 Internal | Dev | Random selection |
| `ranking_system_analysis.md` | 🔒 Internal | Dev | Ranking design |
| `workspace_serialization.md` | 🔒 Internal | Dev | Workspace format |

### `/docs/user_guide/` (User Tutorials - NOT in RTD)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `aggregation.md` | ⚠️ Check | User | Sample aggregation guide |
| `api_migration.md` | ✅ Good | User | Migration from old to new API |
| `dataset_migration_guide.md` | ⚠️ Check | User | Dataset migration |
| `dataset_troubleshooting.md` | ⚠️ Check | User | Troubleshooting |
| `export_bundles.md` | ✅ Excellent | User | Bundle export/import guide |
| `force_group_splitting.md` | ⚠️ Check | User | Group splitting |
| `logging.md` | ✅ Good | User | Logging configuration |
| `migration_guide.md` | ⚠️ Check | User | General migration |
| `prediction_model_reuse.md` | ⚠️ Check | User | Prediction workflows |
| `preprocessing.md` | ✅ Excellent | User | Complete preprocessing reference |
| `preprocessing_cheatsheet.md` | ⚠️ Check | User | Quick reference |
| `preprocessings_handbook.md` | ⚠️ Duplicate? | Check | May overlap with preprocessing.md |
| `retrain_transfer.md` | ⚠️ Check | User | Retrain/transfer guide |
| `sample_augmentation.md` | ⚠️ Check | User | Sample augmentation |
| `sample_augmentation_detailed.md` | ⚠️ Check | User | Detailed augmentation |
| `sample_filtering.md` | ⚠️ Check | User | Sample filtering |
| `stacking.md` | ✅ Excellent | User | Meta-model stacking guide |

### `/docs/nirs4all_v2_design/` (Future v2 Architecture - Internal)

| File | Status | Utility | Notes |
|------|--------|---------|-------|
| `00_onboarding.md` | 🔒 Internal | Future | v2 architecture overview |
| `01_architecture_overview.md` | 🔒 Internal | Future | Detailed architecture |
| `02_data_layer.md` | 🔒 Internal | Future | Data layer design |
| `03_dag_engine.md` | 🔒 Internal | Future | DAG execution engine |
| `04_api_layer.md` | 🔒 Internal | Future | API layer design |
| `05_implementation_plan.md` | 🔒 Internal | Future | Implementation roadmap |

## RTD Integration Issues

### Current RTD Configuration (`docs/source/conf.py`)

```python
# Issues identified:
release = '0.4.2'  # ❌ Outdated (should be 0.5.x)

# toctree in index.md:
# tutorials
# reference
# architecture
# api/module_api
# api/sklearn_integration
# api/modules  # ❌ This file doesn't exist!
```

### What's NOT in RTD Currently

The following valuable content is **NOT integrated** into RTD:

1. **`/docs/user_guide/`** - All user tutorials (export, preprocessing, stacking, etc.)
2. **`/docs/cli/`** - CLI documentation
3. **`/docs/reference/`** - Operator catalog, writing pipelines, etc.
4. **`/docs/explanations/`** - Deep-dive explanations
5. **`/docs/specifications/merge_syntax.md`** - Important user-facing syntax docs

### Broken/Missing Links in RTD

1. `api/modules` - Referenced in toctree but doesn't exist
2. Example links (U01-U27, D01-D22) - Some examples may not exist yet
3. Cross-references between docs are inconsistent

## Summary of Issues

| Issue Category | Count | Priority |
|----------------|-------|----------|
| **Outdated content** | 5+ files | High |
| **Duplicate files** | 4+ files | Medium |
| **Not in RTD** | 25+ files | High |
| **Missing examples** | Unknown | High |
| **Broken links** | Several | Medium |
| **Internal docs exposed** | 20+ files | Low |
| **Inconsistent structure** | Entire tree | High |

---

# Part 2: Proposed Documentation Architecture

## Design Principles

1. **Diátaxis Framework**: Organize by user intent (tutorials, how-to, reference, explanation)
2. **User vs Developer**: Clear separation of audience
3. **RTD-First**: All public docs must be in RTD toctree
4. **Examples as Documentation**: Examples are primary tutorials
5. **Markdown-Only**: Use MyST for Sphinx/RTD compatibility

## Proposed Directory Structure

```
docs/
├── source/                         # RTD source directory
│   ├── conf.py                     # Sphinx configuration
│   ├── index.md                    # Landing page
│   │
│   ├── getting_started/            # 🚀 TUTORIALS (learning-oriented)
│   │   ├── index.md                # Getting started overview
│   │   ├── installation.md         # Installation guide
│   │   ├── quickstart.md           # 5-minute quickstart
│   │   ├── first_pipeline.md       # Hello World tutorial
│   │   └── next_steps.md           # Where to go next
│   │
│   ├── user_guide/                 # 📖 HOW-TO GUIDES (task-oriented)
│   │   ├── index.md                # User guide overview
│   │   ├── data_handling.md        # Loading and managing data
│   │   ├── preprocessing.md        # NIRS preprocessing (from current)
│   │   ├── models.md               # Model training and comparison
│   │   ├── cross_validation.md     # CV strategies
│   │   ├── hyperparameter_tuning.md # Optuna optimization
│   │   ├── branching_merging.md    # Pipeline branching
│   │   ├── stacking.md             # Meta-model stacking (from current)
│   │   ├── export_deploy.md        # Export bundles (from current)
│   │   ├── explainability.md       # SHAP and feature importance
│   │   └── logging.md              # Logging configuration
│   │
│   ├── reference/                  # 📚 REFERENCE (information-oriented)
│   │   ├── index.md                # Reference overview
│   │   ├── api/                    # API Reference
│   │   │   ├── module_api.md       # nirs4all.run(), predict(), etc.
│   │   │   ├── sklearn_wrapper.md  # NIRSPipeline
│   │   │   └── predictions.md      # Predictions/RunResult classes
│   │   ├── pipeline_syntax.md      # Complete pipeline syntax
│   │   ├── generator_syntax.md     # Generator keywords (_or_, _range_)
│   │   ├── operator_catalog.md     # All operators (from current)
│   │   ├── preprocessing_reference.md  # Preprocessing operators
│   │   └── cli.md                  # CLI reference
│   │
│   ├── developer/                  # 🔧 EXPLANATION (understanding-oriented)
│   │   ├── index.md                # Developer guide overview
│   │   ├── architecture.md         # Pipeline architecture
│   │   ├── controllers.md          # Controller system
│   │   ├── custom_extensions.md    # Writing custom controllers
│   │   ├── artifacts.md            # Artifact storage system
│   │   ├── deep_learning.md        # TF/PyTorch/JAX integration
│   │   └── contributing.md         # Contribution guidelines
│   │
│   ├── examples/                   # 📝 EXAMPLES INDEX
│   │   ├── index.md                # Examples overview (link to examples/)
│   │   ├── user_path.md            # User examples (U01-U27)
│   │   └── developer_path.md       # Developer examples (D01-D22)
│   │
│   ├── changelog.md                # Version history
│   └── migration.md                # Migration guides
│
├── _internal/                      # 🔒 INTERNAL (not published)
│   ├── specifications/             # Design specs
│   ├── nirs4all_v2_design/         # Future architecture
│   └── archives/                   # Deprecated docs
│
└── assets/                         # Images, diagrams
```

## Content Migration Plan

### Move to `getting_started/`

| Current Location | New Location | Action |
|------------------|--------------|--------|
| (new) | `installation.md` | Extract from index.md |
| (new) | `quickstart.md` | Create from U01 example |
| `source/architecture.md` | `first_pipeline.md` | Rewrite as tutorial |

### Move to `user_guide/`

| Current Location | New Location | Action |
|------------------|--------------|--------|
| `user_guide/preprocessing.md` | `user_guide/preprocessing.md` | Move as-is |
| `user_guide/stacking.md` | `user_guide/stacking.md` | Move as-is |
| `user_guide/export_bundles.md` | `user_guide/export_deploy.md` | Move + expand |
| `user_guide/logging.md` | `user_guide/logging.md` | Move as-is |
| `specifications/merge_syntax.md` | `user_guide/branching_merging.md` | Move + rewrite |
| `user_guide/sample_augmentation.md` | `user_guide/preprocessing.md` | Merge |
| `user_guide/sample_filtering.md` | `user_guide/cross_validation.md` | Merge |

### Move to `reference/`

| Current Location | New Location | Action |
|------------------|--------------|--------|
| `source/api/module_api.md` | `reference/api/module_api.md` | Move |
| `source/api/sklearn_integration.md` | `reference/api/sklearn_wrapper.md` | Move + rename |
| `reference/writing_pipelines.md` | `reference/pipeline_syntax.md` | Move + rename |
| `reference/operator_catalog.md` | `reference/operator_catalog.md` | Move |
| `cli/workspace_commands.md` | `reference/cli.md` | Move |

### Move to `developer/`

| Current Location | New Location | Action |
|------------------|--------------|--------|
| `explanations/pipeline_architecture.md` | `developer/architecture.md` | Move |
| `explanations/artifacts_developer_guide.md` | `developer/artifacts.md` | Move |
| `api/storage.md` | `developer/artifacts.md` | Merge |
| (new) | `developer/custom_extensions.md` | Create from architecture.md |

### Move to `_internal/` (Not Published)

| Current Location | Action |
|------------------|--------|
| `specifications/*.md` | Move to `_internal/specifications/` |
| `nirs4all_v2_design/*.md` | Move to `_internal/nirs4all_v2_design/` |
| `source/archives/` | Move to `_internal/archives/` |

### Delete (Duplicates/Obsolete)

| File | Reason |
|------|--------|
| `api/module_api.md` | Duplicate of source/api/module_api.md |
| `api/sklearn_integration.md` | Duplicate |
| `specifications/pipeline_syntax_duplicate.md` | Duplicate |
| `user_guide/preprocessings_handbook.md` | Check if duplicate |
| `user_guide/migration_guide.md` | Merge into migration.md |
| `user_guide/dataset_migration_guide.md` | Merge into migration.md |
| `user_guide/api_migration.md` | Merge into migration.md |

## Examples Alignment

The examples have been reorganized into `user/` and `developer/` paths. Documentation should mirror this:

### User Examples (U01-U27) → User Guide Topics

| Examples | User Guide Section |
|----------|-------------------|
| U01-U04 (getting_started) | `getting_started/` |
| U05-U08 (data_handling) | `user_guide/data_handling.md` |
| U09-U12 (preprocessing) | `user_guide/preprocessing.md` |
| U13-U16 (models) | `user_guide/models.md` |
| U17-U20 (cross_validation) | `user_guide/cross_validation.md` |
| U21-U24 (deployment) | `user_guide/export_deploy.md` |
| U25-U27 (explainability) | `user_guide/explainability.md` |

### Developer Examples (D01-D22) → Developer Guide Topics

| Examples | Developer Guide Section |
|----------|------------------------|
| D01-D05 (advanced_pipelines) | `user_guide/branching_merging.md` |
| D06-D09 (generators) | `reference/generator_syntax.md` |
| D10-D13 (deep_learning) | `developer/deep_learning.md` |
| D14-D16 (transfer_learning) | `user_guide/models.md` (subsection) |
| D17-D19 (advanced_features) | `developer/custom_extensions.md` |
| D20-D21 (internals) | `developer/controllers.md` |

## What's Missing (New Content Needed)

| Topic | Priority | Notes |
|-------|----------|-------|
| Installation guide | High | Extract from index.md, add troubleshooting |
| Quickstart tutorial | High | Based on U01, 5-minute getting started |
| Data handling guide | High | Cover all input formats |
| Model training guide | High | Comparison, tuning, ensembles |
| Cross-validation guide | Medium | CV strategies, group splitting |
| Hyperparameter tuning guide | Medium | Optuna integration |
| SHAP/explainability guide | Medium | Based on U25-U27 |
| Contributing guide | Low | For developers |
| Changelog | Low | Version history |

---

# Part 3: Action Plan for Professional RTD

## Phase 1: Foundation (Week 1)

### 1.1 Update Sphinx Configuration

```python
# docs/source/conf.py updates needed:

# Update version
release = '0.5.x'  # Get from pyproject.toml

# Add extensions for better docs
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',     # ADD: Cross-reference to sklearn, numpy
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',               # ADD: Cards, tabs, grids
]

# MyST extensions for rich Markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",               # ADD: Variable substitution
    "tasklist",                   # ADD: Checkboxes
    "attrs_block",                # ADD: Block attributes
]

# Intersphinx for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}
```

### 1.2 Create Directory Structure

```bash
# Create new structure
mkdir -p docs/source/{getting_started,user_guide,reference/api,developer,examples}
mkdir -p docs/_internal/{specifications,archives}
```

### 1.3 Update `readthedocs.requirements.txt`

```txt
# Remove outdated mkdocs entries
# Keep only Sphinx-related
sphinx>=7.0
sphinx-rtd-theme>=2.0
myst-parser>=2.0
sphinx-copybutton>=0.5
sphinx-design>=0.5

# Project dependencies
numpy>=1.20.0
pandas>=1.0.0
scipy>=1.5.0
scikit-learn>=1.0.0

# Install package
-e ..
```

## Phase 2: Content Migration (Week 2)

### 2.1 Core User Guide Content

1. **Move and update `preprocessing.md`**
   - Add code examples from U09-U12
   - Update to new API syntax
   - Add preprocessing decision flowchart

2. **Move and update `stacking.md`**
   - Update code examples to new API
   - Add complete MetaModel reference

3. **Move and update `export_bundles.md`**
   - Rename to `export_deploy.md`
   - Add deployment patterns (Docker, Lambda, FastAPI)

4. **Create `branching_merging.md`**
   - Combine from `merge_syntax.md` and branching examples
   - Add visual diagrams

### 2.2 Reference Documentation

1. **Move `writing_pipelines.md`**
   - Rename to `pipeline_syntax.md`
   - Split into pipeline_syntax.md and generator_syntax.md

2. **Move `operator_catalog.md`**
   - Update to current version
   - Add usage examples

3. **Create `cli.md`**
   - Combine CLI documentation
   - Add examples for all commands

### 2.3 Developer Documentation

1. **Create `architecture.md`**
   - High-level architecture overview
   - Component diagram
   - Data flow

2. **Create `controllers.md`**
   - From `pipeline_architecture.md`
   - Controller registration
   - Custom controller tutorial

## Phase 3: Polish and Quality (Week 3)

### 3.1 Create Landing Page (`index.md`)

```markdown
# nirs4all Documentation

{logo}

**nirs4all** is a Python library for Near-Infrared Spectroscopy (NIRS) data analysis.

:::{grid} 2
:gutter: 2

:::{grid-item-card} 🚀 Getting Started
:link: getting_started/index
:link-type: doc

New to nirs4all? Start here with installation and your first pipeline.
:::

:::{grid-item-card} 📖 User Guide
:link: user_guide/index
:link-type: doc

Step-by-step guides for common tasks and workflows.
:::

:::{grid-item-card} 📚 Reference
:link: reference/index
:link-type: doc

Complete API reference, pipeline syntax, and operator catalog.
:::

:::{grid-item-card} 🔧 Developer
:link: developer/index
:link-type: doc

Architecture, internals, and contribution guidelines.
:::

:::
```

### 3.2 Add Visual Elements

1. **Architecture diagrams**
   - Pipeline execution flow
   - Data flow through operators
   - Branch/merge visualization

2. **Code highlighting**
   - Syntax highlighting for all code blocks
   - Copy buttons for code snippets

3. **Admonitions**
   ```markdown
   :::{tip}
   Use `nirs4all.run()` for the simplest API.
   :::

   :::{warning}
   Deep learning models require the full `.n4a` bundle format.
   :::
   ```

### 3.3 Cross-References and Navigation

1. **Add intersphinx links**
   ```markdown
   See {external:py:class}`sklearn.preprocessing.StandardScaler` for details.
   ```

2. **Add prev/next navigation**
   - Footer navigation between related pages

3. **Add see-also sections**
   ```markdown
   ## See Also

   - {doc}`/user_guide/preprocessing`
   - {doc}`/reference/operator_catalog`
   - [Example: U09_preprocessing_basics.py](https://github.com/...)
   ```

## Phase 4: Automation and CI (Week 4)

### 4.1 RTD Configuration

Create `.readthedocs.yaml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: true

python:
  install:
    - requirements: docs/readthedocs.requirements.txt
    - method: pip
      path: .
```

### 4.2 Documentation CI

Add to GitHub Actions:

```yaml
name: Documentation

on:
  push:
    paths:
      - 'docs/**'
      - 'nirs4all/**'

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build docs
        run: |
          pip install -r docs/readthedocs.requirements.txt
          pip install -e .
          cd docs && make html

      - name: Check links
        run: |
          cd docs && make linkcheck
```

### 4.3 Example Verification

```yaml
  verify-examples:
    runs-on: ubuntu-latest
    steps:
      - name: Run documented examples
        run: |
          cd examples
          ./run.sh -c user -q  # Quick mode, user examples
```

## Summary Checklist

### Immediate Actions (This Week)

- [ ] Update `conf.py` version to 0.5.x
- [ ] Remove `api/modules` from toctree (doesn't exist)
- [ ] Create new directory structure
- [ ] Delete duplicate files
- [ ] Move internal docs to `_internal/`

### Short-Term (2 Weeks)

- [ ] Migrate user_guide content to source/user_guide/
- [ ] Migrate reference content to source/reference/
- [ ] Create getting_started/ with quickstart
- [ ] Update all code examples to new API
- [ ] Add sphinx-design for cards/grids

### Medium-Term (1 Month)

- [ ] Complete all user guide sections
- [ ] Add diagrams and visual aids
- [ ] Create developer documentation
- [ ] Set up CI for docs
- [ ] Verify all example links work

### Long-Term (Ongoing)

- [ ] Keep docs in sync with code changes
- [ ] Add API changelog
- [ ] Expand tutorials based on user feedback
- [ ] Add more advanced examples

---

## Appendix: RTD Configuration Reference

### Toctree Structure for `index.md`

```markdown
```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/index
getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/index
user_guide/preprocessing
user_guide/models
user_guide/branching_merging
user_guide/stacking
user_guide/export_deploy
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
reference/api/module_api
reference/pipeline_syntax
reference/operator_catalog
reference/cli
```

```{toctree}
:maxdepth: 2
:caption: Developer

developer/index
developer/architecture
developer/controllers
developer/contributing
```
```

### MyST Syntax Examples

```markdown
# Admonitions
:::{note}
This is a note.
:::

# Code with line numbers
```{code-block} python
:linenos:
:emphasize-lines: 3-5

import nirs4all
result = nirs4all.run(pipeline, data)
print(result.best_rmse)
```

# Tabs
::::{tab-set}
:::{tab-item} Python
```python
result = nirs4all.run(pipeline, data)
```
:::
:::{tab-item} CLI
```bash
nirs4all run --config pipeline.yaml data/
```
:::
::::
```

---

*Document generated: December 26, 2025*
