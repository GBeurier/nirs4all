# Documentation Update Roadmap

> **Status**: ✅ Complete (All 7 Phases)
> **Created**: 2024-12-27
> **Updated**: 2025-12-27
> **Target**: v0.6.0 release

This document outlines the complete reorganization and update plan for nirs4all documentation.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Target Structure](#target-structure)
3. [Missing Documentation](#missing-documentation)
4. [Implementation Phases](#implementation-phases)
5. [Content Update Checklist](#content-update-checklist)
6. [Writing Guidelines](#writing-guidelines)

---

## Current State Analysis

### Current Structure (Problems)

```
docs/
├── _build/              # 🔧 Build output (RTD) - gitignored
├── _internal/           # 📋 Internal specs & archives (NOT for users)
│   ├── archives/        # Old RST files, deprecated
│   ├── nirs4all_v2_design/  # Internal design docs
│   └── specifications/  # Technical specs (merge_syntax, pipeline_syntax, etc.)
├── api/                 # ❌ MISPLACED - 2 files (storage.md, workspace.md)
├── assets/              # 🖼️ Images (logo, etc.)
├── cli/                 # ❌ MISPLACED - 1 file (workspace_commands.md)
├── explanations/        # ❌ MISPLACED - 7 files (developer/theory docs)
├── reference/           # ❌ MISPLACED - 13 files (syntax specs, cheatsheets)
├── user_guide/          # ❌ MISPLACED - 16 files (how-to guides)
├── source/              # 📖 RTD SOURCE (Sphinx)
│   ├── api/             # Auto-generated RST (150+ files)
│   ├── assets/          # Duplicate assets folder
│   ├── conf.py          # Sphinx config
│   ├── developer/       # 4 files (architecture docs)
│   ├── examples/        # 1 stub file
│   ├── getting_started/ # 1 stub file
│   ├── index.md         # RTD homepage
│   ├── reference/       # 5 files (pipeline_syntax, cli, etc.)
│   └── user_guide/      # 5 files (stacking, preprocessing, etc.)
├── make.bat             # Windows Sphinx build
└── readthedocs.requirements.txt
```

### Identified Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| **Duplicate folders** | `docs/user_guide/` vs `docs/source/user_guide/` | Consolidate into `source/` |
| **Misplaced content** | Files outside `source/` won't be in RTD | Move to appropriate `source/` subfolder |
| **No clear separation** | Internal specs mixed with user docs | Keep specs in `_internal/` |
| **Stub pages** | `getting_started/` and `examples/` are empty | Write actual content |
| **Outdated content** | Many docs reference old APIs or patterns | Update systematically |
| **Missing essentials** | No quickstart, no concepts intro | Write from scratch |

### Files Outside `source/` (Need Migration)

#### `docs/api/` (2 files)
- `storage.md` → `source/reference/storage.md` or `source/developer/`
- `workspace.md` → `source/reference/workspace.md`

#### `docs/cli/` (1 file)
- `workspace_commands.md` → merge into `source/reference/cli.md`

#### `docs/explanations/` (7 files)
- `artifacts_developer_guide.md` → `source/developer/artifacts.md`
- `metadata.md` → `source/developer/`
- `pipeline_architecture.md` → `source/developer/architecture.md`
- `pls_study.md` → `source/examples/` or `_internal/`
- `resampler.md` → `source/reference/` or `source/user_guide/preprocessing/`
- `shap.md` → `source/user_guide/visualization/`
- `snv.md` → `source/user_guide/preprocessing/` (merge with preprocessing docs)

#### `docs/reference/` (13 files)
- `analyzer_charts.md` → `source/user_guide/visualization/`
- `artifacts_system_v2.md` → `source/developer/artifacts.md`
- `augmentations.md` → `source/user_guide/augmentation/`
- `branching.md` → `source/user_guide/pipelines/branching_merging.md`
- `combination_generator.md` → `source/reference/generator_keywords.md`
- `generator_keywords.md` → `source/reference/`
- `operator_catalog.md` → `source/reference/`
- `outputs_vs_artifacts.md` → `source/developer/`
- `prediction_results_list.md` → `source/reference/`
- `quick_reference_prediction_results_list.md` → merge with above
- `synthetic_nirs_generator.md` → `source/user_guide/augmentation/`
- `transfer_preprocessing_selector_cheatsheet.md` → `source/user_guide/preprocessing/`
- `writing_pipelines.md` → `source/user_guide/pipelines/`

#### `docs/user_guide/` (16 files)
- `aggregation.md` → `source/user_guide/data/`
- `api_migration.md` → `source/user_guide/troubleshooting/`
- `dataset_migration_guide.md` → `source/user_guide/troubleshooting/`
- `dataset_troubleshooting.md` → `source/user_guide/troubleshooting/`
- `export_bundles.md` → `source/user_guide/deployment/`
- `force_group_splitting.md` → `source/user_guide/pipelines/`
- `logging.md` → `source/user_guide/` or `source/reference/`
- `migration_guide.md` → `source/user_guide/troubleshooting/`
- `prediction_model_reuse.md` → `source/user_guide/deployment/`
- `preprocessing.md` → `source/user_guide/preprocessing/`
- `preprocessing_cheatsheet.md` → `source/user_guide/preprocessing/`
- `preprocessings_handbook.md` → `source/user_guide/preprocessing/`
- `retrain_transfer.md` → `source/user_guide/deployment/`
- `sample_augmentation.md` → `source/user_guide/augmentation/`
- `sample_augmentation_detailed.md` → merge with above
- `sample_filtering.md` → `source/user_guide/data/`
- `stacking.md` → `source/user_guide/pipelines/`

---

## Target Structure

```
docs/
├── source/                      # 📖 EVERYTHING for RTD goes here
│   ├── conf.py
│   ├── index.md
│   ├── assets/                  # Images, CSS, logos
│   │   └── nirs4all_logo.png
│   │
│   ├── getting_started/         # 🚀 Quick start (5-min read)
│   │   ├── index.md
│   │   ├── installation.md      # From INSTALLATION.md
│   │   ├── quickstart.md        # First pipeline (NEW)
│   │   └── concepts.md          # Core concepts intro (NEW)
│   │
│   ├── user_guide/              # 📖 Task-oriented how-to guides
│   │   ├── index.md
│   │   ├── data/
│   │   │   ├── index.md
│   │   │   ├── loading_data.md       # DatasetConfigs, formats (NEW)
│   │   │   ├── sample_filtering.md
│   │   │   └── aggregation.md
│   │   ├── preprocessing/
│   │   │   ├── index.md
│   │   │   ├── overview.md           # Consolidated from 3+ files
│   │   │   ├── cheatsheet.md
│   │   │   └── handbook.md
│   │   ├── augmentation/
│   │   │   ├── index.md
│   │   │   ├── sample_augmentation.md
│   │   │   ├── feature_augmentation.md
│   │   │   └── synthetic_data.md
│   │   ├── pipelines/
│   │   │   ├── index.md
│   │   │   ├── writing_pipelines.md
│   │   │   ├── branching_merging.md
│   │   │   ├── stacking.md
│   │   │   └── multi_source.md       # source_branch (NEW)
│   │   ├── models/
│   │   │   ├── index.md
│   │   │   ├── training.md           # Basic model training (NEW)
│   │   │   ├── hyperparameter_tuning.md  # Optuna, _range_ (NEW)
│   │   │   └── deep_learning.md      # TF/PyTorch/JAX (NEW)
│   │   ├── deployment/
│   │   │   ├── index.md
│   │   │   ├── export_bundles.md
│   │   │   ├── retrain_transfer.md
│   │   │   └── prediction_reuse.md
│   │   ├── visualization/
│   │   │   ├── index.md
│   │   │   ├── prediction_charts.md  # analyzer_charts.md
│   │   │   ├── spectral_plots.md     # NEW
│   │   │   └── shap_analysis.md
│   │   └── troubleshooting/
│   │       ├── index.md
│   │       ├── migration_guide.md    # Consolidated migrations
│   │       ├── dataset_issues.md
│   │       └── faq.md                # NEW
│   │
│   ├── reference/               # 📚 Complete specifications
│   │   ├── index.md
│   │   ├── pipeline_syntax.md        # Full syntax reference
│   │   ├── operator_catalog.md       # All operators
│   │   ├── generator_keywords.md     # _or_, _range_, etc.
│   │   ├── merge_syntax.md           # branch/merge spec
│   │   ├── cli.md                    # CLI commands
│   │   ├── configuration.md          # Config format spec (NEW)
│   │   ├── metrics.md                # Evaluation metrics (NEW)
│   │   ├── predictions_api.md        # Predictions object
│   │   ├── workspace.md
│   │   └── logging.md
│   │
│   ├── api/                     # 🔧 Auto-generated API docs
│   │   ├── modules.rst
│   │   └── nirs4all.*.rst       # (auto-generated by sphinx-apidoc)
│   │
│   ├── developer/               # 🛠️ For contributors
│   │   ├── index.md
│   │   ├── architecture.md
│   │   ├── controllers.md
│   │   ├── artifacts.md
│   │   ├── testing.md                # NEW
│   │   └── contributing.md           # From CONTRIBUTING.md
│   │
│   └── examples/                # 💡 Annotated examples
│       ├── index.md                  # Links to examples/*.py
│       ├── basic_regression.md
│       ├── classification.md
│       ├── preprocessing_comparison.md
│       ├── branching_example.md
│       └── transfer_learning.md
│
├── _internal/                   # 📋 NOT published (internal use only)
│   ├── specifications/          # Design specs for development
│   ├── archives/                # Deprecated docs
│   └── design/                  # Architecture decision records
│
├── doc_update.md                # THIS FILE
├── make.bat
└── readthedocs.requirements.txt
```

---

## Missing Documentation

### Priority 1: Essential (Blockers for Usability)

| Document | Purpose | Effort |
|----------|---------|--------|
| `getting_started/quickstart.md` | First working pipeline in 2 min | Medium |
| `getting_started/concepts.md` | SpectroDataset, Pipeline, Controllers | Medium |
| `user_guide/data/loading_data.md` | DatasetConfigs, file formats, multi-source | High |
| `user_guide/models/training.md` | Model step, cross-validation basics | Medium |
| `reference/configuration.md` | PipelineConfigs, DatasetConfigs full spec | High |
| `reference/metrics.md` | RMSE, R², classification metrics | Low |
| `examples/index.md` | Curated example index with descriptions | Medium |

### Priority 2: Important (Improves UX significantly)

| Document | Purpose | Effort |
|----------|---------|--------|
| `user_guide/models/hyperparameter_tuning.md` | Optuna integration, _range_ generator | Medium |
| `user_guide/pipelines/multi_source.md` | source_branch, merge_sources | Medium |
| `user_guide/models/deep_learning.md` | TensorFlow/PyTorch/JAX model usage | High |
| `user_guide/visualization/spectral_plots.md` | Visualizing spectra, preprocessing effects | Medium |
| `developer/testing.md` | Running tests, markers, fixtures | Low |
| `user_guide/troubleshooting/faq.md` | Common questions & gotchas | Medium |

### Priority 3: Nice to Have

| Document | Purpose | Effort |
|----------|---------|--------|
| `glossary.md` | NIRS terminology, nirs4all concepts | Low |
| `cookbook.md` | Recipes for specific use cases | High |
| `changelog.md` | Version history in RTD (from CHANGELOG.md) | Low |
| Tutorial series | Step-by-step multi-page tutorial | Very High |

---

## Implementation Phases

### Phase 1: Structure Reorganization
**Goal**: Clean folder structure, no content changes
**Effort**: 1-2 days
**Status**: ✅ Complete
**Completed**: 2024-12-27

#### Tasks
- [x] Create new folder structure in `docs/source/`
  - [x] `user_guide/data/`
  - [x] `user_guide/preprocessing/`
  - [x] `user_guide/augmentation/`
  - [x] `user_guide/pipelines/`
  - [x] `user_guide/models/`
  - [x] `user_guide/deployment/`
  - [x] `user_guide/visualization/`
  - [x] `user_guide/troubleshooting/`
- [x] Move files from `docs/user_guide/` → `docs/source/user_guide/` (appropriate subfolders)
- [x] Move files from `docs/reference/` → `docs/source/reference/`
- [x] Move files from `docs/explanations/` → `docs/source/developer/`
- [x] Move files from `docs/api/` → `docs/source/reference/`
- [x] Move files from `docs/cli/` → merge into `docs/source/reference/cli.md`
- [x] Delete empty old folders (`docs/api/`, `docs/cli/`, `docs/reference/`, etc.)
- [x] Remove duplicate `docs/assets/` (keep only `docs/source/assets/`)
- [ ] Update all internal links in moved files *(deferred to Phase 2)*
- [x] Verify Sphinx build works

#### Final Structure After Phase 1
```
docs/
├── _build/              # Build output
├── _internal/           # Internal specs (pls_study.md moved here)
├── doc_update.md        # This file
├── make.bat
├── readthedocs.requirements.txt
└── source/              # All RTD content
    ├── api/             # Auto-generated API docs
    ├── assets/          # Images, logos (consolidated)
    ├── conf.py
    ├── developer/       # 8 files (architecture, artifacts, controllers, etc.)
    ├── examples/        # 1 stub file
    ├── getting_started/ # 1 stub file
    ├── index.md
    ├── reference/       # 11 files (cli, generator_keywords, operator_catalog, etc.)
    └── user_guide/      # Reorganized into subfolders
        ├── augmentation/    # 4 files
        ├── data/            # 2 files
        ├── deployment/      # 3 files
        ├── index.md
        ├── logging.md
        ├── models/          # (empty - awaiting new docs)
        ├── pipelines/       # 4 files
        ├── preprocessing/   # 6 files
        ├── troubleshooting/ # 4 files
        └── visualization/   # 2 files
```

### Phase 2: Index Files and Navigation
**Goal**: Working navigation, toctrees updated
**Effort**: 1 day
**Status**: ✅ Complete
**Completed**: 2024-12-27

#### Tasks
- [x] Create/update `index.md` for each new folder
  - [x] `user_guide/data/index.md`
  - [x] `user_guide/preprocessing/index.md`
  - [x] `user_guide/augmentation/index.md`
  - [x] `user_guide/pipelines/index.md`
  - [x] `user_guide/models/index.md`
  - [x] `user_guide/deployment/index.md`
  - [x] `user_guide/visualization/index.md`
  - [x] `user_guide/troubleshooting/index.md`
- [x] Update `docs/source/index.md` toctree
- [x] Update `docs/source/user_guide/index.md`
- [x] Update `docs/source/reference/index.md`
- [x] Update `docs/source/developer/index.md`
- [x] Update `docs/source/examples/index.md`
- [x] Update `docs/source/getting_started/index.md`
- [x] Verify all pages accessible from navigation
- [x] Test Sphinx build

#### Summary of Changes
- Created 8 new index files for user_guide subfolders with proper toctrees and grid cards
- Updated user_guide/index.md to link to all subfolders (data, preprocessing, augmentation, pipelines, models, deployment, visualization, troubleshooting)
- Updated reference/index.md to include all 9 reference documents
- Updated developer/index.md to include all 8 developer documents
- Updated examples/index.md to include reference examples (R01-R04)
- Sphinx build successful with only minor warnings (API docs not in toctree)

### Phase 3: Content Consolidation
**Goal**: Merge duplicate/overlapping content
**Effort**: 2-3 days
**Status**: ✅ Complete
**Completed**: 2025-12-27

#### Tasks
- [x] **Preprocessing docs**: Merged `preprocessing.md`, `preprocessing_cheatsheet.md`, `preprocessings_handbook.md` into `overview.md`, `cheatsheet.md`, `handbook.md`
- [x] **Migration docs**: Consolidated `api_migration.md`, `dataset_migration_guide.md`, `migration_guide.md` into single `migration.md`
- [x] **Augmentation docs**: Merged `sample_augmentation.md` and `sample_augmentation_detailed.md` into `sample_augmentation_guide.md`
- [x] **Prediction results**: Merged `prediction_results_list.md` and `quick_reference_prediction_results_list.md` into `predictions_api.md`
- [x] **Artifacts docs**: Consolidated `artifacts_system_v2.md`, `artifacts_developer_guide.md` into `artifacts_internals.md` (kept `artifacts.md` as user guide)
- [x] Remove duplicate content after merging
- [x] Update cross-references and index files
- [x] Sphinx build verified

#### Summary of Changes
- **Preprocessing**: Created 3 consolidated files from 3 overlapping files - `overview.md` (comprehensive guide), `cheatsheet.md` (quick reference by model type), `handbook.md` (in-depth theory and advanced techniques)
- **Migration**: Consolidated 3 migration guides into single comprehensive `migration.md` covering API migration, dataset configuration migration, and prediction format migration
- **Augmentation**: Merged quick reference and detailed guide into single `sample_augmentation_guide.md` with both overview and in-depth sections
- **Prediction Results**: Combined two overlapping prediction results docs into single `predictions_api.md` with both quick reference and detailed usage
- **Artifacts**: Kept existing `artifacts.md` as user guide, created `artifacts_internals.md` for developer-focused implementation details, removed outdated V2 specification
- All index files updated with new document names and links
- Old duplicate files removed

### Phase 4: Content Updates (Outdated Docs)
**Goal**: All existing docs accurate for v0.5.x
**Effort**: 3-5 days
**Status**: ✅ Complete
**Completed**: 2025-12-27

#### Files Reviewed and Updated

##### High Priority (Core Functionality) - ✅ All Complete
- [x] `user_guide/pipelines/writing_pipelines.md` - Verified syntax, examples work
- [x] `user_guide/pipelines/branching.md` - Fixed broken links, updated to use Sphinx cross-references
- [x] `user_guide/pipelines/stacking.md` - Fixed imports (DatasetConfigs, PipelineConfigs), updated branch syntax to use dict format, fixed See Also links
- [x] `user_guide/preprocessing/` - All preprocessing docs verified accurate
- [x] `user_guide/deployment/export_bundles.md` - Verified .n4a format, bundle API correct
- [x] `reference/pipeline_syntax.md` - Complete syntax reference verified
- [x] `reference/operator_catalog.md` - All available operators documented

##### Medium Priority - ✅ All Complete
- [x] `user_guide/data/sample_filtering.md` - Fixed See Also links
- [x] `user_guide/data/aggregation.md` - Fixed See Also links
- [x] `user_guide/augmentation/sample_augmentation_guide.md` - Verified
- [x] `user_guide/deployment/retrain_transfer.md` - Fixed See Also links
- [x] `user_guide/visualization/prediction_charts.md` - Renamed from analyzer_charts.md, updated index
- [x] `reference/generator_keywords.md` - Comprehensive and up-to-date
- [x] `reference/cli.md` - Verified workspace CLI commands
- [x] `developer/architecture.md` - Verified architecture overview
- [x] `developer/controllers.md` - Verified controller system docs

##### Lower Priority - ✅ All Complete
- [x] `user_guide/troubleshooting/migration.md` - Comprehensive migration guide
- [x] `reference/workspace.md` - Verified workspace architecture
- [x] `user_guide/logging.md` - Verified logging system docs
- [x] `developer/artifacts.md` - Fixed See Also links

#### Summary of Changes
- **Stacking.md**: Fixed incorrect import `DatasetConfigs` from `nirs4all.data` (not `nirs4all.data.dataset`), added `PipelineConfigs` import, updated branch stacking example to use dict syntax instead of deprecated `Branch` class, fixed See Also to use Sphinx cross-references
- **Branching.md**: Updated See Also to use Sphinx cross-references
- **Sample filtering/Aggregation**: Updated See Also links to use Sphinx cross-references pointing to correct document paths
- **Retrain/Transfer**: Updated See Also links
- **Visualization**: Renamed `analyzer_charts.md` to `prediction_charts.md` to match target structure, updated visualization index
- **Artifacts**: Fixed See Also links to use correct reference paths

### Phase 5: Write Missing Essential Docs
**Goal**: Complete getting started experience
**Effort**: 4-6 days
**Status**: ✅ Complete
**Completed**: 2025-12-27

#### Tasks

##### Getting Started (Priority 1)
- [x] **`getting_started/installation.md`**
  - Adapted from INSTALLATION.md
  - Added troubleshooting section
  - Added verification command (`nirs4all --test-install`)

- [x] **`getting_started/quickstart.md`**
  - 5-minute first pipeline
  - Load sample data
  - Run preprocessing + PLS
  - View results
  - Export model

- [x] **`getting_started/concepts.md`**
  - SpectroDataset explained
  - Pipeline structure (steps, operators, controllers)
  - Execution flow
  - Key terminology

##### User Guide (Priority 1)
- [x] **`user_guide/data/loading_data.md`**
  - DatasetConfigs overview
  - Supported formats (CSV, Excel, MATLAB, NumPy, Parquet)
  - Column roles (features, targets, metadata)
  - Multi-source datasets
  - Examples for each format

- [x] **`user_guide/models/training.md`**
  - Model step syntax
  - Cross-validation basics
  - Viewing results (Predictions object)
  - Regression vs classification

##### Reference (Priority 1)
- [x] **`reference/configuration.md`**
  - PipelineConfigs full spec
  - DatasetConfigs full spec
  - YAML/dict format
  - All options documented

- [x] **`reference/metrics.md`**
  - Regression: RMSE, MAE, R², RMSEP, SEP, bias, RPD
  - Classification: accuracy, F1, precision, recall, balanced_accuracy
  - How metrics are computed
  - Per-fold vs aggregated

##### Examples (Priority 1)
- [x] **`examples/index.md`**
  - Updated See Also links to new docs
  - Already had organized tables with difficulty level
  - Topics covered and links to example files

#### Summary of Changes
- Created 7 new essential documentation files
- **getting_started/installation.md**: Complete installation guide with GPU support, troubleshooting
- **getting_started/quickstart.md**: 5-minute first pipeline with step-by-step code
- **getting_started/concepts.md**: Core concepts (SpectroDataset, Pipeline, Controllers) with diagrams
- **user_guide/data/loading_data.md**: Comprehensive DatasetConfigs guide with all file formats
- **user_guide/models/training.md**: Model training, cross-validation, result access
- **reference/configuration.md**: Full PipelineConfigs and DatasetConfigs specification
- **reference/metrics.md**: All 30+ regression and classification metrics documented
- Updated index files: getting_started, user_guide/data, user_guide/models, reference, examples
- Sphinx build verified with no errors

### Phase 6: Write Important Missing Docs
**Goal**: Complete user guide coverage
**Effort**: 3-4 days
**Status**: ✅ Complete
**Completed**: 2025-12-27

#### Tasks
- [x] **`user_guide/models/hyperparameter_tuning.md`**
  - Optuna integration
  - `_range_` generator syntax
  - Defining search spaces
  - Viewing optimization results

- [x] **`user_guide/pipelines/multi_source.md`**
  - Multi-source datasets
  - `source_branch` controller
  - `merge_sources` options
  - Use cases (NIR + markers, etc.)

- [x] **`user_guide/models/deep_learning.md`**
  - TensorFlow/Keras models
  - PyTorch models
  - JAX models
  - GPU configuration
  - nicon/decon architectures

- [x] **`user_guide/visualization/spectral_plots.md`**
  - Plotting spectra
  - Before/after preprocessing
  - Chart controllers in pipeline

- [x] **`developer/testing.md`**
  - Running tests with pytest
  - Test markers (sklearn, tensorflow, torch)
  - Running examples
  - Writing new tests

- [x] **`user_guide/troubleshooting/faq.md`**
  - Common errors and solutions
  - Performance tips
  - Best practices

#### Summary of Changes
- **hyperparameter_tuning.md**: Comprehensive guide to Optuna integration, search methods (grid, random, TPE, CMA-ES, hyperband), parameter types, tuning approaches, and visualization of results
- **multi_source.md**: Complete guide to multi-source datasets with source_branch, merge_sources, source weights, selective merging, and practical examples
- **deep_learning.md**: Full documentation of TensorFlow/PyTorch/JAX integration, nicon/decon architectures, model configuration, GPU setup, and framework comparison
- **spectral_plots.md**: Guide to spectral visualization with chart_2d, chart_3d, spectral_distribution, multi-processing visualization, and programmatic plotting
- **testing.md**: Developer guide for running tests, pytest markers, test structure, writing tests, and debugging
- **faq.md**: Comprehensive FAQ covering installation, data loading, pipelines, preprocessing, models, deep learning, results, performance, and troubleshooting
- All index files updated with new documents and grid cards
- Sphinx build verified with no errors

### Phase 7: Polish and Review
**Goal**: Publication-ready documentation
**Effort**: 2-3 days
**Status**: ✅ Complete
**Completed**: 2025-12-27

#### Tasks
- [x] Proofread all documents
- [x] Consistent formatting (headers, code blocks, admonitions)
- [x] Add cross-references between related docs
- [x] Verify all code examples
- [x] Check all links (internal and external)
- [x] Review API docs generation
- [x] Fix mermaid diagram syntax
- [x] Final Sphinx build test
- [ ] Deploy to ReadTheDocs (external step)

#### Summary of Changes
- **Fixed 9 broken cross-references** across documentation:
  - `/user_guide/preprocessing` → `/user_guide/preprocessing/index`
  - `/user_guide/branching_merging` → `/user_guide/pipelines/branching`
  - `/user_guide/troubleshooting/dataset_issues` → `/user_guide/troubleshooting/dataset_troubleshooting`
  - `/user_guide/visualization/shap_analysis` → `/user_guide/visualization/shap`
  - `/reference/merge_syntax` → `/reference/pipeline_syntax`
  - `/developer/contributing` removed (file doesn't exist)
  - Relative path `stacking` → `/user_guide/pipelines/stacking`
- **Fixed mermaid diagram syntax** in 2 files:
  - `user_guide/deployment/index.md`: Changed `\`\`\`mermaid` to `\`\`\`{mermaid}`
  - `developer/pipeline_architecture.md`: Changed `\`\`\`mermaid` to `\`\`\`{mermaid}`
- **Verified all documents** for consistent formatting (headers, code blocks, admonitions)
- **Reviewed API docs generation** - auto-generated by sphinx-apidoc in conf.py
- **Final Sphinx build succeeded** with 4 minor warnings (orphan API docs)
- Documentation now ready for ReadTheDocs deployment

---

## Content Update Checklist

### Per-Document Review Template

```markdown
## Document: [filename]

### Metadata
- Last reviewed: YYYY-MM-DD
- Reviewer: [name]
- Status: ⬜ Not reviewed / 🟡 Needs updates / ✅ Current

### Checks
- [ ] All code examples run successfully
- [ ] API references match current code (v0.5.x)
- [ ] No deprecated features referenced
- [ ] All imports correct
- [ ] Internal links work
- [ ] External links work
- [ ] Consistent with other docs
- [ ] Clear and well-organized

### Issues Found
1. [Issue description]

### Updates Made
1. [Update description]
```

---

## Writing Guidelines

### Style Guide

1. **Voice**: Second person ("you"), active voice
2. **Tense**: Present tense for instructions
3. **Length**: Keep paragraphs short (3-4 sentences max)
4. **Code examples**: Always test before including
5. **Admonitions**: Use `:::{tip}`, `:::{warning}`, `:::{note}` appropriately

### Document Structure Template

```markdown
# Title

Brief introduction (1-2 sentences) explaining what this doc covers.

## Prerequisites

List what the reader should know/have before reading.

## Overview

High-level explanation of the concept/feature.

## Step-by-step Guide / Usage

Detailed instructions with code examples.

## Examples

Complete, runnable examples.

## Common Issues

Troubleshooting section if applicable.

## See Also

Links to related documentation.
```

### Code Example Standards

```python
# Always include imports
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

# Use realistic but minimal examples
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3),
    {"model": PLSRegression(n_components=10)}
]

# Show expected output in comments where helpful
# Expected output: RMSE ≈ 0.15
```

### Linking Conventions

- Internal links: `{doc}../reference/pipeline_syntax`
- API links: `` {py:class}`nirs4all.pipeline.PipelineRunner` ``
- Example links: `[See Q1_basic_regression.py](../../examples/Q1_basic_regression.py)`

---

## Progress Tracking

| Phase | Status | Start Date | End Date | Notes |
|-------|--------|------------|----------|-------|
| 1. Structure Reorganization | ✅ Complete | 2024-12-27 | 2024-12-27 | All files moved, Sphinx build verified |
| 2. Index Files and Navigation | ✅ Complete | 2024-12-27 | 2024-12-27 | All index files created, toctrees updated |
| 3. Content Consolidation | ✅ Complete | 2024-12-27 | 2024-12-27 | Merge syntax consolidated from specs |
| 4. Content Updates | ✅ Complete | 2024-12-27 | 2024-12-27 | All docs reviewed, links fixed, imports updated |
| 5. Write Essential Docs | ✅ Complete | 2025-12-27 | 2025-12-27 | 7 new docs: installation, quickstart, concepts, loading_data, training, configuration, metrics |
| 6. Write Important Docs | ✅ Complete | 2025-12-27 | 2025-12-27 | 6 new docs: hyperparameter_tuning, multi_source, deep_learning, spectral_plots, testing, faq |
| 7. Polish and Review | ✅ Complete | 2025-12-27 | 2025-12-27 | Fixed 9 broken cross-references, mermaid syntax, final build succeeded |

**Legend**: ⬜ Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked

---

## Notes

- Keep `_internal/specifications/` as source of truth for syntax specs
- Auto-generated API docs (`source/api/*.rst`) are rebuilt on each Sphinx run
- Examples in `examples/*.py` serve as integration tests - keep them working
- Consider adding a "What's New" section for each release
