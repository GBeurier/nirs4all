"""
Q25: Complex Pipeline with PLS Regression
==========================================
This example demonstrates a complete NIRS analysis pipeline using PLS regression
with complex nested configuration generation.

This showcases:
- Loading real spectral data
- Defining preprocessing pipelines
- Grid search with generators
- Model training and evaluation
- Configuration comparison

Usage:
    python Q25_complex_pipeline_pls.py

Note: This example uses sample data from examples/sample_data/regression/
"""

import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from itertools import islice

from nirs4all.pipeline.config.generator import (
    expand_spec,
    count_combinations,
    expand_spec_iter,
    batch_iter,
    register_preset,
    resolve_presets_recursive,
    clear_presets,
    to_dataframe,
    print_expansion_tree,
    summarize_configs,
)


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


# ============================================================
# PART 1: Setup - Define Presets for NIRS Analysis
# ============================================================
print_section("PART 1: Setting up Configuration Presets")

# Clear any existing presets
clear_presets()

# 1.1 Spectral Preprocessing Presets
print("1.1 Registering spectral preprocessing presets...")

register_preset(
    "snv_transform",
    {"class": "SNV", "params": {}},
    description="Standard Normal Variate transform",
    tags=["preprocessing", "scatter"]
)

register_preset(
    "msc_transform",
    {"class": "MSC", "params": {}},
    description="Multiplicative Scatter Correction",
    tags=["preprocessing", "scatter"]
)

register_preset(
    "detrend_options",
    {
        "_or_": [
            {"class": "Detrend", "params": {"order": 1}},
            {"class": "Detrend", "params": {"order": 2}}
        ]
    },
    description="Linear and quadratic detrending",
    tags=["preprocessing", "baseline"]
)

register_preset(
    "savgol_options",
    {
        "_grid_": {
            "class": ["SavitzkyGolay"],
            "params": {
                "_grid_": {
                    "window_length": [5, 11, 21],
                    "polyorder": [2, 3],
                    "deriv": [0, 1, 2]
                }
            }
        }
    },
    description="Savitzky-Golay filter options for smoothing/derivatives",
    tags=["preprocessing", "smoothing", "derivatives"]
)

# 1.2 PLS Model Presets
print("1.2 Registering PLS model presets...")

register_preset(
    "pls_quick_search",
    {
        "_grid_": {
            "class": ["PLSRegression"],
            "params": {
                "_grid_": {
                    "n_components": {"_range_": [2, 15, 1]},
                    "scale": [True]
                }
            }
        }
    },
    description="Quick PLS search with reasonable component range",
    tags=["model", "pls", "quick"]
)

register_preset(
    "pls_thorough_search",
    {
        "_grid_": {
            "class": ["PLSRegression"],
            "params": {
                "_grid_": {
                    "n_components": {"_range_": [2, 25, 1]},
                    "scale": [True, False],
                    "max_iter": [500, 1000]
                }
            }
        }
    },
    description="Thorough PLS hyperparameter search",
    tags=["model", "pls", "thorough"]
)

# 1.3 Cross-Validation Presets
print("1.3 Registering cross-validation presets...")

register_preset(
    "cv_kfold",
    {
        "_or_": [
            {"class": "KFold", "n_splits": 5, "shuffle": True},
            {"class": "KFold", "n_splits": 10, "shuffle": True}
        ]
    },
    description="K-Fold cross-validation options",
    tags=["cv"]
)

register_preset(
    "cv_repeated",
    {
        "class": "RepeatedKFold",
        "n_splits": 5,
        "n_repeats": {"_or_": [3, 5, 10]}
    },
    description="Repeated K-Fold for robust estimation",
    tags=["cv", "robust"]
)

print("Presets registered successfully!")
print()


# ============================================================
# PART 2: Define Pipeline Configuration
# ============================================================
print_section("PART 2: Pipeline Configuration Definition")

# 2.1 Simple Pipeline (Quick Testing)
print("2.1 Simple pipeline for quick testing:")
simple_pipeline = {
    "name": "quick_pls",
    "preprocessing": {
        "_chain_": [
            {"_preset_": "snv_transform"},
            {"_preset_": "detrend_options"}
        ]
    },
    "model": {"_preset_": "pls_quick_search"},
    "cv": {"class": "KFold", "n_splits": 5, "shuffle": True}
}

simple_resolved = resolve_presets_recursive(simple_pipeline)
simple_count = count_combinations(simple_resolved)
print(f"Simple pipeline configurations: {simple_count}")
print()

# 2.2 Complex Nested Pipeline (Full Search)
print("2.2 Complex nested pipeline for thorough search:")
complex_pipeline = {
    "name": "thorough_search",

    # Preprocessing: Sequential or single transforms
    "preprocessing": {
        "_or_": [
            # Option 1: SNV alone
            {"_preset_": "snv_transform"},

            # Option 2: MSC alone
            {
                "_chain_": [
                    {"_preset_": "snv_transform"},
                    {"_preset_": "savgol_options"}
                ]
            },

            # Option 4: Detrend + SNV
            {
                "_chain_": [
                    {"_preset_": "detrend_options"},
                    {"_preset_": "snv_transform"}
                ]
            }
        ],
        # Selection options: pick 1, get all
        "pick": 1
    },

    # Feature selection (optional)
    "feature_selection": {
        "_or_": [
            None,  # No feature selection
            {
                "class": "SelectKBest",
                "k": {"_or_": [50, 100, 200, "all"]}
            },
            {
                "class": "VarianceThreshold",
                "threshold": {"_log_range_": [0.001, 0.1, 5]}
            }
        ]
    },

    # Model: PLS with various configurations
    "model": {"_preset_": "pls_thorough_search"},

    # Cross-validation strategy
    "cv": {"_preset_": "cv_kfold"},

    # Evaluation metrics
    "metrics": ["r2", "rmse", "mae", "rpd"]
}

complex_resolved = resolve_presets_recursive(complex_pipeline)
complex_count = count_combinations(complex_resolved)
print(f"Complex pipeline configurations: {complex_count:,}")
print()

# 2.3 Visualize Configuration Space
print("2.3 Configuration Space Visualization:")
print()
print("Expansion Tree (depth limited):")
print(print_expansion_tree(complex_resolved, max_depth=3))
print()


# ============================================================
# PART 3: Efficient Iteration with Large Config Spaces
# ============================================================
print_section("PART 3: Efficient Iteration Strategies")

# 3.1 Random Sampling for Initial Exploration
print("3.1 Random sampling for initial exploration:")
print(f"Total configs: {complex_count:,}")
print("Sampling 20 configurations for initial screening...")

sampled_configs = list(expand_spec_iter(complex_resolved, seed=42, sample_size=20))
print(f"Sampled {len(sampled_configs)} configurations")
print()

# Show diversity
sample_summary = summarize_configs(sampled_configs)
print("Sample diversity:")
for key, info in sample_summary["keys"].items():
    if key != "name" and key != "metrics":
        unique = info["unique_count"]
        print(f"  {key}: {unique} unique values")
print()

# 3.2 Batch Processing
print("3.2 Batch processing for model training:")
print("Using batches of 100 configurations for parallel training...")

batch_count = 0
for batch in batch_iter(complex_resolved, batch_size=100):
    batch_count += 1
    if batch_count <= 3:
        print(f"  Batch {batch_count}: {len(batch)} configs")
    elif batch_count == 4:
        print("  ...")

print(f"Total batches: {batch_count}")
print()

# 3.3 Staged Evaluation
print("3.3 Staged evaluation strategy:")
print()

# Stage 1: Coarse grid
stage1_spec = {
    "preprocessing": {
        "_or_": [
            {"class": "SNV"},
            {"class": "MSC"}
        ]
    },
    "model": {
        "_grid_": {
            "class": ["PLSRegression"],
            "params": {
                "_grid_": {
                    "n_components": {"_range_": [5, 20, 5]}  # [5, 10, 15, 20]
                }
            }
        }
    }
}
stage1_count = count_combinations(stage1_spec)
print(f"Stage 1 (coarse): {stage1_count} configs")

# Stage 2: Fine-tune around best
# (Simulated - in practice, use results from stage 1)
best_preprocessing = "SNV"
best_n_components = 12

stage2_spec = {
    "preprocessing": {"class": best_preprocessing},
    "model": {
        "_grid_": {
            "class": ["PLSRegression"],
            "params": {
                "_grid_": {
                    # Fine grid around best
                    "n_components": {"_range_": [10, 15, 1]},  # [10, 11, 12, 13, 14, 15]
                    "scale": [True, False]
                }
            }
        }
    }
}
stage2_count = count_combinations(stage2_spec)
print(f"Stage 2 (fine-tune): {stage2_count} configs")
print(f"Total: {stage1_count + stage2_count} (vs {complex_count} exhaustive)")
print()


# ============================================================
# PART 4: Configuration Analysis and Comparison
# ============================================================
print_section("PART 4: Configuration Analysis")

# 4.1 Generate some sample configurations
print("4.1 Analyzing configuration distribution:")
sample_size = min(100, complex_count)
configs = list(expand_spec_iter(complex_resolved, seed=12345, sample_size=sample_size))

# Convert to DataFrame for analysis
try:
    df = to_dataframe(configs)
    print(f"Configuration DataFrame shape: {df.shape}")
    print()

    # Show column info
    print("DataFrame columns:")
    for col in df.columns:
        print(f"  - {col}")
    print()

except ImportError:
    print("(pandas not available - using dict analysis)")
    configs_summary = summarize_configs(configs)
    print(f"Analyzed {configs_summary['count']} configs")
print()

# 4.2 Configuration Templates
print("4.2 Creating focused configuration templates:")

# Template: Simple robust
simple_robust = {
    "preprocessing": [
        {"class": "SNV"}
    ],
    "feature_selection": None,
    "model": {
        "class": "PLSRegression",
        "params": {"n_components": 10, "scale": True, "max_iter": 500}
    },
    "cv": {"class": "KFold", "n_splits": 5, "shuffle": True}
}
print("Simple robust config:")
print("  Preprocessing: SNV")
print("  Model: PLS with 10 components")
print("  CV: 5-fold")
print()

# Template: Thorough derivative
thorough_derivative = {
    "preprocessing": [
        {"class": "SNV"},
        {"class": "SavitzkyGolay", "params": {"window_length": 11, "polyorder": 2, "deriv": 1}}
    ],
    "feature_selection": {"class": "SelectKBest", "k": 100},
    "model": {
        "_grid_": {
            "class": ["PLSRegression"],
            "params": {
                "_grid_": {
                    "n_components": {"_range_": [5, 15, 1]},
                    "scale": [True, False]
                }
            }
        }
    },
    "cv": {"class": "KFold", "n_splits": 10, "shuffle": True}
}
thorough_count = count_combinations(thorough_derivative)
print("Thorough derivative config:")
print("  Preprocessing: SNV + SavGol 1st derivative")
print("  Feature selection: SelectKBest (k=100)")
print(f"  Model grid: {thorough_count} PLS configurations")
print("  CV: 10-fold")
print()


# ============================================================
# PART 5: Integration with nirs4all Pipeline
# ============================================================
print_section("PART 5: Pipeline Integration Pattern")

print("In a full nirs4all pipeline, the workflow would be:")
print()
print("1. Load spectral data:")
print("   >>> from nirs4all.data import load_spectrum")
print("   >>> data = load_spectrum('examples/sample_data/regression/dataset.csv')")
print()
print("2. Define configuration with generators:")
print("   >>> config = {")
print("   ...     'preprocessing': {...},")
print("   ...     'model': {'_preset_': 'pls_search'},")
print("   ...     'cv': {'_preset_': 'cv_kfold'}")
print("   ... }")
print()
print("3. Expand configurations:")
print("   >>> configs = list(expand_spec_iter(config, sample_size=100))")
print()
print("4. Run grid search:")
print("   >>> from nirs4all.pipeline import Pipeline")
print("   >>> results = []")
print("   >>> for cfg in configs:")
print("   ...     pipeline = Pipeline.from_config(cfg)")
print("   ...     score = pipeline.fit(X, y).score()")
print("   ...     results.append((cfg, score))")
print()
print("5. Analyze results:")
print("   >>> df = to_dataframe([r[0] for r in results])")
print("   >>> df['score'] = [r[1] for r in results]")
print("   >>> best = df.sort_values('score', ascending=False).head(10)")
print()


# ============================================================
# PART 6: Advanced Selection Patterns
# ============================================================
print_section("PART 6: Advanced Selection Patterns")

# 6.1 Ordered chains for dependent preprocessing
print("6.1 Ordered preprocessing chains (then_pick):")
ordered_preprocessing = {
    "_or_": [
        "SNV",
        "MSC",
        {"class": "SavitzkyGolay", "deriv": 1},
        {"class": "SavitzkyGolay", "deriv": 2}
    ],
    # Select 1-3 in specific order
    "then_pick": (1, 3)
}
chain_configs = expand_spec(ordered_preprocessing, seed=42)
print(f"Generated {len(chain_configs)} ordered chains:")
for cfg in chain_configs[:10]:
    print(f"  {cfg}")
if len(chain_configs) > 10:
    print(f"  ... ({len(chain_configs) - 10} more)")
print()

# 6.2 Constrained selection
print("6.2 Constrained preprocessing selection:")
constrained_spec = {
    "_or_": ["SNV", "MSC", "Detrend", "SavGol_d1", "SavGol_d2"],
    "pick": 2,
    "_mutex_": [
        ["SNV", "MSC"],  # Don't combine scatter corrections
        ["SavGol_d1", "SavGol_d2"]  # Don't combine derivatives
    ]
}
constrained_result = expand_spec(constrained_spec)
print(f"Constrained combinations: {len(constrained_result)}")
print(f"Results: {constrained_result}")
print()


# ============================================================
# Summary
# ============================================================
print_section("SUMMARY")

print("This example demonstrated:")
print()
print("1. PRESET SYSTEM")
print("   - Defined reusable presets for preprocessing, models, and CV")
print("   - Composed complex pipelines from presets")
print("   - Resolved nested preset references")
print()
print("2. CONFIGURATION SPACES")
print(f"   - Simple pipeline: {simple_count} configs")
print(f"   - Complex pipeline: {complex_count:,} configs")
print()
print("3. EFFICIENT ITERATION")
print("   - Random sampling for exploration")
print("   - Batch processing for parallel training")
print("   - Staged evaluation for efficiency")
print()
print("4. ANALYSIS UTILITIES")
print("   - DataFrame conversion")
print("   - Configuration summarization")
print("   - Tree visualization")
print()
print("5. INTEGRATION PATTERNS")
print("   - Pipeline construction from configs")
print("   - Grid search workflow")
print("   - Results analysis")
print()
print("Example completed successfully!")
