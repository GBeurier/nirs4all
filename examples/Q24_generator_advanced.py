"""
Q24: Advanced Generator Features
================================
This example demonstrates advanced generator features added in Phases 3 and 4:

Phase 3 Features:
- _log_range_: Logarithmic numeric sequences
- _grid_: Grid search style parameter expansion
- _zip_: Parallel iteration (positional pairing)
- _chain_: Sequential ordered configurations
- _sample_: Statistical sampling from distributions

Phase 4 Features:
- Constraints: _mutex_, _requires_, _exclude_
- Presets: Reusable named configuration templates
- Iterator API: Memory-efficient lazy expansion
- Export utilities: DataFrame, diff, tree visualization

Usage:
    python Q24_generator_advanced.py
"""

import json
from itertools import islice

from nirs4all.pipeline.config.generator import (
    # Core API
    expand_spec,
    count_combinations,
    # Phase 3 strategies (implicit via expand_spec)
    # Phase 4: Iterator API
    expand_spec_iter,
    batch_iter,
    # Phase 4: Constraints
    apply_mutex_constraint,
    apply_requires_constraint,
    apply_exclude_constraint,
    apply_all_constraints,
    # Phase 4: Presets
    register_preset,
    get_preset,
    list_presets,
    clear_presets,
    has_preset,
    resolve_presets_recursive,
    export_presets,
    import_presets,
    # Phase 4: Export utilities
    to_dataframe,
    diff_configs,
    summarize_configs,
    print_expansion_tree,
    get_expansion_tree,
    format_config_table,
)


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def show_spec_result(name: str, spec, seed: int = None, max_show: int = 10):
    """Display specification and its expansion results."""
    print(f"--- {name} ---")
    print(f"Spec: {json.dumps(spec, indent=2, default=str)}")

    count = count_combinations(spec)
    results = expand_spec(spec, seed=seed)

    print(f"Count: {count}")
    if len(results) <= max_show:
        print(f"Results: {results}")
    else:
        print(f"First {max_show}: {results[:max_show]}")
        print(f"  ... ({len(results) - max_show} more)")
    print()


# ============================================================
# PART 1: Constraint System
# ============================================================
print_section("PART 1: Constraint System")

print("The constraint system allows filtering combinations based on rules:")
print("  _mutex_: Mutual exclusion - items cannot appear together")
print("  _requires_: Dependency - if A is selected, B must also be")
print("  _exclude_: Exclusion - specific combinations to remove")
print()

# Base example: all combinations
transforms = ["SNV", "MSC", "Detrend", "SavGol"]
base_spec = {"_or_": transforms, "pick": 2}
all_combos = expand_spec(base_spec)
print(f"All 2-combinations of {transforms}:")
print(f"  {all_combos}")
print()

# 1.1 Mutual Exclusion (_mutex_)
print("--- 1.1 Mutual Exclusion (_mutex_) ---")
print("SNV and MSC are similar transforms - don't use together:")
mutex_spec = {"_or_": transforms, "pick": 2, "_mutex_": [["SNV", "MSC"]]}
mutex_result = expand_spec(mutex_spec)
print(f"After _mutex_: {mutex_result}")
print("Removed: [SNV, MSC] combination")
print()

# 1.2 Dependency Requirements (_requires_)
print("--- 1.2 Dependency Requirements (_requires_) ---")
print("If Detrend is selected, SNV must also be selected:")
requires_spec = {"_or_": transforms, "pick": 2, "_requires_": [["Detrend", "SNV"]]}
requires_result = expand_spec(requires_spec)
print(f"After _requires_: {requires_result}")
print("Only combos with Detrend+SNV or without Detrend are valid")
print()

# 1.3 Specific Exclusions (_exclude_)
print("--- 1.3 Specific Exclusions (_exclude_) ---")
print("Exclude specific problematic combinations:")
exclude_spec = {
    "_or_": transforms,
    "pick": 2,
    "_exclude_": [["SNV", "MSC"], ["Detrend", "SavGol"]]
}
exclude_result = expand_spec(exclude_spec)
print(f"After _exclude_: {exclude_result}")
print()

# 1.4 Combined Constraints
print("--- 1.4 Combined Constraints ---")
print("Multiple constraint types can be combined:")
combined_spec = {
    "_or_": transforms,
    "pick": 2,
    "_mutex_": [["SNV", "MSC"]],  # Don't use together
    "_requires_": [["SavGol", "Detrend"]],  # SavGol needs Detrend
}
combined_result = expand_spec(combined_spec)
print(f"Combined constraints: {combined_result}")
print()

# 1.5 Direct Constraint Functions (API usage)
print("--- 1.5 Direct Constraint Functions ---")
combos = [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"], ["C", "D"]]
print(f"Starting combos: {combos}")

# Apply mutex directly
mutex_result = apply_mutex_constraint(combos, [["A", "B"]])
print(f"After apply_mutex_constraint([A,B]): {mutex_result}")

# Apply requires directly
requires_result = apply_requires_constraint(combos, [["A", "C"]])
print(f"After apply_requires_constraint([A,C]): {requires_result}")

# Apply all at once
all_result = apply_all_constraints(
    combos,
    mutex_groups=[["A", "B"]],
    requires_groups=[["C", "D"]],
    exclude_combos=[["A", "D"]]
)
print(f"After apply_all_constraints: {all_result}")
print()


# ============================================================
# PART 2: Preset System
# ============================================================
print_section("PART 2: Preset System")

# Clear any existing presets
clear_presets()

print("Presets allow defining reusable configuration templates.")
print()

# 2.1 Registering Presets
print("--- 2.1 Registering Presets ---")

register_preset(
    "spectral_preprocessing",
    {
        "_or_": [
            {"class": "SNV"},
            {"class": "MSC"},
            {"class": "Detrend", "order": 1},
            {"class": "SavitzkyGolay", "window": 11, "polyorder": 2}
        ],
        "pick": (1, 2)
    },
    description="Common NIRS spectral preprocessing transforms",
    tags=["preprocessing", "spectral"]
)

register_preset(
    "pls_hyperparams",
    {
        "_grid_": {
            "n_components": {"_range_": [2, 15]},
            "scale": [True, False]
        }
    },
    description="PLS regression hyperparameter grid",
    tags=["model", "pls", "hyperparameter"]
)

register_preset(
    "learning_rates",
    {"_log_range_": [0.0001, 0.1, 10]},
    description="Common learning rate search range",
    tags=["hyperparameter", "deep_learning"]
)

register_preset(
    "standard_scalers",
    {
        "_or_": [
            {"class": "StandardScaler"},
            {"class": "MinMaxScaler"},
            {"class": "RobustScaler"},
            None
        ]
    },
    description="Standard sklearn scalers",
    tags=["preprocessing", "sklearn"]
)

print(f"Registered presets: {list_presets()}")
print(f"Preprocessing presets: {list_presets(tags=['preprocessing'])}")
print()

# 2.2 Using Presets
print("--- 2.2 Using Presets ---")

config_with_presets = {
    "scaler": {"_preset_": "standard_scalers"},
    "preprocessing": {"_preset_": "spectral_preprocessing"},
    "model": {
        "class": "PLSRegression",
        "params": {"_preset_": "pls_hyperparams"}
    }
}

print("Config with presets:")
print(json.dumps(config_with_presets, indent=2))
print()

# Resolve presets
resolved_config = resolve_presets_recursive(config_with_presets)
print("After resolving presets:")
print(json.dumps(resolved_config, indent=2, default=str))
print()

print(f"Total combinations: {count_combinations(resolved_config)}")
print()

# 2.3 Getting Preset Info
print("--- 2.3 Getting Preset Info ---")
from nirs4all.pipeline.config.generator import get_preset_info

info = get_preset_info("pls_hyperparams")
print("Preset 'pls_hyperparams':")
print(f"  Description: {info['description']}")
print(f"  Tags: {info['tags']}")
print(f"  Spec: {info['spec']}")
print()

# 2.4 Export/Import Presets
print("--- 2.4 Export/Import Presets ---")
exported = export_presets()
print(f"Exported {len(exported)} presets")

# Clear and re-import
clear_presets()
print(f"After clear: {list_presets()}")

imported_count = import_presets(exported)
print(f"After import: {list_presets()} ({imported_count} imported)")
print()


# ============================================================
# PART 3: Iterator API for Large Spaces
# ============================================================
print_section("PART 3: Iterator API for Large Spaces")

print("expand_spec_iter() generates configurations lazily.")
print("Essential for large configuration spaces.")
print()

# 3.1 Basic Lazy Iteration
print("--- 3.1 Basic Lazy Iteration ---")
large_spec = {"_range_": [1, 1000]}
print(f"Spec: {large_spec}")
print(f"Total: {count_combinations(large_spec)} configurations")

# Get first 10 without generating all
first_10 = list(islice(expand_spec_iter(large_spec), 10))
print(f"First 10 (lazy): {first_10}")
print("Memory: Only 10 items in memory, not 1000")
print()

# 3.2 With Sampling
print("--- 3.2 Random Sampling with Reservoir Algorithm ---")
# Sample 5 from a large range
sampled = list(expand_spec_iter(large_spec, seed=42, sample_size=5))
print(f"Random sample of 5 (seed=42): {sampled}")

# Same seed = same sample
sampled2 = list(expand_spec_iter(large_spec, seed=42, sample_size=5))
print(f"Same seed again: {sampled2}")
print()

# 3.3 Batch Processing
print("--- 3.3 Batch Processing ---")
batch_spec = {"_range_": [1, 20]}
print(f"Processing {count_combinations(batch_spec)} items in batches of 7:")

for i, batch in enumerate(batch_iter(batch_spec, batch_size=7)):
    print(f"  Batch {i}: {batch} (size={len(batch)})")
print()

# 3.4 Very Large Grid (demonstration)
print("--- 3.4 Large Configuration Space ---")
large_grid = {
    "_grid_": {
        "param1": {"_range_": [1, 50]},
        "param2": {"_range_": [1, 50]},
        "param3": {"_or_": ["A", "B", "C"]}
    }
}
total = count_combinations(large_grid)
print(f"Grid spec would generate: {total:,} configurations")
print(f"Using expand_spec would use ~{total * 100 // 1024} KB of memory")
print("Using expand_spec_iter processes one at a time!")

# Get just first 5
first_5 = list(islice(expand_spec_iter(large_grid), 5))
print(f"First 5 configs: {first_5}")
print()


# ============================================================
# PART 4: Export and Visualization Utilities
# ============================================================
print_section("PART 4: Export and Visualization Utilities")

# Sample data for utilities
sample_spec = {
    "_grid_": {
        "model": ["PLS", "RF", "SVR"],
        "n_components": [5, 10, 15],
        "preprocessing": ["Standard", "MinMax"]
    }
}
configs = expand_spec(sample_spec)
print(f"Sample configs: {len(configs)} configurations")
print()

# 4.1 to_dataframe
print("--- 4.1 Convert to DataFrame ---")
try:
    df = to_dataframe(configs)
    print(df.to_string())
    print()
    print("Column types:", dict(df.dtypes))
except ImportError:
    print("(pandas not installed - skipping DataFrame demo)")
print()

# 4.2 summarize_configs
print("--- 4.2 Summarize Configurations ---")
summary = summarize_configs(configs)
print(f"Total configurations: {summary['count']}")
print("Per-key summary:")
for key, info in summary['keys'].items():
    print(f"  {key}: {info['unique_count']} unique values: {info['unique_values']}")
print()

# 4.3 diff_configs
print("--- 4.3 Configuration Diff ---")
c1, c2 = configs[0], configs[-1]
print(f"Config 1: {c1}")
print(f"Config 2: {c2}")
diff = diff_configs(c1, c2)
print(f"Differences: {diff}")
print()

# 4.4 format_config_table
print("--- 4.4 ASCII Table Format ---")
print(format_config_table(configs[:10], max_rows=10))
print()

# 4.5 Expansion Tree
print("--- 4.5 Expansion Tree Visualization ---")
tree_spec = {
    "preprocessing": {
        "_or_": ["SNV", "MSC", "Detrend"]
    },
    "model": {
        "_grid_": {
            "class": ["PLS", "RF"],
            "components": {"_range_": [2, 5]}
        }
    }
}
print("Spec:")
print(json.dumps(tree_spec, indent=2))
print()
print("Expansion Tree:")
print(print_expansion_tree(tree_spec))
print()

# Get tree as object
tree = get_expansion_tree(tree_spec)
print(f"Tree root: {tree.key}, type: {tree.node_type}, count: {tree.count}")
print(f"Children: {[c.key for c in tree.children]}")
print()


# ============================================================
# PART 5: Real-World Pipeline Example
# ============================================================
print_section("PART 5: Real-World Pipeline Configuration")

print("Combining all features for a realistic NIRS analysis pipeline:")
print()

# Clear and register fresh presets
clear_presets()

# Register domain-specific presets
register_preset(
    "nirs_spectral_transforms",
    {
        "_or_": [
            {"class": "SNV"},
            {"class": "MSC"},
            {"class": "Detrend", "order": {"_or_": [1, 2]}},
            {"class": "SavitzkyGolay", "window": {"_or_": [5, 11, 21]}, "polyorder": 2},
            {"class": "FirstDerivative"},
            {"class": "SecondDerivative"}
        ],
        "pick": (1, 3),
        "_mutex_": [
            ["FirstDerivative", "SecondDerivative"],  # Don't use both derivatives
            ["SNV", "MSC"]  # Similar methods
        ]
    },
    tags=["nirs", "preprocessing"]
)

register_preset(
    "pls_search",
    {
        "_grid_": {
            "class": ["PLSRegression"],
            "n_components": {"_range_": [2, 20, 2]}
        }
    },
    tags=["model", "pls"]
)

register_preset(
    "ensemble_models",
    {
        "_or_": [
            {"class": "RandomForestRegressor", "n_estimators": 100},
            {"class": "GradientBoostingRegressor", "n_estimators": 100},
            {"class": "XGBRegressor", "n_estimators": 100}
        ]
    },
    tags=["model", "ensemble"]
)

# Build complex pipeline config
pipeline_config = {
    "feature_augmentation": {"_preset_": "nirs_spectral_transforms"},
    "model": {
        "_chain_": [
            # Try PLS first (quick baseline)
            {"_preset_": "pls_search"},
            # Then ensemble methods
            {"_preset_": "ensemble_models"}
        ]
    },
    "cv": {
        "_or_": [
            {"class": "KFold", "n_splits": 5},
            {"class": "ShuffleSplit", "n_splits": 10, "test_size": 0.2}
        ]
    }
}

print("Pipeline Config with Presets:")
print(json.dumps(pipeline_config, indent=2))
print()

# Resolve and analyze
resolved = resolve_presets_recursive(pipeline_config)
print("Resolved config (truncated):")
print(json.dumps(resolved, indent=2, default=str)[:1000])
print("...")
print()

total_combos = count_combinations(resolved)
print(f"Total configurations to evaluate: {total_combos:,}")
print()

# Show expansion tree
print("Expansion Tree:")
print(print_expansion_tree(resolved, max_depth=2))
print()

# Get a sample of configs
print("Sample of 5 configurations:")
sample_configs = list(expand_spec_iter(resolved, seed=42, sample_size=5))
for i, cfg in enumerate(sample_configs):
    print(f"  Config {i+1}:")
    print(f"    Feature aug: {cfg.get('feature_augmentation', 'N/A')}")
    print(f"    Model: {str(cfg.get('model', 'N/A'))[:60]}...")
print()


# ============================================================
# Summary
# ============================================================
print_section("SUMMARY: Advanced Generator Features")

print("Phase 3 Keywords:")
print("  _log_range_: [0.001, 1, 4] -> [0.001, 0.01, 0.1, 1.0]")
print("  _grid_: Cartesian product of parameters")
print("  _zip_: Parallel pairing (positional)")
print("  _chain_: Sequential ordered configs")
print("  _sample_: Statistical sampling")
print()

print("Phase 4 Features:")
print("  Constraints:")
print("    _mutex_: [['A','B']] - A and B can't appear together")
print("    _requires_: [['A','B']] - If A, then B must be included")
print("    _exclude_: Exclude specific combinations")
print()
print("  Presets:")
print("    register_preset('name', spec, description, tags)")
print("    {'_preset_': 'name'} - Reference preset")
print("    resolve_presets_recursive(config) - Expand all presets")
print()
print("  Iterator API:")
print("    expand_spec_iter(spec) - Lazy iteration")
print("    batch_iter(spec, batch_size) - Batch processing")
print()
print("  Export Utilities:")
print("    to_dataframe(configs) - Pandas DataFrame")
print("    summarize_configs(configs) - Statistics")
print("    diff_configs(c1, c2) - Find differences")
print("    print_expansion_tree(spec) - Visual tree")
print()

print("Example completed successfully!")
