"""
Q26: Nested _or_ for Preprocessing Pipeline Combinations
=========================================================
This example demonstrates how to generate all combinations of preprocessing
steps organized by functional categories (scattering, smoothing, derivatives).

The Goal:
---------
Generate pipelines with ordered stages, each stage having multiple options:
- Stage 1 (Scattering): MSC, SNV, EMSC, LSNV, RSNV
- Stage 2 (Smoothing): SavGol, Gaussian, None
- Stage 3 (Derivative): Identity, 1st deriv, 2nd deriv, SavGol+deriv
- Stage 4 (Optional): Wavelets, etc.

Each generated pipeline is a list like:
    [MSC, Gaussian, FirstDeriv, Wavelet]
    [SNV, SavGol, Identity, None]
    etc.

Solution:
---------
Use a LIST containing _or_ nodes at each position. The generator automatically
computes the Cartesian product of all positions.

Key Insight:
    [{"_or_": [A, B]}, {"_or_": [X, Y]}]
    -> [[A, X], [A, Y], [B, X], [B, Y]]

This is different from _chain_ which preserves order without Cartesian product.

Usage:
    python Q26_nested_or_preprocessing.py
"""

from nirs4all.pipeline.config.generator import (
    expand_spec,
    count_combinations,
    expand_spec_iter,
    register_preset,
    resolve_presets_recursive,
    clear_presets,
    list_presets,
    to_dataframe,
    print_expansion_tree,
    summarize_configs,
    CARTESIAN_KEYWORD,  # New keyword for staged pipeline expansion
)


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def show_expansion(name: str, spec, max_show: int = 20):
    """Show expansion results."""
    count = count_combinations(spec)
    results = expand_spec(spec)
    print(f"--- {name} ---")
    print(f"Spec: {spec}")
    print(f"Count: {count}")

    if len(results) <= max_show:
        for i, r in enumerate(results, 1):
            print(f"  {i:3d}. {r}")
    else:
        for i, r in enumerate(results[:max_show // 2], 1):
            print(f"  {i:3d}. {r}")
        print(f"  ... ({count - max_show} more)")
        for i, r in enumerate(results[-3:], count - 2):
            print(f"  {i:3d}. {r}")
    print()


# ============================================================
# PART 1: Basic Nested _or_ for Sequential Stages
# ============================================================
print_section("PART 1: Basic Nested _or_ for Sequential Stages")

print("When you have a LIST with _or_ at each position,")
print("the generator computes Cartesian product of all positions.")
print()

# Simple example: 2 stages with 2 options each -> 4 combinations
simple_pipeline = [
    {"_or_": ["MSC", "SNV"]},  # Stage 1: scatter correction
    {"_or_": ["SavGol", "Gaussian"]}  # Stage 2: smoothing
]

show_expansion("Two stages, 2 options each (2×2=4)", simple_pipeline)

# Three stages
three_stage = [
    {"_or_": ["MSC", "SNV"]},  # Stage 1
    {"_or_": ["SavGol", "Gaussian"]},  # Stage 2
    {"_or_": ["Identity", "Deriv1", "Deriv2"]}  # Stage 3
]

show_expansion("Three stages (2×2×3=12)", three_stage)


# ============================================================
# PART 2: Full NIRS Preprocessing Pipeline
# ============================================================
print_section("PART 2: Full NIRS Preprocessing Pipeline")

print("Defining stages for a complete NIRS preprocessing pipeline:")
print("  Stage 1 - Scattering: MSC, SNV, EMSC, LSNV, RSNV")
print("  Stage 2 - Smoothing: SavGol, Gaussian, None")
print("  Stage 3 - Derivative: Identity, 1st, 2nd, SavGol+deriv")
print()

# Full pipeline with all stages
full_preprocessing_pipeline = [
    # Stage 1: Scattering correction (5 options)
    {"_or_": [
        {"class": "MSC"},
        {"class": "SNV"},
        {"class": "EMSC"},
        {"class": "LSNV"},
        {"class": "RSNV"},
    ]},

    # Stage 2: Smoothing (3 options, including None)
    {"_or_": [
        {"class": "SavitzkyGolay", "window": 11, "polyorder": 2},
        {"class": "GaussianFilter", "sigma": 2},
        None,  # No smoothing
    ]},

    # Stage 3: Derivative (4 options)
    {"_or_": [
        None,  # Identity / no derivative
        {"class": "FirstDerivative"},
        {"class": "SecondDerivative"},
        {"class": "SavitzkyGolay", "window": 11, "deriv": 1},
    ]},
]

total = count_combinations(full_preprocessing_pipeline)
print(f"Total preprocessing combinations: {total} (5×3×4=60)")
print()

# Show first few
print("First 10 combinations:")
results = expand_spec(full_preprocessing_pipeline)
for i, r in enumerate(results[:10], 1):
    print(f"  {i:2d}. {r}")
print(f"  ... and {total - 10} more")
print()


# ============================================================
# PART 3: With Optional 4th Stage
# ============================================================
print_section("PART 3: With Optional 4th Stage (Alternative Transforms)")

print("Adding an optional 4th stage for alternative transforms.")
print()

extended_pipeline = [
    # Stage 1: Scattering
    {"_or_": [
        {"class": "MSC"},
        {"class": "SNV"},
        {"class": "EMSC"},
    ]},

    # Stage 2: Smoothing
    {"_or_": [
        {"class": "SavitzkyGolay", "window": 11, "polyorder": 2},
        {"class": "GaussianFilter", "sigma": 2},
    ]},

    # Stage 3: Derivative
    {"_or_": [
        None,
        {"class": "FirstDerivative"},
    ]},

    # Stage 4: Optional alternative transforms
    {"_or_": [
        None,  # No additional transform
        {"class": "Wavelet", "wavelet": "db4", "level": 3},
        {"class": "FourierTransform", "n_components": 50},
    ]},
]

total_extended = count_combinations(extended_pipeline)
print(f"Extended pipeline combinations: {total_extended} (3×2×2×3=36)")

show_expansion("Extended 4-stage pipeline", extended_pipeline, max_show=15)


# ============================================================
# PART 4: Using Presets for Cleaner Syntax
# ============================================================
print_section("PART 4: Using Presets for Cleaner Syntax")

# Clear presets
clear_presets()

# Define stage presets
register_preset(
    "scatter_stage",
    {"_or_": [
        {"class": "MSC"},
        {"class": "SNV"},
        {"class": "EMSC"},
        {"class": "LSNV"},
        {"class": "RSNV"},
    ]},
    description="Scattering correction options",
    tags=["preprocessing", "scatter"]
)

register_preset(
    "smooth_stage",
    {"_or_": [
        {"class": "SavitzkyGolay", "window": {"_or_": [5, 11, 21]}, "polyorder": 2},
        {"class": "GaussianFilter", "sigma": {"_or_": [1, 2, 3]}},
        None,
    ]},
    description="Smoothing options with parameter variations",
    tags=["preprocessing", "smooth"]
)

register_preset(
    "deriv_stage",
    {"_or_": [
        None,
        {"class": "FirstDerivative"},
        {"class": "SecondDerivative"},
        {"class": "SavitzkyGolay", "window": 11, "deriv": {"_or_": [1, 2]}},
    ]},
    description="Derivative options",
    tags=["preprocessing", "derivative"]
)

register_preset(
    "alternative_stage",
    {"_or_": [
        None,
        {"class": "Wavelet", "wavelet": {"_or_": ["db4", "sym4", "coif4"]}, "level": 3},
    ]},
    description="Optional alternative transforms",
    tags=["preprocessing", "alternative"]
)

print(f"Registered presets: {list_presets()}")
print()

# Use presets to define pipeline
preset_pipeline = [
    {"_preset_": "scatter_stage"},
    {"_preset_": "smooth_stage"},
    {"_preset_": "deriv_stage"},
]

# Resolve presets first
resolved = resolve_presets_recursive(preset_pipeline)
print("Resolved pipeline structure:")
print(f"  Stage 1: {len(expand_spec(resolved[0]))} scatter options")
print(f"  Stage 2: {len(expand_spec(resolved[1]))} smooth options (with param variations)")
print(f"  Stage 3: {len(expand_spec(resolved[2]))} derivative options")

total_preset = count_combinations(resolved)
print(f"\nTotal combinations: {total_preset}")
print()


# ============================================================
# PART 5: Nested _or_ Inside Dict Structure
# ============================================================
print_section("PART 5: Nested _or_ Inside Dict Structure")

print("The pipeline list can be inside a dict, useful for complete configs.")
print()

complete_config = {
    "name": "NIRS_preprocessing_search",

    # The preprocessing is a list of stages
    "preprocessing": [
        {"_or_": [
            {"class": "MSC"},
            {"class": "SNV"},
        ]},
        {"_or_": [
            {"class": "SavitzkyGolay", "window": 11},
            None,
        ]},
        {"_or_": [
            None,
            {"class": "FirstDerivative"},
        ]},
    ],

    # Model with its own options
    "model": {
        "class": "PLSRegression",
        "n_components": {"_or_": [5, 10, 15]},
    },

    # Fixed settings
    "cv_folds": 5,
}

total_config = count_combinations(complete_config)
print(f"Total complete configurations: {total_config} (2×2×2×3=24)")
print()

# Show a few expanded configs
expanded_configs = expand_spec(complete_config)
print("Sample expanded configurations:")
for i, cfg in enumerate(expanded_configs[:5], 1):
    pp = [s.get('class', 'None') if isinstance(s, dict) else 'None' for s in cfg['preprocessing']]
    print(f"  {i}. preprocessing={pp}, n_components={cfg['model']['n_components']}")
print(f"  ... and {len(expanded_configs) - 5} more")
print()


# ============================================================
# PART 6: Constrained Stage Combinations
# ============================================================
print_section("PART 6: Constrained Stage Combinations with _mutex_")

print("You can apply constraints to filter out invalid combinations.")
print("But note: _mutex_ works on items within a single _or_, not across stages.")
print()
print("For cross-stage constraints, use post-filtering or structured approach.")
print()

# Example: Within one stage, some options are mutually exclusive
# If using EMSC, don't also try LSNV (they're similar)
constrained_scatter = {
    "_or_": ["MSC", "SNV", "EMSC", "LSNV", "RSNV"],
    # This would work if picking multiple from same stage:
    # "_mutex_": [["EMSC", "LSNV"], ["EMSC", "RSNV"]]
}

print("For single-choice stages (pick 1), mutual exclusion is automatic.")
print("Each combination uses exactly one option per stage.")
print()


# ============================================================
# PART 7: Efficient Iteration for Large Spaces
# ============================================================
print_section("PART 7: Efficient Iteration for Large Spaces")

# Define a large preprocessing space
large_pipeline = [
    {"_or_": [{"class": f"Scatter{i}"} for i in range(10)]},  # 10 scatter options
    {"_or_": [{"class": f"Smooth{i}"} for i in range(5)]},   # 5 smooth options
    {"_or_": [{"class": f"Deriv{i}"} for i in range(4)]},    # 4 deriv options
    {"_or_": [{"class": f"Alt{i}"} for i in range(3)]},      # 3 alternative options
]

large_total = count_combinations(large_pipeline)
print(f"Large pipeline space: {large_total} combinations (10×5×4×3=600)")
print()

# Use iterator for memory-efficient processing
print("Using expand_spec_iter for lazy evaluation:")
from itertools import islice

first_10 = list(islice(expand_spec_iter(large_pipeline), 10))
print(f"First 10 (lazy): {len(first_10)} configs loaded")
for i, cfg in enumerate(first_10[:5], 1):
    classes = [s.get('class', 'None') if s else 'None' for s in cfg]
    print(f"  {i}. {classes}")
print()

# Random sampling from large space
print("Random sample of 5 from 600 combinations (seed=42):")
sampled = list(expand_spec_iter(large_pipeline, seed=42, sample_size=5))
for i, cfg in enumerate(sampled, 1):
    classes = [s.get('class', 'None') if s else 'None' for s in cfg]
    print(f"  {i}. {classes}")
print()


# ============================================================
# PART 8: Real-World NIRS Preprocessing Example
# ============================================================
print_section("PART 8: Real-World NIRS Preprocessing Example")

print("A realistic NIRS preprocessing search space:")
print()

# Realistic preprocessing transforms with actual parameters
nirs_preprocessing = {
    "name": "NIRS_Preprocessing_Search",
    "dataset": "wheat_protein",

    "preprocessing": [
        # Stage 1: Scatter correction
        {"_or_": [
            {"class": "StandardNormalVariate"},
            {"class": "MultiplicativeScatterCorrection"},
            {"class": "ExtendedMSC", "reference": "mean"},
            {"class": "LocalSNV", "window": {"_or_": [5, 11, 21]}},
            {"class": "RobustSNV", "quantile": 0.95},
        ]},

        # Stage 2: Smoothing/Filtering
        {"_or_": [
            None,  # No smoothing
            {"class": "SavitzkyGolay", "window_length": {"_or_": [5, 11, 17]}, "polyorder": 2, "deriv": 0},
            {"class": "GaussianFilter", "sigma": {"_or_": [1.0, 2.0]}},
            {"class": "MovingAverage", "window": {"_or_": [3, 5, 7]}},
        ]},

        # Stage 3: Derivative
        {"_or_": [
            None,  # No derivative (identity)
            {"class": "SavitzkyGolay", "window_length": 11, "polyorder": 2, "deriv": 1},
            {"class": "SavitzkyGolay", "window_length": 11, "polyorder": 3, "deriv": 2},
            {"class": "GapDerivative", "gap": {"_or_": [5, 10, 15]}},
        ]},
    ],

    "model": {
        "_or_": [
            {"class": "PLSRegression", "n_components": {"_range_": [3, 15]}},
            {"class": "RidgeRegression", "alpha": {"_log_range_": [0.001, 100, 7]}},
        ]
    },

    "cv": {"class": "KFold", "n_splits": 5, "shuffle": True},
    "metrics": ["R2", "RMSE", "RPD"],
}

nirs_total = count_combinations(nirs_preprocessing)
print(f"Total NIRS configurations to evaluate: {nirs_total:,}")
print()

# Summarize the space
print("Configuration space breakdown:")
print("  Scatter options: 5 base × some with params")
print("  Smoothing options: 4 base × some with params")
print("  Derivative options: 4 base × some with params")
print("  Model options: PLS (13 components) + Ridge (7 alphas)")
print()

# Show first few
print("Sample configurations (first 5):")
sample = list(islice(expand_spec_iter(nirs_preprocessing), 5))
for i, cfg in enumerate(sample, 1):
    pp_classes = []
    for step in cfg['preprocessing']:
        if step is None:
            pp_classes.append("None")
        else:
            pp_classes.append(step.get('class', 'Unknown'))
    model = cfg['model'].get('class', 'Unknown')
    print(f"  {i}. {' > '.join(pp_classes)} -> {model}")
print()


# ============================================================
# PART 9: Feature Augmentation with Multiple Pipelines
# ============================================================
print_section("PART 9: Feature Augmentation with Multiple Pipelines")

print("Use case: Pick 1-3 COMPLETE preprocessing pipelines for feature_augmentation")
print("Each pipeline = [Scatter, Smooth, Derivative] in order")
print()
print("Solution: Two-step expansion")
print("  Step 1: Generate all pipelines (Cartesian product of stages)")
print("  Step 2: Pick 1-3 from those pipelines")
print()

# Step 1: Define the pipeline stages and generate all combinations
pipeline_stages = [
    {"_or_": ["MSC", "SNV", "EMSC"]},      # Stage 1: Scatter (3 options)
    {"_or_": ["SavGol", "Gaussian", None]},  # Stage 2: Smooth (3 options)
    {"_or_": [None, "Deriv1", "Deriv2"]},   # Stage 3: Derivative (3 options)
]

# This generates 3×3×3 = 27 complete pipelines
all_pipelines = expand_spec(pipeline_stages)
print(f"Step 1: Generated {len(all_pipelines)} complete pipelines (3×3×3)")
print("Sample pipelines:")
for i, p in enumerate(all_pipelines[:5], 1):
    print(f"  {i}. {p}")
print("  ...")
print()

# Step 2: Use these pipelines as choices for feature_augmentation
feature_augmentation_spec = {
    "_or_": all_pipelines,  # Each pipeline is now a choice
    "pick": (1, 3),         # Pick 1-3 complete pipelines
    "count": 15,            # Limit to 15 configurations
}

print("Step 2: Pick 1-3 pipelines for feature_augmentation")
augmentation_configs = expand_spec(feature_augmentation_spec, seed=42)
print(f"Generated {len(augmentation_configs)} feature_augmentation configurations:")
for i, cfg in enumerate(augmentation_configs, 1):
    if len(cfg) == 1:
        print(f"  {i}. Single pipeline: {cfg[0]}")
    else:
        print(f"  {i}. {len(cfg)} pipelines: {cfg}")
print()

# Complete example with model
print("--- Complete Configuration Example ---")
print()

# In practice, you'd structure it like this:
complete_spec = {
    "name": "Multi-Pipeline Feature Augmentation",

    # The feature_augmentation uses the pre-generated pipelines
    "feature_augmentation": {
        "_or_": all_pipelines,
        "pick": (1, 2),
        "count": 5,
    },

    "model": {
        "class": "PLSRegression",
        "n_components": {"_or_": [5, 10, 15]},
    },
}

print("Complete config with feature_augmentation:")
complete_configs = expand_spec(complete_spec, seed=123)
print(f"Total: {len(complete_configs)} configurations")
for i, cfg in enumerate(complete_configs[:5], 1):
    fa = cfg["feature_augmentation"]
    n = cfg["model"]["n_components"]
    if isinstance(fa, list) and all(isinstance(x, list) for x in fa):
        print(f"  {i}. {len(fa)} pipelines, n_components={n}")
    else:
        print(f"  {i}. 1 pipeline: {fa}, n_components={n}")
print()

# Helper function for cleaner syntax
print("--- Helper Function for Cleaner Syntax ---")
print()


def make_pipeline_choices(stages_spec, pick_range=(1, 2), count=None, seed=None):
    """Generate pipeline choices from stage specifications.

    Args:
        stages_spec: List of stage specifications with _or_ at each position.
        pick_range: How many pipelines to pick (int or tuple).
        count: Limit number of results.
        seed: Random seed.

    Returns:
        Expanded list of pipeline combinations.
    """
    pipelines = expand_spec(stages_spec)
    pick_spec = {
        "_or_": pipelines,
        "pick": pick_range,
    }
    if count:
        pick_spec["count"] = count
    return expand_spec(pick_spec, seed=seed)


# Usage
stages = [
    {"_or_": ["MSC", "SNV", "EMSC", "LSNV", "RSNV"]},  # 5 scatter
    {"_or_": ["SavGol", "Gaussian", None]},              # 3 smooth
    {"_or_": [None, "Deriv1", "Deriv2", "SavGol_D1"]},   # 4 derivative
]

print("Using helper function:")
print(f"  Stages: {len(expand_spec(stages))} total pipelines (5×3×4=60)")
augmented = make_pipeline_choices(stages, pick_range=(1, 3), count=8, seed=42)
print(f"  Picked {len(augmented)} configurations with 1-3 pipelines each:")
for i, cfg in enumerate(augmented, 1):
    if isinstance(cfg, list) and isinstance(cfg[0], list):
        print(f"    {i}. {len(cfg)} pipelines")
    else:
        print(f"    {i}. 1 pipeline: {cfg}")
print()


# ============================================================
# PART 10: The _cartesian_ Keyword (Single-Spec Solution)
# ============================================================
print_section("PART 10: The _cartesian_ Keyword (Single-Spec Solution)")

print("The _cartesian_ keyword combines both steps in a single specification!")
print(f"Keyword: {CARTESIAN_KEYWORD}")
print()
print("Syntax:")
print("  {")
print('    "_cartesian_": [stage1, stage2, stage3],  # Stages with _or_')
print('    "pick": (1, 3),  # Pick 1-3 complete pipelines')
print('    "count": N       # Limit results')
print("  }")
print()

# Example 1: Basic _cartesian_ with pick
print("--- Example 1: Basic _cartesian_ with pick ---")
cartesian_spec = {
    "_cartesian_": [
        {"_or_": ["MSC", "SNV", "EMSC"]},       # Stage 1: Scatter
        {"_or_": ["SavGol", "Gaussian", None]},  # Stage 2: Smooth
        {"_or_": [None, "Deriv1", "Deriv2"]},   # Stage 3: Derivative
    ],
    "pick": 2,  # Pick 2 complete pipelines
    "count": 10,
}

print(f"Spec: {cartesian_spec}")
print("Total pipelines from stages: 3×3×3 = 27")
print("Picking 2 from 27: C(27,2) = 351 combinations")
print("With count=10: limited to 10")
print()

cartesian_results = expand_spec(cartesian_spec, seed=42)
print(f"Results ({len(cartesian_results)}):")
for idx, result in enumerate(cartesian_results, 1):
    print(f"  {idx}. {result}")
print()

# Example 2: Pick range (1-3 pipelines)
print("--- Example 2: Pick range (1-3 pipelines) ---")
cartesian_range_spec = {
    "_cartesian_": [
        {"_or_": ["MSC", "SNV"]},      # 2 scatter options
        {"_or_": ["SavGol", None]},     # 2 smooth options
    ],
    "pick": (1, 2),  # Pick 1 or 2 pipelines
}

print("Stages: 2×2 = 4 total pipelines")
range_count = count_combinations(cartesian_range_spec)
print(f"Pick 1-2: C(4,1) + C(4,2) = 4 + 6 = {range_count}")

range_results = expand_spec(cartesian_range_spec)
print(f"Results ({len(range_results)}):")
for idx, result in enumerate(range_results, 1):
    pipelines = result if isinstance(result[0], list) else [result]
    print(f"  {idx}. {len(pipelines)} pipeline(s): {result}")
print()

# Example 3: Full preprocessing with _cartesian_
print("--- Example 3: Full NIRS Preprocessing with _cartesian_ ---")
nirs_cartesian = {
    "_cartesian_": [
        # Stage 1: Scatter correction
        {"_or_": [
            {"class": "MSC"},
            {"class": "SNV"},
            {"class": "EMSC"},
            {"class": "LSNV"},
            {"class": "RSNV"},
        ]},
        # Stage 2: Smoothing
        {"_or_": [
            {"class": "SavitzkyGolay", "window": 11, "polyorder": 2},
            {"class": "GaussianFilter", "sigma": 2},
            None,
        ]},
        # Stage 3: Derivative
        {"_or_": [
            None,
            {"class": "FirstDerivative"},
            {"class": "SecondDerivative"},
            {"class": "SavitzkyGolay", "window": 11, "deriv": 1},
        ]},
    ],
    "pick": (1, 2),   # Pick 1-2 preprocessing pipelines
    "count": 15,      # Limit results
}

total_pipelines = 5 * 3 * 4  # 60 pipelines
print(f"Total pipelines: 5×3×4 = {total_pipelines}")
nirs_count = count_combinations(nirs_cartesian)
print(f"Pick 1-2 with count=15: {nirs_count}")

nirs_results = expand_spec(nirs_cartesian, seed=42)
print(f"\nSample results ({len(nirs_results)}):")
for idx, result in enumerate(nirs_results[:8], 1):
    if isinstance(result, list) and isinstance(result[0], list):
        # Multiple pipelines
        print(f"  {idx}. {len(result)} pipelines:")
        for p_idx, pipeline in enumerate(result, 1):
            classes = [s.get('class', 'None') if s else 'None' for s in pipeline]
            print(f"       {p_idx}. {' > '.join(classes)}")
    else:
        # Single pipeline (result is the pipeline)
        classes = [s.get('class', 'None') if s else 'None' for s in result]
        print(f"  {idx}. 1 pipeline: {' > '.join(classes)}")
print()

# Example 4: Using _cartesian_ in complete config
print("--- Example 4: Complete Config with _cartesian_ ---")
complete_cartesian_config = {
    "name": "NIRS_Cartesian_Search",

    "feature_augmentation": {
        "_cartesian_": [
            {"_or_": ["MSC", "SNV", "EMSC"]},
            {"_or_": ["SavGol", None]},
            {"_or_": [None, "Deriv1"]},
        ],
        "pick": (1, 2),
        "count": 5,
    },

    "model": {
        "_or_": [
            {"class": "PLSRegression", "n_components": 10},
            {"class": "RidgeRegression", "alpha": 1.0},
        ]
    },
}

config_count = count_combinations(complete_cartesian_config)
print(f"Total configurations: {config_count}")

config_results = expand_spec(complete_cartesian_config, seed=42)
print(f"Sample ({len(config_results[:5])}):")
for idx, cfg in enumerate(config_results[:5], 1):
    fa = cfg["feature_augmentation"]
    model = cfg["model"]["class"]
    if isinstance(fa, list) and isinstance(fa[0], list):
        print(f"  {idx}. {len(fa)} pipelines + {model}")
    else:
        print(f"  {idx}. 1 pipeline + {model}")
print()

# Comparison: _cartesian_ vs two-step
print("--- Comparison: _cartesian_ vs Two-Step Approach ---")
print()
print("TWO-STEP (PART 9):")
print("  step1 = expand_spec([stage1, stage2, stage3])")
print("  step2 = expand_spec({'_or_': step1, 'pick': 2})")
print()
print("SINGLE-SPEC with _cartesian_ (PART 10):")
print("  expand_spec({'_cartesian_': [stage1, stage2, stage3], 'pick': 2})")
print()
print("Both produce the same results!")
print("_cartesian_ is cleaner for declarative configs (YAML/JSON).")
print()


# ============================================================
# Summary
# ============================================================
print_section("SUMMARY: Nested _or_ for Pipeline Combinations")

print("""
Key Concepts:
=============

1. LIST with _or_ at each position = Cartesian product of all stages
   [{"_or_": [A, B]}, {"_or_": [X, Y]}] -> [[A,X], [A,Y], [B,X], [B,Y]]

2. This is the SOLUTION for your requirement:
   - Stage 1 (Scatter): {"_or_": [MSC, SNV, EMSC, LSNV, RSNV]}
   - Stage 2 (Smooth):  {"_or_": [SavGol, Gaussian, None]}
   - Stage 3 (Deriv):   {"_or_": [None, 1st, 2nd, SavGol+deriv]}
   - Stage 4 (Alt):     {"_or_": [None, Wavelet, FFT]}

3. Nested parameters within stages:
   {"class": "SavGol", "window": {"_or_": [5, 11, 21]}}
   This adds parameter variations to the Cartesian product.

4. Use presets for reusability:
   register_preset("scatter_stage", {"_or_": [...]})

5. For large spaces, use expand_spec_iter() for lazy evaluation.

6. _chain_ is DIFFERENT: it preserves order but doesn't do Cartesian product.
   _chain_ is for sequential configs, not for "all combinations of stages".

7. For MULTIPLE PIPELINES in feature_augmentation (pick 1-3 complete pipelines):

   TWO-STEP approach:
     Step 1: all_pipelines = expand_spec([stage1, stage2, stage3])
     Step 2: {"_or_": all_pipelines, "pick": (1, 3), "count": N}

   Or use the helper: make_pipeline_choices(stages, pick_range, count)

8. _cartesian_ keyword for SINGLE-SPEC solution:
   {
     "_cartesian_": [stage1, stage2, stage3],  # Stages with _or_
     "pick": (1, 3),  # Pick 1-3 complete pipelines
     "count": N       # Limit results
   }

   This is the cleanest approach for declarative configs (YAML/JSON).
""")


if __name__ == "__main__":
    print("\n✓ All examples completed successfully!")
