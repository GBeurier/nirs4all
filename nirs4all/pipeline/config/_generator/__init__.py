"""Generator package for pipeline configuration expansion.

This package provides functionality for expanding pipeline configuration
specifications into concrete pipeline variants. It handles combinatorial
keywords (_or_, _range_, size, count, pick, arrange) and generates all
possible combinations.

This is an internal package. The main API is exposed through the parent
generator.py module.

Phase 2 Keywords:
-----------------
- _or_: Choice between alternatives
- _range_: Numeric sequence generation
- size: Number of items to select from choices (legacy, uses combinations)
- pick: Unordered selection (combinations) - explicit intent
- arrange: Ordered arrangement (permutations) - explicit intent
- count: Limit number of generated variants

Phase 3 Keywords:
-----------------
- _log_range_: Logarithmic sequence generation
- _grid_: Grid search style Cartesian product
- _zip_: Parallel iteration (like Python's zip)
- _chain_: Sequential ordered choices
- _sample_: Statistical sampling (uniform, log-uniform, normal)
- _tags_: Configuration tagging for filtering
- _metadata_: Arbitrary metadata attachment
- _mutex_: Mutual exclusion constraints
- _requires_: Dependency requirements
- _exclude_: Exclusion rules

Phase 4 Features:
-----------------
- expand_spec_iter: Lazy iterator-based expansion for large spaces
- Constraint system: _mutex_, _requires_, _exclude_ filtering
- Preset system: Named configuration templates via _preset_
- Export utilities: DataFrame conversion, diff, tree visualization

Architecture (Phase 4):
-----------------------
- core.py: Main expansion/counting logic using strategy dispatch
- iterator.py: Lazy expansion with expand_spec_iter
- constraints.py: Constraint evaluation for filtering
- presets.py: Preset registry and resolution
- strategies/: Strategy pattern implementation for node types
    - base.py: ExpansionStrategy ABC
    - registry.py: Strategy registration and dispatch
    - range_strategy.py: Handles _range_ nodes
    - or_strategy.py: Handles _or_ nodes with pick/arrange/constraints
    - log_range_strategy.py: Handles _log_range_ nodes
    - grid_strategy.py: Handles _grid_ nodes
    - zip_strategy.py: Handles _zip_ nodes
    - chain_strategy.py: Handles _chain_ nodes
    - sample_strategy.py: Handles _sample_ nodes
- validators/: Configuration validation
    - schema.py: Schema and specification validation
- keywords.py: Keyword constants and detection utilities
- utils/: Helper functions (sampling, combinatorics, export)
    - export.py: DataFrame, diff, tree visualization (Phase 4)
"""

# Import core API
from .core import expand_spec, count_combinations

# Import iterator API (Phase 4)
from .iterator import (
    expand_spec_iter,
    batch_iter,
    iter_with_progress,
)

# Import from keywords module
from .keywords import (
    # Constants - Core
    OR_KEYWORD,
    RANGE_KEYWORD,
    LOG_RANGE_KEYWORD,
    GRID_KEYWORD,
    ZIP_KEYWORD,
    CHAIN_KEYWORD,
    SAMPLE_KEYWORD,
    # Constants - Modifiers
    SIZE_KEYWORD,
    COUNT_KEYWORD,
    SEED_KEYWORD,
    WEIGHTS_KEYWORD,
    # Constants - Selection
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    # Constants - Metadata
    TAGS_KEYWORD,
    METADATA_KEYWORD,
    # Constants - Constraints
    MUTEX_KEYWORD,
    REQUIRES_KEYWORD,
    DEPENDS_ON_KEYWORD,
    EXCLUDE_KEYWORD,
    # Keyword groups
    GENERATION_KEYWORDS,
    SELECTION_KEYWORDS,
    MODIFIER_KEYWORDS,
    METADATA_KEYWORDS,
    CONSTRAINT_KEYWORDS,
    ALL_KEYWORDS,
    PURE_OR_KEYS,
    PURE_RANGE_KEYS,
    PURE_LOG_RANGE_KEYS,
    PURE_GRID_KEYS,
    PURE_ZIP_KEYS,
    PURE_CHAIN_KEYS,
    PURE_SAMPLE_KEYS,
    # Functions - Detection
    is_generator_node,
    is_pure_or_node,
    is_pure_range_node,
    is_pure_log_range_node,
    is_pure_grid_node,
    is_pure_zip_node,
    is_pure_chain_node,
    is_pure_sample_node,
    has_or_keyword,
    has_range_keyword,
    has_log_range_keyword,
    has_grid_keyword,
    has_zip_keyword,
    has_chain_keyword,
    has_sample_keyword,
    # Functions - Extraction
    extract_modifiers,
    extract_base_node,
    extract_or_choices,
    extract_range_spec,
    extract_tags,
    extract_metadata,
    extract_constraints,
)

# Import utilities
from .utils import (
    sample_with_seed,
    shuffle_with_seed,
    random_choice_with_seed,
    generate_combinations,
    generate_combinations_range,
    generate_permutations,
    generate_cartesian_product,
    count_combinations as count_combs,  # Renamed to avoid conflict
    count_combinations_range,
    count_permutations,
    count_permutations_range,
    normalize_size_spec,
    is_nested_size_spec,
    expand_combination_cartesian,
)

__all__ = [
    # Core API
    "expand_spec",
    "count_combinations",
    # Iterator API (Phase 4)
    "expand_spec_iter",
    "batch_iter",
    "iter_with_progress",
    # Keywords - Core
    "OR_KEYWORD",
    "RANGE_KEYWORD",
    "LOG_RANGE_KEYWORD",
    "GRID_KEYWORD",
    "ZIP_KEYWORD",
    "CHAIN_KEYWORD",
    "SAMPLE_KEYWORD",
    # Keywords - Modifiers
    "SIZE_KEYWORD",
    "COUNT_KEYWORD",
    "SEED_KEYWORD",
    "WEIGHTS_KEYWORD",
    # Keywords - Selection
    "PICK_KEYWORD",
    "ARRANGE_KEYWORD",
    "THEN_PICK_KEYWORD",
    "THEN_ARRANGE_KEYWORD",
    # Keywords - Metadata
    "TAGS_KEYWORD",
    "METADATA_KEYWORD",
    # Keywords - Constraints
    "MUTEX_KEYWORD",
    "REQUIRES_KEYWORD",
    "DEPENDS_ON_KEYWORD",
    "EXCLUDE_KEYWORD",
    # Keyword groups
    "GENERATION_KEYWORDS",
    "SELECTION_KEYWORDS",
    "MODIFIER_KEYWORDS",
    "METADATA_KEYWORDS",
    "CONSTRAINT_KEYWORDS",
    "ALL_KEYWORDS",
    "PURE_OR_KEYS",
    "PURE_RANGE_KEYS",
    "PURE_LOG_RANGE_KEYS",
    "PURE_GRID_KEYS",
    "PURE_ZIP_KEYS",
    "PURE_CHAIN_KEYS",
    "PURE_SAMPLE_KEYS",
    # Keywords - Detection Functions
    "is_generator_node",
    "is_pure_or_node",
    "is_pure_range_node",
    "is_pure_log_range_node",
    "is_pure_grid_node",
    "is_pure_zip_node",
    "is_pure_chain_node",
    "is_pure_sample_node",
    "has_or_keyword",
    "has_range_keyword",
    "has_log_range_keyword",
    "has_grid_keyword",
    "has_zip_keyword",
    "has_chain_keyword",
    "has_sample_keyword",
    # Keywords - Extraction Functions
    "extract_modifiers",
    "extract_base_node",
    "extract_or_choices",
    "extract_range_spec",
    "extract_tags",
    "extract_metadata",
    "extract_constraints",
    # Utilities - Sampling
    "sample_with_seed",
    "shuffle_with_seed",
    "random_choice_with_seed",
    # Utilities - Combinatorics
    "generate_combinations",
    "generate_combinations_range",
    "generate_permutations",
    "generate_cartesian_product",
    "count_combs",  # Renamed from count_combinations
    "count_combinations_range",
    "count_permutations",
    "count_permutations_range",
    "normalize_size_spec",
    "is_nested_size_spec",
    "expand_combination_cartesian",
]

# Import strategy-related exports
from .strategies import (
    ExpansionStrategy,
    get_strategy,
    register_strategy,
    # Phase 2 strategies
    RangeStrategy,
    OrStrategy,
    # Phase 3 strategies
    LogRangeStrategy,
    GridStrategy,
    ZipStrategy,
    ChainStrategy,
    SampleStrategy,
)

__all__ += [
    # Strategies
    "ExpansionStrategy",
    "get_strategy",
    "register_strategy",
    # Phase 2 strategies
    "RangeStrategy",
    "OrStrategy",
    # Phase 3 strategies
    "LogRangeStrategy",
    "GridStrategy",
    "ZipStrategy",
    "ChainStrategy",
    "SampleStrategy",
]

# Import validators
from .validators import (
    ValidationError,
    ValidationResult,
    ValidationSeverity,
    validate_spec,
    validate_config,
    validate_expanded_configs,
)

__all__ += [
    # Validators
    "ValidationError",
    "ValidationResult",
    "ValidationSeverity",
    "validate_spec",
    "validate_config",
    "validate_expanded_configs",
]

# Import constraints (Phase 4)
from .constraints import (
    apply_mutex_constraint,
    apply_requires_constraint,
    apply_exclude_constraint,
    apply_all_constraints,
    parse_constraints,
    validate_constraints,
)

__all__ += [
    # Constraints (Phase 4)
    "apply_mutex_constraint",
    "apply_requires_constraint",
    "apply_exclude_constraint",
    "apply_all_constraints",
    "parse_constraints",
    "validate_constraints",
]

# Import presets (Phase 4)
from .presets import (
    PRESET_KEYWORD,
    register_preset,
    unregister_preset,
    get_preset,
    get_preset_info,
    list_presets,
    clear_presets,
    has_preset,
    is_preset_reference,
    resolve_preset,
    resolve_presets_recursive,
    export_presets,
    import_presets,
    register_builtin_presets,
)

__all__ += [
    # Presets (Phase 4)
    "PRESET_KEYWORD",
    "register_preset",
    "unregister_preset",
    "get_preset",
    "get_preset_info",
    "list_presets",
    "clear_presets",
    "has_preset",
    "is_preset_reference",
    "resolve_preset",
    "resolve_presets_recursive",
    "export_presets",
    "import_presets",
    "register_builtin_presets",
]

# Import export utilities (Phase 4)
from .utils.export import (
    to_dataframe,
    diff_configs,
    summarize_configs,
    get_expansion_tree,
    print_expansion_tree,
    format_config_table,
    ExpansionTreeNode,
)

__all__ += [
    # Export utilities (Phase 4)
    "to_dataframe",
    "diff_configs",
    "summarize_configs",
    "get_expansion_tree",
    "print_expansion_tree",
    "format_config_table",
    "ExpansionTreeNode",
]
