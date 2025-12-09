"""Generator module for pipeline configuration expansion.

This module expands pipeline configuration specifications into concrete
pipeline variants. It handles combinatorial keywords (_or_, _range_, size,
count, pick, arrange) and generates all possible combinations.

This is the public API module. The implementation is in the _generator
subpackage, which uses a Strategy pattern for modular node handling.

Main Functions:
    expand_spec(node, seed): Expand a configuration node into all variants
    expand_spec_iter(node, seed): Lazy iterator version for large spaces
    count_combinations(node): Count variants without generating them

Keywords:
    _or_: Choice between alternatives
    _range_: Numeric sequence generation
    size: Number of items to select (legacy, uses combinations)
    pick: Unordered selection (combinations) - explicit intent
    arrange: Ordered arrangement (permutations) - explicit intent
    then_pick: Second-order combination selection
    then_arrange: Second-order permutation selection
    count: Limit number of generated variants

    _log_range_: Logarithmic sequence generation
    _grid_: Grid search style Cartesian product
    _zip_: Parallel iteration (like Python's zip)
    _chain_: Sequential ordered choices
    _sample_: Statistical sampling (uniform, log-uniform, normal)
    _tags_: Configuration tagging for filtering
    _metadata_: Arbitrary metadata attachment

    Constraints: _mutex_, _requires_, _exclude_ for filtering combinations
    Presets: _preset_ for named configuration templates
    Iterator: expand_spec_iter for memory-efficient lazy expansion
    Export: to_dataframe, diff_configs, print_expansion_tree utilities

Examples:
    Basic choice expansion:
        >>> expand_spec({"_or_": ["A", "B", "C"]})
        ['A', 'B', 'C']

    Pick (combinations):
        >>> expand_spec({"_or_": ["A", "B", "C"], "pick": 2})
        [['A', 'B'], ['A', 'C'], ['B', 'C']]

    Arrange (permutations):
        >>> expand_spec({"_or_": ["A", "B", "C"], "arrange": 2})
        [['A', 'B'], ['B', 'A'], ['A', 'C'], ['C', 'A'], ['B', 'C'], ['C', 'B']]

    Mutual exclusion constraint (Phase 4):
        >>> expand_spec({"_or_": ["A", "B", "C"], "pick": 2, "_mutex_": [["A", "B"]]})
        [['A', 'C'], ['B', 'C']]  # ["A", "B"] excluded

    Lazy iteration for large spaces (Phase 4):
        >>> for config in expand_spec_iter({"_range_": [1, 1000000]}):
        ...     process(config)  # Memory efficient

    Numeric range:
        >>> expand_spec({"_range_": [1, 5]})
        [1, 2, 3, 4, 5]

    Logarithmic range:
        >>> expand_spec({"_log_range_": [0.001, 1, 4]})
        [0.001, 0.01, 0.1, 1.0]

    Grid search:
        >>> expand_spec({"_grid_": {"x": [1, 2], "y": ["A", "B"]}})
        [{'x': 1, 'y': 'A'}, {'x': 1, 'y': 'B'}, {'x': 2, 'y': 'A'}, {'x': 2, 'y': 'B'}]

    Parallel zip:
        >>> expand_spec({"_zip_": {"x": [1, 2], "y": ["A", "B"]}})
        [{'x': 1, 'y': 'A'}, {'x': 2, 'y': 'B'}]

    Nested dict expansion:
        >>> expand_spec({"x": {"_or_": [1, 2]}, "y": 3})
        [{'x': 1, 'y': 3}, {'x': 2, 'y': 3}]

Architecture:
    The _generator subpackage uses the Strategy pattern:
    - strategies/base.py: ExpansionStrategy abstract base class
    - strategies/registry.py: Strategy registration and dispatch
    - strategies/range_strategy.py: Handles _range_ nodes
    - strategies/or_strategy.py: Handles _or_ nodes with pick/arrange/constraints
    - strategies/log_range_strategy.py: Handles _log_range_ nodes (Phase 3)
    - strategies/grid_strategy.py: Handles _grid_ nodes (Phase 3)
    - strategies/zip_strategy.py: Handles _zip_ nodes (Phase 3)
    - strategies/chain_strategy.py: Handles _chain_ nodes (Phase 3)
    - strategies/sample_strategy.py: Handles _sample_ nodes (Phase 3)
    - validators/schema.py: Specification and config validation (Phase 3)
    - iterator.py: Lazy expansion with expand_spec_iter (Phase 4)
    - constraints.py: Constraint evaluation (_mutex_, _requires_) (Phase 4)
    - presets.py: Preset registry and resolution (Phase 4)
    - core.py: Main expansion logic using strategy dispatch
    - keywords.py: Keyword constants and detection utilities
    - utils/: Helper functions (sampling, combinatorics, export)
"""

# Re-export core API from _generator package
from ._generator.core import expand_spec, expand_spec_with_choices, count_combinations

# Re-export iterator API (Phase 4)
from ._generator.iterator import (  # noqa: F401
    expand_spec_iter,
    batch_iter,
    iter_with_progress,
)

# Re-export keyword constants for external use
from ._generator.keywords import (  # noqa: F401
    # Core keywords
    OR_KEYWORD,
    RANGE_KEYWORD,
    LOG_RANGE_KEYWORD,
    GRID_KEYWORD,
    ZIP_KEYWORD,
    CHAIN_KEYWORD,
    SAMPLE_KEYWORD,
    CARTESIAN_KEYWORD,
    # Modifier keywords
    SIZE_KEYWORD,
    COUNT_KEYWORD,
    SEED_KEYWORD,
    WEIGHTS_KEYWORD,
    # Selection keywords
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    # Metadata keywords
    TAGS_KEYWORD,
    METADATA_KEYWORD,
    # Constraint keywords
    MUTEX_KEYWORD,
    REQUIRES_KEYWORD,
    DEPENDS_ON_KEYWORD,
    EXCLUDE_KEYWORD,
    # Keyword groups
    PURE_OR_KEYS,
    PURE_RANGE_KEYS,
    PURE_LOG_RANGE_KEYS,
    PURE_GRID_KEYS,
    PURE_ZIP_KEYS,
    PURE_CHAIN_KEYS,
    PURE_SAMPLE_KEYS,
    PURE_CARTESIAN_KEYS,
    GENERATION_KEYWORDS,
    SELECTION_KEYWORDS,
    MODIFIER_KEYWORDS,
    METADATA_KEYWORDS,
    CONSTRAINT_KEYWORDS,
    ALL_KEYWORDS,
    # Detection functions
    is_generator_node,
    is_pure_or_node,
    is_pure_range_node,
    is_pure_log_range_node,
    is_pure_grid_node,
    is_pure_zip_node,
    is_pure_chain_node,
    is_pure_sample_node,
    is_pure_cartesian_node,
    has_or_keyword,
    has_range_keyword,
    has_log_range_keyword,
    has_grid_keyword,
    has_zip_keyword,
    has_chain_keyword,
    has_sample_keyword,
    has_cartesian_keyword,
    # Extraction functions
    extract_modifiers,
    extract_base_node,
    extract_or_choices,
    extract_range_spec,
    extract_tags,
    extract_metadata,
    extract_constraints,
)

# Re-export strategies for advanced usage
from ._generator.strategies import (  # noqa: F401
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
    # Phase 4+ strategies
    CartesianStrategy,
)

# Re-export validators (Phase 3)
from ._generator.validators import (  # noqa: F401
    ValidationError,
    ValidationResult,
    ValidationSeverity,
    validate_spec,
    validate_config,
    validate_expanded_configs,
)

# Re-export utilities (used by tests and advanced usage)
from ._generator.utils.sampling import sample_with_seed  # noqa: F401

# Re-export constraints (Phase 4)
from ._generator.constraints import (  # noqa: F401
    apply_mutex_constraint,
    apply_requires_constraint,
    apply_exclude_constraint,
    apply_all_constraints,
    parse_constraints,
    validate_constraints,
)

# Re-export presets (Phase 4)
from ._generator.presets import (  # noqa: F401
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

# Re-export export utilities (Phase 4)
from ._generator.utils.export import (  # noqa: F401
    to_dataframe,
    diff_configs,
    summarize_configs,
    get_expansion_tree,
    print_expansion_tree,
    format_config_table,
    ExpansionTreeNode,
)

__all__ = [
    # Core API
    "expand_spec",
    "expand_spec_with_choices",
    "count_combinations",
    # Iterator API (Phase 4)
    "expand_spec_iter",
    "batch_iter",
    "iter_with_progress",
    # Core keyword constants
    "OR_KEYWORD",
    "RANGE_KEYWORD",
    "LOG_RANGE_KEYWORD",
    "GRID_KEYWORD",
    "ZIP_KEYWORD",
    "CHAIN_KEYWORD",
    "SAMPLE_KEYWORD",
    "CARTESIAN_KEYWORD",
    # Modifier keyword constants
    "SIZE_KEYWORD",
    "COUNT_KEYWORD",
    "SEED_KEYWORD",
    "WEIGHTS_KEYWORD",
    # Selection keyword constants
    "PICK_KEYWORD",
    "ARRANGE_KEYWORD",
    "THEN_PICK_KEYWORD",
    "THEN_ARRANGE_KEYWORD",
    # Metadata keyword constants
    "TAGS_KEYWORD",
    "METADATA_KEYWORD",
    # Constraint keyword constants
    "MUTEX_KEYWORD",
    "REQUIRES_KEYWORD",
    "DEPENDS_ON_KEYWORD",
    "EXCLUDE_KEYWORD",
    # Keyword groups
    "PURE_OR_KEYS",
    "PURE_RANGE_KEYS",
    "PURE_LOG_RANGE_KEYS",
    "PURE_GRID_KEYS",
    "PURE_ZIP_KEYS",
    "PURE_CHAIN_KEYS",
    "PURE_SAMPLE_KEYS",
    "PURE_CARTESIAN_KEYS",
    "GENERATION_KEYWORDS",
    "SELECTION_KEYWORDS",
    "MODIFIER_KEYWORDS",
    "METADATA_KEYWORDS",
    "CONSTRAINT_KEYWORDS",
    "ALL_KEYWORDS",
    # Detection functions
    "is_generator_node",
    "is_pure_or_node",
    "is_pure_range_node",
    "is_pure_log_range_node",
    "is_pure_grid_node",
    "is_pure_zip_node",
    "is_pure_chain_node",
    "is_pure_sample_node",
    "is_pure_cartesian_node",
    "has_or_keyword",
    "has_range_keyword",
    "has_log_range_keyword",
    "has_grid_keyword",
    "has_zip_keyword",
    "has_chain_keyword",
    "has_sample_keyword",
    "has_cartesian_keyword",
    # Extraction functions
    "extract_modifiers",
    "extract_base_node",
    "extract_or_choices",
    "extract_range_spec",
    "extract_tags",
    "extract_metadata",
    "extract_constraints",
    # Strategy pattern
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
    # Phase 4+ strategies
    "CartesianStrategy",
    # Validators
    "ValidationError",
    "ValidationResult",
    "ValidationSeverity",
    "validate_spec",
    "validate_config",
    "validate_expanded_configs",
    # Utilities
    "sample_with_seed",
    # Constraints (Phase 4)
    "apply_mutex_constraint",
    "apply_requires_constraint",
    "apply_exclude_constraint",
    "apply_all_constraints",
    "parse_constraints",
    "validate_constraints",
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
    # Export utilities (Phase 4)
    "to_dataframe",
    "diff_configs",
    "summarize_configs",
    "get_expansion_tree",
    "print_expansion_tree",
    "format_config_table",
    "ExpansionTreeNode",
]
