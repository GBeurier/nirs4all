"""
Study Full training Runner - Wrapper for Advanced Mode
====================================================
Provides run_study() function for direct execution with Python objects.
This module wraps study_full_training.py to support both CLI and function call modes.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_study(config: Dict[str, Any]) -> None:
    """Run the full study with advanced configuration.

    This function is called directly from the base runner in advanced mode,
    bypassing CLI argument parsing and allowing Python objects to be passed.

    Args:
        config: Configuration dictionary with all parameters including:
            - folder_list: List of dataset paths
            - aggregation_key_list: List of aggregation keys
            - test_mode: bool
            - global_pp: Optional preprocessing spec dict
            - tabpfn_pp: Optional list of preprocessing pipelines
            - transfer_pp_config: Optional TransferPreprocessingSelector kwargs
            - All other training parameters
    """
    # Import study_full_training main logic
    import bench.studies.study_training as training

    # Override module-level variables with config
    training.TEST_MODE = config.get('test_mode', False)
    training.FOLDER_LIST = config.get('folder_list', [])
    training.AGGREGATION_KEY_LIST = config.get('aggregation_key_list', [])

    # Override GLOBAL_PP if provided
    if config.get('global_pp') is not None:
        training.GLOBAL_PP = config['global_pp']

    # Override TABPFN_PP if provided
    if config.get('tabpfn_pp') is not None:
        training.TABPFN_PP = config['tabpfn_pp']

    # Override training parameters
    training.TRANSFER_PP_PRESET = config.get('transfer_pp_preset', 'balanced')
    training.TRANSFER_PP_SELECTED = config.get('transfer_pp_selected', 10)
    training.PLS_PP_COUNT = config.get('pls_pp_count', 40)
    training.PLS_PP_TOP_SELECTED_COUNT = config.get('pls_pp_top_selected', 10)
    training.PLS_TRIALS = config.get('pls_trials', 20)
    training.OPLS_TRIALS = config.get('opls_trials', 30)
    training.TEST_LW_PLS = config.get('test_lwpls', False)
    training.RIDGE_TRIALS = config.get('ridge_trials', 20)
    training.TABPFN_TRIALS = config.get('tabpfn_trials', 10)
    training.TABPFN_MODEL_VARIANTS = config.get('tabpfn_model_variants', ['default', 'real'])
    training.TABPFN_PP_MAX_COUNT = config.get('tabpfn_pp_max_count', 20)
    training.TABPFN_PP_MAX_SIZE = config.get('tabpfn_pp_max_size', 3)

    # Create a mock args object for compatibility
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    training.args = Args(
        show=config.get('show_plots', False),
        verbose=config.get('verbose', 1),
        device=config.get('device', 'cuda'),
        workspace=config.get('workspace_path', 'wk'),
    )

    # Handle TransferPreprocessingSelector config if provided
    transfer_pp_config = config.get('transfer_pp_config')
    if transfer_pp_config:
        # Store it for use in the pipeline
        training.TRANSFER_PP_CONFIG = transfer_pp_config
    else:
        training.TRANSFER_PP_CONFIG = None

    # Run the main function
    training.main()
