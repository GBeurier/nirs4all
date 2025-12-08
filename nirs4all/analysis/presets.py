"""
Transfer Preprocessing Selection Presets.

This module defines preset configurations for the TransferPreprocessingSelector.
Presets provide convenient defaults for common use cases, from fast exploration
to exhaustive analysis.

Presets:
    fast: Single preprocessing evaluation only (~5s)
    balanced: Stage 1 + 2 with top-5 stacking (~20s)
    thorough: Stage 1 + 2 + 3 with augmentation (~40s)
    full: All stages with supervised validation (~60s)
    exhaustive: Maximum depth/breadth for research (~30min)
"""

from typing import Any, Dict

# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "run_stage2": False,
        "run_stage3": False,
        "run_stage4": False,
        "n_components": 10,
    },
    "balanced": {
        "run_stage2": True,
        "stage2_top_k": 5,
        "stage2_max_depth": 2,
        "run_stage3": False,
        "run_stage4": False,
        "n_components": 10,
    },
    "thorough": {
        "run_stage2": True,
        "stage2_top_k": 5,
        "stage2_max_depth": 2,
        "run_stage3": True,
        "stage3_top_k": 5,
        "stage3_max_order": 2,
        "run_stage4": False,
        "n_components": 10,
    },
    "full": {
        "run_stage2": True,
        "stage2_top_k": 10,
        "stage2_max_depth": 2,
        "run_stage3": True,
        "stage3_top_k": 10,
        "stage3_max_order": 2,
        "run_stage4": True,
        "stage4_top_k": 10,
        "stage4_cv_folds": 3,
        "n_components": 10,
    },
    "exhaustive": {
        "run_stage2": True,
        "stage2_top_k": None,  # All candidates
        "stage2_max_depth": 3,
        "stage2_exhaustive": True,
        "run_stage3": True,
        "stage3_top_k": 100,
        "stage3_max_order": 3,
        "run_stage4": True,
        "stage4_top_k": 50,
        "stage4_cv_folds": 5,
        "n_components": 30,
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name ('fast', 'balanced', 'thorough', 'full', 'exhaustive').

    Returns:
        Dictionary of configuration parameters.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {name}. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name].copy()


def list_presets() -> Dict[str, str]:
    """
    List available presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return {
        "fast": "Single preprocessing evaluation only (~5s)",
        "balanced": "Stage 1 + 2 with top-5 stacking (~20s)",
        "thorough": "Stage 1 + 2 + 3 with augmentation (~40s)",
        "full": "All stages with supervised validation (~60s)",
        "exhaustive": "Maximum depth/breadth for research (~30min)",
    }
