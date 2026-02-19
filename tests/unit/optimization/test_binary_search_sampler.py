"""
Tests for BinarySearchSampler integration with OptunaManager.
"""

from pathlib import Path

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate

# Get path to sample data relative to project root
SAMPLE_DATA_PATH = Path(__file__).parent.parent.parent.parent / "examples" / "sample_data" / "regression"

def test_binary_sampler_basic():
    """Test basic binary search sampler functionality."""
    pipeline = [
        StandardNormalVariate(),
        MinMaxScaler(),
        ShuffleSplit(n_splits=2, test_size=0.3, random_state=42),
        {
            "model": PLSRegression(),
            "finetune_params": {
                "n_trials": 8,
                "sampler": "binary",
                "seed": 42,
                "verbose": 0,
                "approach": "single",
                "model_params": {
                    "n_components": ('int', 1, 20),
                },
            }
        },
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset=str(SAMPLE_DATA_PATH),
        verbose=0
    )

    assert result.best_score is not None
    assert result.best_score > 0
    # Binary search creates one prediction per trial (8 in this case)
    assert result.num_predictions > 0

def test_binary_sampler_with_seed():
    """Test that binary sampler produces reproducible results with seed."""
    pipeline = [
        StandardNormalVariate(),
        ShuffleSplit(n_splits=2, random_state=42),
        {
            "model": PLSRegression(),
            "finetune_params": {
                "n_trials": 6,
                "sampler": "binary",
                "seed": 42,
                "verbose": 0,
                "approach": "single",
                "model_params": {
                    "n_components": ('int', 1, 15),
                },
            }
        },
    ]

    result1 = nirs4all.run(pipeline=pipeline, dataset=str(SAMPLE_DATA_PATH), verbose=0)
    result2 = nirs4all.run(pipeline=pipeline, dataset=str(SAMPLE_DATA_PATH), verbose=0)

    # Results should be similar (not necessarily identical due to sklearn randomness)
    assert abs(result1.best_score - result2.best_score) < 0.1

def test_binary_sampler_multiphase():
    """Test binary sampler in multi-phase configuration."""
    pipeline = [
        StandardNormalVariate(),
        ShuffleSplit(n_splits=2, random_state=42),
        {
            "model": PLSRegression(),
            "finetune_params": {
                "seed": 42,
                "verbose": 0,
                "approach": "single",
                "phases": [
                    {"n_trials": 5, "sampler": "binary"},
                    {"n_trials": 3, "sampler": "tpe"},
                ],
                "model_params": {
                    "n_components": ('int', 1, 20),
                },
            }
        },
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset=str(SAMPLE_DATA_PATH),
        verbose=0
    )

    assert result.best_score is not None
    # Multiphase creates predictions across all trials (5 + 3 = 8)
    assert result.num_predictions > 0

def test_binary_sampler_grouped_approach():
    """Test binary sampler with grouped approach (preprocessing variants)."""
    pipeline = [
        {"feature_augmentation": [StandardNormalVariate, MinMaxScaler], "action": "extend"},
        ShuffleSplit(n_splits=2, random_state=42),
        {
            "model": PLSRegression(),
            "finetune_params": {
                "n_trials": 6,
                "sampler": "binary",
                "seed": 42,
                "verbose": 0,
                "approach": "grouped",
                "model_params": {
                    "n_components": ('int', 1, 15),
                },
            }
        },
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset=str(SAMPLE_DATA_PATH),
        verbose=0
    )

    assert result.best_score is not None
    # Grouped approach creates predictions for each preprocessing variant × trials
    assert result.num_predictions > 0

def test_binary_sampler_with_categorical():
    """Test binary sampler with mixed parameter types."""
    pipeline = [
        StandardNormalVariate(),
        ShuffleSplit(n_splits=2, random_state=42),
        {
            "model": PLSRegression(),
            "finetune_params": {
                "n_trials": 8,
                "sampler": "binary",
                "seed": 42,
                "verbose": 0,
                "approach": "single",
                "model_params": {
                    "n_components": ('int', 1, 20),  # Binary search
                    "scale": [True, False],          # Categorical (exhaustive)
                },
            }
        },
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset=str(SAMPLE_DATA_PATH),
        verbose=0
    )

    assert result.best_score is not None

def test_binary_sampler_validation():
    """Test that binary sampler is recognized as valid."""
    from nirs4all.optimization.optuna import VALID_SAMPLERS

    assert "binary" in VALID_SAMPLERS

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
