"""Tests for PipelineConfigs reproducibility and hash contracts."""

from __future__ import annotations

import importlib
import logging

from nirs4all.pipeline.config import pipeline_config as pipeline_config_module
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs


def _make_seeded_definition(seed: int) -> dict:
    """Return a minimal pipeline definition with seed-aware generator syntax."""
    return {
        "random_state": seed,
        "pipeline": [
            {
                "model": {
                    "class": "sklearn.linear_model.Ridge",
                    "params": {
                        "alpha": {
                            "_sample_": {
                                "distribution": "uniform",
                                "from": 0.01,
                                "to": 1.0,
                                "num": 8,
                            },
                            "count": 3,
                        }
                    },
                }
            }
        ],
    }


class TestPipelineConfig:
    """Test suite for PipelineConfigs."""

    def test_generator_expansion_is_seeded_via_random_state(self):
        """PipelineConfigs forwards random_state to expand_spec_with_choices()."""
        cfg1 = PipelineConfigs(_make_seeded_definition(42), name="seeded")
        cfg2 = PipelineConfigs(_make_seeded_definition(42), name="seeded")
        cfg3 = PipelineConfigs(_make_seeded_definition(43), name="seeded")

        assert cfg1.steps == cfg2.steps
        assert cfg1.generator_choices == cfg2.generator_choices
        assert cfg1.steps != cfg3.steps

    def test_identity_and_display_hash_are_separate(self):
        """Identity hash is long; display hash is short and name-facing."""
        steps = [{"model": {"class": "sklearn.linear_model.Ridge", "params": {"alpha": 1.0}}}]

        identity_hash = PipelineConfigs.get_hash(steps)
        display_hash = PipelineConfigs.get_display_hash(steps)
        config = PipelineConfigs({"pipeline": steps}, name="ridge_cfg")
        config_display_hash = PipelineConfigs.get_display_hash(config.steps[0])

        assert len(identity_hash) == PipelineConfigs.IDENTITY_HASH_LENGTH
        assert len(display_hash) == PipelineConfigs.DISPLAY_HASH_LENGTH
        assert identity_hash.startswith(display_hash)
        assert config.names[0].endswith(config_display_hash)

    def test_import_has_no_root_logger_side_effects(self):
        """Reloading module must not mutate root logger handlers or level."""
        root = logging.getLogger()
        handlers_before = tuple(root.handlers)
        level_before = root.level

        importlib.reload(pipeline_config_module)

        assert tuple(root.handlers) == handlers_before
        assert root.level == level_before
