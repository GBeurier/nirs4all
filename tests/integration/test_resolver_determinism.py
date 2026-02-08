"""Integration coverage for resolver determinism and mode policy."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import polars as pl
import pytest
import yaml

from nirs4all.pipeline.resolver import PredictionResolver


def _create_pipeline_layout(workspace, run_name: str, pipeline_uid: str) -> str:
    """Create a minimal filesystem layout resolvable by PredictionResolver."""
    run_dir = workspace / "runs" / run_name
    pipeline_dir = run_dir / pipeline_uid
    pipeline_dir.mkdir(parents=True)
    with open(pipeline_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"pipeline_uid": pipeline_uid, "execution_traces": {}}, f)
    with open(pipeline_dir / "pipeline.json", "w", encoding="utf-8") as f:
        json.dump({"steps": [{"model": "PLS"}]}, f)
    return str(run_dir)


def test_structural_branch_path_matching_is_deterministic(tmp_path):
    """Branch-path comparison should be structural, not JSON-string based."""
    workspace = tmp_path / "workspace"
    _create_pipeline_layout(workspace, "2026-02-07_run", "pipe_det")

    resolver = PredictionResolver(workspace)
    chains_df = pl.DataFrame(
        {
            "chain_id": ["chain_other", "chain_target"],
            "pipeline_id": ["pipe_det", "pipe_det"],
            "branch_path": ["[2,1]", "[ 1, 2 ]"],
            "model_step_idx": [2, 2],
            "model_class": ["PLS", "PLS"],
            "preprocessings": ["SNV>SG", "SNV>SG"],
        }
    )
    prediction = {"branch_path": [1, 2], "step_idx": 2, "model_classname": "PLS", "preprocessings": "SNV>SG"}

    assert resolver._pick_chain_for_prediction(chains_df, prediction) == "chain_target"


def test_ambiguous_chain_resolution_warns_and_picks_first(tmp_path):
    """Resolver should warn on ambiguity and pick the first deterministic candidate."""
    workspace = tmp_path / "workspace"
    _create_pipeline_layout(workspace, "2026-02-07_run", "pipe_amb")

    resolver = PredictionResolver(workspace)
    chains_df = pl.DataFrame(
        {
            "chain_id": ["chain_a", "chain_b"],
            "pipeline_id": ["pipe_amb", "pipe_amb"],
            "branch_path": [None, None],
            "model_step_idx": [1, 1],
            "model_class": ["PLS", "PLS"],
            "preprocessings": ["SNV", "SNV"],
        }
    )

    result = resolver._pick_chain_for_prediction(chains_df, {"step_idx": 1, "model_classname": "PLS", "preprocessings": "SNV"})
    assert result is None  # ambiguous → returns None for filesystem fallback


def test_mode_policy_auto_and_store_only(tmp_path):
    """auto mode falls back to filesystem; store-only mode fails on store miss."""
    workspace = tmp_path / "workspace"
    run_dir = _create_pipeline_layout(workspace, "2026-02-07_run", "pipe_mode")

    store = MagicMock()
    store.get_pipeline.return_value = None
    store.get_prediction.return_value = None
    store.get_chains_for_pipeline.return_value = pl.DataFrame({"chain_id": []})

    resolver = PredictionResolver(workspace, store=store)
    prediction = {"pipeline_uid": "pipe_mode", "run_dir": run_dir, "model_name": "PLS"}

    resolved = resolver.resolve(prediction, resolution_mode="auto")
    assert resolved.pipeline_uid == "pipe_mode"

    with pytest.raises(FileNotFoundError):
        resolver.resolve(prediction, resolution_mode="store")
