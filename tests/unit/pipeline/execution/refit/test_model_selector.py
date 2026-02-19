"""Tests for per-model variant selection logic.

Covers:
- Single variant (no generators): all models share variant 0
- Multiple variants with single model
- Multiple variants with multiple models (independent selection)
- Branching without merge (alternative models)
- Branching with merge (stacking context)
- Ascending/descending metric inference
- Missing predictions fallback
- Variant resolution from prediction fields
- Score aggregation across folds
- Edge case: empty predictions list
"""

from __future__ import annotations

import pytest

from nirs4all.pipeline.analysis.topology import (
    ModelNodeInfo,
    PipelineTopology,
    analyze_topology,
)
from nirs4all.pipeline.execution.refit.model_selector import (
    PerModelSelection,
    _aggregate_scores_per_variant,
    _infer_ascending,
    _resolve_variant_index,
    select_best_per_model,
)

# =========================================================================
# Helpers
# =========================================================================

def _make_topology(
    *model_specs: tuple[str, list[int], str | None],
    has_stacking: bool = False,
) -> PipelineTopology:
    """Create a PipelineTopology with given model nodes.

    Each model_spec is (model_class, branch_path, merge_type).
    """
    topo = PipelineTopology(has_stacking=has_stacking)
    for idx, (model_class, branch_path, merge_type) in enumerate(model_specs):
        topo.model_nodes.append(
            ModelNodeInfo(
                model_class=model_class,
                branch_path=branch_path,
                step_index=idx,
                merge_type=merge_type,
                branch_depth=len(branch_path),
            )
        )
    return topo

def _make_predictions(
    entries: list[dict],
) -> list[dict]:
    """Create prediction dicts with standard fields."""
    result = []
    for e in entries:
        pred = {
            "model_name": e.get("model_name", ""),
            "model_classname": e.get("model_classname", ""),
            "val_score": e.get("val_score"),
            "variant_index": e.get("variant_index"),
            "config_name": e.get("config_name", ""),
            "pipeline_uid": e.get("pipeline_uid"),
            "branch_id": e.get("branch_id"),
            "branch_name": e.get("branch_name"),
            "fold_id": e.get("fold_id"),
        }
        result.append(pred)
    return result

def _make_variant_configs(n: int) -> list[dict]:
    """Create n variant config dicts."""
    return [
        {
            "variant_index": i,
            "name": f"variant_{i}",
            "expanded_steps": [{"step": f"step_{i}"}],
            "best_params": {"n_components": 5 + i},
        }
        for i in range(n)
    ]

# =========================================================================
# Single variant
# =========================================================================

class TestSingleVariant:
    """All models share variant 0 when there's only one variant."""

    def test_single_model_single_variant(self):
        """One model, one variant -> variant_index=0."""
        topo = _make_topology(("PLSRegression", [], None))
        configs = _make_variant_configs(1)
        preds = _make_predictions([])  # No predictions needed for single variant

        result = select_best_per_model(preds, topo, configs)

        assert "PLSRegression" in result
        assert result["PLSRegression"].variant_index == 0

    def test_multiple_models_single_variant(self):
        """Multiple models, one variant -> all get variant_index=0."""
        topo = _make_topology(
            ("PLSRegression", [0], "predictions"),
            ("RandomForest", [1], "predictions"),
            ("Ridge", [], None),
        )
        configs = _make_variant_configs(1)
        preds = _make_predictions([])

        result = select_best_per_model(preds, topo, configs)

        assert len(result) == 3
        for sel in result.values():
            assert sel.variant_index == 0

    def test_single_variant_preserves_branch_path(self):
        """Branch path from topology is preserved in selection."""
        topo = _make_topology(("PLSRegression", [0, 1], "predictions"))
        configs = _make_variant_configs(1)
        preds = _make_predictions([])

        result = select_best_per_model(preds, topo, configs)
        assert result["PLSRegression"].branch_path == [0, 1]

# =========================================================================
# Multiple variants with single model
# =========================================================================

class TestMultiVariantSingleModel:
    """Select best variant for a single model across multiple variants."""

    def test_lower_is_better(self):
        """RMSE (ascending=True): lower score is better."""
        topo = _make_topology(("PLSRegression", [], None))
        configs = _make_variant_configs(3)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": 0.5, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.3, "variant_index": 1},
            {"model_name": "PLSRegression", "val_score": 0.4, "variant_index": 2},
        ])

        result = select_best_per_model(preds, topo, configs, metric="rmse", ascending=True)

        assert result["PLSRegression"].variant_index == 1
        assert result["PLSRegression"].best_score == pytest.approx(0.3)

    def test_higher_is_better(self):
        """R2 (ascending=False): higher score is better."""
        topo = _make_topology(("PLSRegression", [], None))
        configs = _make_variant_configs(3)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": 0.7, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.9, "variant_index": 1},
            {"model_name": "PLSRegression", "val_score": 0.8, "variant_index": 2},
        ])

        result = select_best_per_model(preds, topo, configs, metric="r2", ascending=False)

        assert result["PLSRegression"].variant_index == 1
        assert result["PLSRegression"].best_score == pytest.approx(0.9)

    def test_best_params_from_winning_variant(self):
        """Best params come from the winning variant's config."""
        topo = _make_topology(("PLSRegression", [], None))
        configs = _make_variant_configs(3)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": 0.5, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.2, "variant_index": 1},
            {"model_name": "PLSRegression", "val_score": 0.4, "variant_index": 2},
        ])

        result = select_best_per_model(preds, topo, configs, ascending=True)

        # Variant 1 wins, its best_params should be {"n_components": 6}
        assert result["PLSRegression"].best_params == {"n_components": 6}

# =========================================================================
# Multiple variants with multiple models (independent selection)
# =========================================================================

class TestMultiVariantMultiModel:
    """Each model independently selects its best variant."""

    def test_independent_selection(self):
        """Different models can select different best variants."""
        topo = _make_topology(
            ("PLSRegression", [], None),
            ("RandomForest", [], None),
        )
        configs = _make_variant_configs(3)
        preds = _make_predictions([
            # PLS best at variant 2
            {"model_name": "PLSRegression", "val_score": 0.5, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.4, "variant_index": 1},
            {"model_name": "PLSRegression", "val_score": 0.2, "variant_index": 2},
            # RF best at variant 0
            {"model_name": "RandomForest", "val_score": 0.1, "variant_index": 0},
            {"model_name": "RandomForest", "val_score": 0.3, "variant_index": 1},
            {"model_name": "RandomForest", "val_score": 0.5, "variant_index": 2},
        ])

        result = select_best_per_model(preds, topo, configs, ascending=True)

        assert result["PLSRegression"].variant_index == 2
        assert result["RandomForest"].variant_index == 0

    def test_models_with_no_overlap(self):
        """Models only appearing in some variants still get selected."""
        topo = _make_topology(
            ("ModelA", [], None),
            ("ModelB", [], None),
        )
        configs = _make_variant_configs(2)
        preds = _make_predictions([
            {"model_name": "ModelA", "val_score": 0.3, "variant_index": 0},
            # ModelA not present in variant 1
            {"model_name": "ModelB", "val_score": 0.5, "variant_index": 1},
            # ModelB not present in variant 0
        ])

        result = select_best_per_model(preds, topo, configs, ascending=True)

        assert result["ModelA"].variant_index == 0
        assert result["ModelB"].variant_index == 1

# =========================================================================
# Branching scenarios
# =========================================================================

class TestBranching:
    """Branch-related selection scenarios."""

    def test_branches_without_merge_independent(self):
        """Models in different branches without merge are alternatives."""
        topo = _make_topology(
            ("PLSRegression", [0], None),
            ("RandomForest", [1], None),
            has_stacking=False,
        )
        topo.has_branches_without_merge = True
        configs = _make_variant_configs(2)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": 0.3, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.5, "variant_index": 1},
            {"model_name": "RandomForest", "val_score": 0.4, "variant_index": 0},
            {"model_name": "RandomForest", "val_score": 0.2, "variant_index": 1},
        ])

        result = select_best_per_model(preds, topo, configs, ascending=True)

        # PLS best at variant 0 (0.3 < 0.5)
        assert result["PLSRegression"].variant_index == 0
        # RF best at variant 1 (0.2 < 0.4)
        assert result["RandomForest"].variant_index == 1

    def test_stacking_with_meta_model(self):
        """Stacking: meta-model variant is logged."""
        topo = _make_topology(
            ("PLSRegression", [0], "predictions"),
            ("RandomForest", [1], "predictions"),
            ("Ridge", [], None),  # Meta-model
            has_stacking=True,
        )
        configs = _make_variant_configs(2)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": 0.3, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.4, "variant_index": 1},
            {"model_name": "RandomForest", "val_score": 0.2, "variant_index": 0},
            {"model_name": "RandomForest", "val_score": 0.5, "variant_index": 1},
            {"model_name": "Ridge", "val_score": 0.25, "variant_index": 0},
            {"model_name": "Ridge", "val_score": 0.35, "variant_index": 1},
        ])

        result = select_best_per_model(preds, topo, configs, ascending=True)

        assert len(result) == 3
        # Each model selects independently
        assert result["PLSRegression"].variant_index == 0  # 0.3 < 0.4
        assert result["RandomForest"].variant_index == 0  # 0.2 < 0.5
        assert result["Ridge"].variant_index == 0  # 0.25 < 0.35

# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge cases and fallbacks."""

    def test_empty_predictions(self):
        """Empty predictions list: fallback to variant 0."""
        topo = _make_topology(("PLSRegression", [], None))
        configs = _make_variant_configs(3)
        preds = _make_predictions([])

        result = select_best_per_model(preds, topo, configs)

        assert result["PLSRegression"].variant_index == 0

    def test_no_variant_configs(self):
        """Empty variant_configs: handles gracefully."""
        topo = _make_topology(("PLSRegression", [], None))
        result = select_best_per_model([], topo, [])
        assert result["PLSRegression"].variant_index == 0
        assert result["PLSRegression"].expanded_steps == []

    def test_no_model_nodes_in_topology(self):
        """Topology with no model nodes returns empty dict."""
        topo = PipelineTopology()
        configs = _make_variant_configs(2)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": 0.3, "variant_index": 0},
        ])

        result = select_best_per_model(preds, topo, configs)
        assert result == {}

    def test_model_with_none_val_score(self):
        """Predictions with None val_score are skipped."""
        topo = _make_topology(("PLSRegression", [], None))
        configs = _make_variant_configs(2)
        preds = _make_predictions([
            {"model_name": "PLSRegression", "val_score": None, "variant_index": 0},
            {"model_name": "PLSRegression", "val_score": 0.3, "variant_index": 1},
        ])

        result = select_best_per_model(preds, topo, configs, ascending=True)
        assert result["PLSRegression"].variant_index == 1

# =========================================================================
# Score aggregation
# =========================================================================

class TestScoreAggregation:
    """Aggregation of scores across folds."""

    def test_average_across_folds(self):
        """Scores are averaged across folds for each variant."""
        scores = [
            (0, 0.4, {}),
            (0, 0.6, {}),
            (1, 0.3, {}),
            (1, 0.5, {}),
        ]
        result = _aggregate_scores_per_variant(scores)
        assert result[0] == pytest.approx(0.5)  # (0.4 + 0.6) / 2
        assert result[1] == pytest.approx(0.4)  # (0.3 + 0.5) / 2

    def test_single_fold_per_variant(self):
        """Single fold: no averaging needed."""
        scores = [(0, 0.3, {}), (1, 0.5, {})]
        result = _aggregate_scores_per_variant(scores)
        assert result[0] == pytest.approx(0.3)
        assert result[1] == pytest.approx(0.5)

# =========================================================================
# Metric inference
# =========================================================================

class TestMetricInference:
    """Inferring ascending/descending from metric name."""

    def test_rmse_is_ascending(self):
        assert _infer_ascending("rmse") is True

    def test_mse_is_ascending(self):
        assert _infer_ascending("mse") is True

    def test_r2_is_descending(self):
        assert _infer_ascending("r2") is False

    def test_accuracy_is_descending(self):
        assert _infer_ascending("accuracy") is False

    def test_empty_metric_defaults_ascending(self):
        assert _infer_ascending("") is True

    def test_f1_is_descending(self):
        assert _infer_ascending("f1") is False

# =========================================================================
# Variant resolution
# =========================================================================

class TestVariantResolution:
    """Resolving variant_index from prediction fields."""

    def test_explicit_variant_index(self):
        """Explicit variant_index field is used directly."""
        configs = _make_variant_configs(3)
        pred = {"variant_index": 2}
        assert _resolve_variant_index(pred, configs) == 2

    def test_match_by_config_name(self):
        """Falls back to matching by config_name."""
        configs = _make_variant_configs(3)
        pred = {"config_name": "variant_1"}
        assert _resolve_variant_index(pred, configs) == 1

    def test_fallback_to_zero(self):
        """When no match is found, defaults to 0."""
        configs = _make_variant_configs(3)
        pred = {"config_name": "unknown"}
        assert _resolve_variant_index(pred, configs) == 0

    def test_match_by_pipeline_uid(self):
        """Match by pipeline_uid field."""
        configs = [
            {"pipeline_uid": "uid_0", "name": "v0"},
            {"pipeline_uid": "uid_1", "name": "v1"},
        ]
        pred = {"pipeline_uid": "uid_1"}
        assert _resolve_variant_index(pred, configs) == 1
